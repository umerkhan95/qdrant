use std::borrow::Cow;
use std::marker::PhantomData;
use std::mem;
use std::ops::Range;
use std::sync::Arc;

use super::controller::CacheRead;
use super::{BLOCK_SIZE, BlockId, BlockOffset, BlockRequest, CacheController, FileId};
use crate::universal_io;
use crate::universal_io::io_uring::IoUringFile;

/// Typed view over a cached file, simulating a `&[T]` backed by the block cache.
///
/// Internally maps element ranges into fixed-size blocks and fetches them
/// through the [`CacheController`]. Blocks that are already cached are returned
/// as zero-copy borrows from the mmap; multi-block reads allocate a `Vec<T>`
/// (not `Vec<u8>`) so alignment is always correct.
pub struct CachedSlice<T> {
    /// The id assigned by the controller for this file.
    file_id: FileId,

    /// Length of this file in bytes.
    len_bytes: usize,

    /// The controller backing this structure.
    controller: Arc<CacheController<IoUringFile>>,

    r#type: PhantomData<T>,
}

impl<T: bytemuck::Pod> CachedSlice<T> {
    /// Open a file through the cache controller and return a typed view over it.
    pub fn open(
        controller: &Arc<CacheController<IoUringFile>>,
        path: &std::path::Path,
    ) -> universal_io::Result<Self> {
        let (file_id, len) = controller.open_file(path)?;
        Ok(Self {
            file_id,
            len_bytes: len,
            controller: Arc::clone(controller),
            r#type: PhantomData,
        })
    }

    /// Get a Cow reference to a range of elements within the file.
    ///
    /// The `range` is in **elements of T**, not bytes. For `T = u8` this is
    /// equivalent to a byte range.
    ///
    /// If the range is contained in a single block, it will return a borrowed
    /// reference into the mmap. Otherwise, it will allocate a `Vec<T>` and copy
    /// block data into it. Allocating as `Vec<T>` (rather than `Vec<u8>`)
    /// guarantees correct alignment for any `T`.
    pub fn get_range(&self, range: Range<usize>) -> universal_io::Result<Cow<'_, [T]>> {
        let t_size = mem::size_of::<T>();
        debug_assert!(t_size != 0, "cannot use zero-sized type");

        let total_elements = range.end - range.start;
        if total_elements == 0 {
            return Ok(Cow::Borrowed(&[]));
        }

        let byte_range = range.start * t_size..range.end * t_size;
        let mut blocks_iter = self.blocks_for(byte_range);

        // TODO(perf): if blocks are consecutive in the big cache file, we can potentially return without allocating.
        if blocks_iter.len() == 1 {
            let req = blocks_iter.next().expect("We just checked len() == 1");
            let result = self.controller.get_from_cache(req, |bytes| {
                let mut vec_t = vec![T::zeroed(); bytes.len() / t_size];
                bytemuck::cast_slice_mut::<T, u8>(&mut vec_t).copy_from_slice(bytes);
                vec_t
            })?;

            return Ok(match result {
                CacheRead::Hit(bytes) => Cow::Borrowed(bytemuck::cast_slice(bytes)),
                CacheRead::Miss(vec_t) => Cow::Owned(vec_t),
            });
        }

        // Multi-block: delegate to the batch path which submits all
        // cold-storage reads together via io_uring.
        let mut result = None;
        self.get_range_batch(std::iter::once(range), |_idx, buf| {
            result = Some(buf.to_vec());
            Ok(())
        })?;
        Ok(Cow::Owned(result.expect("callback was called")))
    }

    /// Batch version of [`get_range`](Self::get_range).
    ///
    /// All block reads across every range are submitted together via a single
    /// `get_from_cache_batch` call, so cold-storage I/O is batched through
    /// io_uring.
    ///
    /// `callback(range_idx, &[T])` is called once per input range with the
    /// fully assembled data for that range.
    pub fn get_range_batch(
        &self,
        ranges: impl IntoIterator<Item = Range<usize>>,
        mut callback: impl FnMut(usize, &[T]) -> universal_io::Result<()>,
    ) -> universal_io::Result<()> {
        let t_size = mem::size_of::<T>();
        debug_assert!(t_size != 0, "cannot use zero-sized type");

        // Flatten all ranges into a single list of block requests, tracking
        // which input range each block belongs to and where within that
        // range's output buffer it should be written.
        struct BlockMeta {
            range_idx: usize,
            /// Byte offset within this range's output buffer.
            dest_offset: usize,
        }
        let mut all_blocks: Vec<BlockRequest> = Vec::new();
        let mut block_meta: Vec<BlockMeta> = Vec::new();

        // Per-range: true if the range spans a single block and can be served
        // directly from the cache callback without an intermediate buffer.
        let mut range_is_single_block: Vec<bool> = Vec::new();
        // Only allocated for multi-block ranges; single-block ranges get None.
        let mut buffers: Vec<Option<Vec<T>>> = Vec::new();

        for (range_idx, range) in ranges.into_iter().enumerate() {
            let total_elements = range.end - range.start;

            if total_elements == 0 {
                range_is_single_block.push(false);
                buffers.push(None);
                continue;
            }

            let byte_range = range.start * t_size..range.end * t_size;
            let blocks = self.blocks_for(byte_range);
            let single_block = blocks.len() == 1;
            range_is_single_block.push(single_block);

            if single_block {
                buffers.push(None);
            } else {
                buffers.push(Some(vec![T::zeroed(); total_elements]));
            }

            let mut dest_offset = 0;
            for block in blocks {
                block_meta.push(BlockMeta {
                    range_idx,
                    dest_offset,
                });
                dest_offset += block.range.len();
                all_blocks.push(block);
            }
        }

        if all_blocks.is_empty() {
            return Ok(());
        }

        let mut callback_err: Option<universal_io::UniversalIoError> = None;

        self.controller
            .get_from_cache_batch(all_blocks, |block_idx, slice| {
                if callback_err.is_some() {
                    return;
                }

                let meta = &block_meta[block_idx];
                let range_idx = meta.range_idx;

                if range_is_single_block[range_idx] {
                    // Single-block range: pass the slice directly to the caller,
                    // avoiding an intermediate buffer allocation.
                    if let Err(e) = callback(range_idx, bytemuck::cast_slice(slice)) {
                        callback_err = Some(e);
                    }
                } else {
                    // Multi-block range: scatter into the output buffer.
                    let buf = buffers[range_idx]
                        .as_mut()
                        .expect("multi-block range has a buffer");
                    let buf_bytes = bytemuck::cast_slice_mut::<T, u8>(buf);
                    let start = meta.dest_offset;
                    buf_bytes[start..start + slice.len()].copy_from_slice(slice);
                }
            })?;

        if let Some(err) = callback_err {
            return Err(err);
        }

        // Deliver completed multi-block buffers.
        for (range_idx, buf) in buffers.into_iter().enumerate() {
            if let Some(buf) = buf {
                callback(range_idx, &buf)?;
            }
        }

        Ok(())
    }

    #[cfg(test)]
    pub fn get(&self, idx: usize) -> universal_io::Result<Cow<'_, T>> {
        let slice = self.get_range(idx..idx + 1)?;

        let cow = match slice {
            Cow::Borrowed(slice) => Cow::Borrowed(&slice[0]),
            Cow::Owned(mut vec) => Cow::Owned(vec.pop().unwrap()),
        };

        Ok(cow)
    }

    #[expect(clippy::len_without_is_empty)] // Doesn't make sense to cache 0-length files
    pub fn len(&self) -> usize {
        self.len_bytes / mem::size_of::<T>()
    }

    /// Returns the block descriptor for the provided bytes range.
    fn blocks_for(&self, bytes_range: Range<usize>) -> impl ExactSizeIterator<Item = BlockRequest> {
        debug_assert!(bytes_range.start <= bytes_range.end);
        debug_assert!(bytes_range.end <= self.len_bytes);
        debug_assert!(!bytes_range.is_empty(), "empty range would underflow");

        blocks_for_range_in_file(self.file_id, bytes_range)
    }
}

// Extracted to make testing simpler
#[inline(always)]
fn blocks_for_range_in_file(
    file_id: FileId,
    bytes_range: Range<usize>,
) -> impl ExactSizeIterator<Item = BlockRequest> {
    let first_block = bytes_range.start / BLOCK_SIZE;
    let leading_offset = bytes_range.start - (first_block * BLOCK_SIZE);
    let last_block = (bytes_range.end - 1) / BLOCK_SIZE;
    let trailing_offset = bytes_range.end - (last_block * BLOCK_SIZE);

    // Not a RangeInclusive (..=) because it doesn't implement ExactSizeIterator
    (first_block..last_block + 1).map(move |block_offset| {
        let block_id = BlockId {
            file_id,
            offset: BlockOffset(
                u32::try_from(block_offset).expect("file too large for block cache (>70 TiB)"),
            ),
        };

        let range_start = if block_offset == first_block {
            leading_offset
        } else {
            0
        };

        let range_end = if block_offset == last_block {
            trailing_offset
        } else {
            BLOCK_SIZE
        };

        BlockRequest {
            key: block_id,
            range: range_start..range_end,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disk_cache::BLOCK_SIZE;

    #[test]
    fn test_block_request_calculation() {
        let file_id = FileId(0);

        // 10 full blocks and 100 extra bytes in last block
        let file_len = BLOCK_SIZE * 10 + 100;

        //     block 0
        // |                          ... |
        //  < range >
        let blocks: Vec<_> = blocks_for_range_in_file(file_id, 0..100).collect();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].key.offset.0, 0);
        assert_eq!(blocks[0].range, 0..100);

        //       block 0          block 1
        // |                |                 |
        //              < range >
        let blocks: Vec<_> =
            blocks_for_range_in_file(file_id, BLOCK_SIZE - 50..BLOCK_SIZE + 50).collect();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].key.offset.0, 0);
        assert_eq!(blocks[0].range, (BLOCK_SIZE - 50)..BLOCK_SIZE);
        assert_eq!(blocks[1].key.offset.0, 1);
        assert_eq!(blocks[1].range, 0..50);

        //     block 2      block 3
        // |            |             |
        // <          range           >
        let blocks: Vec<_> =
            blocks_for_range_in_file(file_id, BLOCK_SIZE * 2..BLOCK_SIZE * 4).collect();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].key.offset.0, 2);
        assert_eq!(blocks[0].range, 0..BLOCK_SIZE);
        assert_eq!(blocks[1].key.offset.0, 3);
        assert_eq!(blocks[1].range, 0..BLOCK_SIZE);

        //  block 9  (last full block)   block 10 (partial block with trailing data)
        // |                           |         000000000000000000000|
        //    <         range                   >
        let blocks: Vec<_> =
            blocks_for_range_in_file(file_id, BLOCK_SIZE * 9 + 50..file_len).collect();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].key.offset.0, 9);
        assert_eq!(blocks[0].range, 50..BLOCK_SIZE);
        assert_eq!(blocks[1].key.offset.0, 10);
        assert_eq!(blocks[1].range, 0..100);
    }
}
