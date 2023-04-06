namespace grid_broadcast {

// Broadcasts per-thread values across threads and blocks.
//
// Function parameters:
// - out: Per-thread output location
// - inp_val: Per-thread input value
// - work_buf: Temporary buffer for communication across threads/blocks
// - sync_flags: A vector of integers for synchronizations
//
// Template parameters:
// - X/Y/Z_BLOCK: When true, broadcasts across thread blocks along the X/Y/Z
//   dimensions
// - X/Y/Z_THREAD: When true, broadcasts across threads along the X/Y/Z
//   dimensions
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    typename T>
__device__ void broadcast(
    T& out,
    const T& inp_val,
    volatile T* work_buf,
    Tensor<int64_t, 1> sync_flags,
    bool read_write_pred) {
  // Number of values broadcasted in the grid dimensions
  const auto grid_seg_size =
      index_utils::maskedSize<X_BLOCK, Y_BLOCK, Z_BLOCK>(gridDim);

  // Index of the broadcast we're performing out of the grid_seg_size
  const auto grid_seg_idx =
      index_utils::maskedOffset<!X_BLOCK, !Y_BLOCK, !Z_BLOCK>(
          blockIdx, gridDim);

  // Number of threads not participating in a broadcast dimension, this is the
  // number of thread entries to expect in the work buffer, therefore a striding
  const auto block_stride =
      index_utils::maskedSize<!X_THREAD, !Y_THREAD, !Z_THREAD>(blockDim);

  // Which broadcast in the block this is to line up the entry with the work
  // buffer
  const auto thread_offset =
      index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
          threadIdx, blockDim);

  const bool has_valid_data = (!X_BLOCK || blockIdx.x == gridDim.x - 1) &&
      (!Y_BLOCK || blockIdx.y == gridDim.y - 1) &&
      (!Z_BLOCK || blockIdx.z == gridDim.z - 1) &&
      (!X_THREAD || threadIdx.x == 0) && (!Y_THREAD || threadIdx.y == 0) &&
      (!Z_THREAD || threadIdx.z == 0);

  if (has_valid_data && read_write_pred) {
    work_buf[grid_seg_idx * block_stride + thread_offset] = inp_val;
    __threadfence();
  }

  grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, true>(
      sync_flags[grid_seg_idx], grid_seg_size);

  if (read_write_pred) {
    out = work_buf[grid_seg_idx * block_stride + thread_offset];
  }

  // Make sure everyone has read from the buffer before continuing the kernel
  // and potentially overwriting
  grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, true>(
      sync_flags[grid_seg_idx], grid_seg_size);
}
} // namespace grid_broadcast
