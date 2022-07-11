// [Z,Y,X]_THREADS is the number of participating threads in the z, y, x
// dimension of the block. If set to false the dimension doesn't
// participate in the reduction. We could start with warp reductions, then
// reduce the warps, this could save some shared memory, but could be slower in
// some instances.
//
//  EXAMPLE USAGE:
//  blockReduceSum<X_THREADS, Y_THREADS, Z_THREADS>
//    (output[output_index], inputs[input_index],
//      [] __device__ (T& a, const T b) { a += b; });
//
// Note: We agressively template functions taking dim3 in the functions below
//       because ROCM uses different types for the various dim3 and maps them
//       directly to intrinsics, but they're dim3 when used after modification.
//
template <
    bool X_REDUCE,
    bool Y_REDUCE,
    bool Z_REDUCE,
    typename T,
    typename Func,
    typename _dim3,
    typename _dim3_2>
__device__ void blockReduce(
    T& out,
    const T& inp_val,
    Func reduction_op,
    const _dim3& thread_idx,
    const _dim3_2& block_dim,
    T* shared_mem,
    bool read_pred,
    bool write_pred,
    T init_val) {
  // If this thread will output a final result
  bool should_write =
      index_utils::maskedIsZero<X_REDUCE, Y_REDUCE, Z_REDUCE>(thread_idx);

  // Size of the reduction segments
  unsigned int reduction_size =
      index_utils::maskedSize<X_REDUCE, Y_REDUCE, Z_REDUCE>(block_dim);

  // Index into the reduction segment
  unsigned int reduction_tid =
      index_utils::maskedOffset<X_REDUCE, Y_REDUCE, Z_REDUCE>(
          thread_idx, block_dim);

  // Index of the reduction segment
  unsigned int reduction_idx =
      index_utils::maskedOffset<!X_REDUCE, !Y_REDUCE, !Z_REDUCE>(
          thread_idx, block_dim);

  // Offset into smem for the current thread
  unsigned int smem_offset = reduction_idx * reduction_size + reduction_tid;

  // Initialize shared memory
  if (read_pred) {
    shared_mem[smem_offset] = inp_val;
  } else {
    shared_mem[smem_offset] = init_val;
  }

  block_sync::sync();
  // Reduce down to nearest power of 2 for the tree reduction:
  int np2 = 1 << (31 - __clz(reduction_size));

  if (reduction_tid < np2 && reduction_tid + np2 < reduction_size) {
    reduction_op(shared_mem[smem_offset], shared_mem[smem_offset + np2]);
  }
  block_sync::sync();

  // loop peel the final iteration to save one syncthread for the end
  for (int factor = np2 / 2; factor > 1; factor >>= 1) {
    if (reduction_tid < factor) {
      reduction_op(shared_mem[smem_offset], shared_mem[smem_offset + factor]);
    }
    block_sync::sync();
  }

  if (should_write && write_pred) {
    T result = out;
    reduction_op(result, shared_mem[smem_offset]);
    if (reduction_size > 1) {
      reduction_op(result, shared_mem[smem_offset + 1]);
    }
    out = result;
  }
  block_sync::sync();
}

// Use the same pred for both reads and writes
template <
    bool X_REDUCE,
    bool Y_REDUCE,
    bool Z_REDUCE,
    typename T,
    typename Func,
    typename _dim3,
    typename _dim3_2>
__device__ void blockReduce(
    T& out,
    const T& inp_val,
    Func reduction_op,
    const _dim3& thread_idx,
    const _dim3_2& block_dim,
    T* shared_mem,
    bool read_write_pred,
    T init_val) {
  blockReduce<X_REDUCE, Y_REDUCE, Z_REDUCE, T, Func, _dim3, _dim3_2>(
      out,
      inp_val,
      reduction_op,
      thread_idx,
      block_dim,
      shared_mem,
      read_write_pred,
      read_write_pred,
      init_val);
}
