// [Z,Y,X]_THREADS is the number of participating threads in the z, y, x
// dimension of the block. If set to 0 it means that dimension doesn't
// participate, otherwise it is the number of threads. We could start with warp
// reductions, then reduce the warps, this could save some shared memory, but
// may actually be slower.
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
    typename _dim3ti,
    typename _dim3bd>
__device__ void blockReduce(
    T& out,
    const T inp_val,
    Func reduction_op,
    const _dim3ti& thread_idx,
    const _dim3bd& block_dim,
    T* shared_mem,
    bool read_write_pred,
    T init_val) {
  unsigned int reduction_size = (X_REDUCE ? block_dim.x : 1) *
      (Y_REDUCE ? block_dim.y : 1) * (Z_REDUCE ? block_dim.z : 1);

  // If this thread will output a final result
  bool should_write = true;

  if (X_REDUCE)
    should_write = should_write && thread_idx.x == 0;
  if (Y_REDUCE)
    should_write = should_write && thread_idx.y == 0;
  if (Z_REDUCE)
    should_write = should_write && thread_idx.z == 0;

  unsigned int reduction_stride;
  unsigned int reduction_tid;
  unsigned int linear_tid;

  if (X_REDUCE && !Y_REDUCE && Z_REDUCE) {
    // Transpose Z and Y in the shared memory so Z and X dims are contiguous in
    // smem
    reduction_stride = 1;
    linear_tid = threadIdx.y * blockDim.z * blockDim.x +
        threadIdx.z * blockDim.x + threadIdx.x;
    reduction_tid = threadIdx.z * blockDim.x + threadIdx.x;
  } else {
    // Normal reduction in order
    reduction_stride =
        (X_REDUCE ? 1
                  : (Y_REDUCE ? block_dim.x
                              : (Z_REDUCE ? block_dim.x * block_dim.y : 0)));

    linear_tid = thread_idx.z * block_dim.y * block_dim.x +
        thread_idx.y * block_dim.x + thread_idx.x;

    reduction_tid = (Z_REDUCE ? thread_idx.z : 0) *
            (Y_REDUCE ? block_dim.y : 1) * (X_REDUCE ? block_dim.x : 1) +
        (Y_REDUCE ? thread_idx.y : 0) * (X_REDUCE ? block_dim.x : 1) +
        (X_REDUCE ? thread_idx.x : 0);
  }

  assert(reduction_stride != 0);

  if (read_write_pred) {
    shared_mem[linear_tid] = inp_val;
  } else {
    shared_mem[linear_tid] = init_val;
  }
  __syncthreads();
  // Reduce down to nearest power of 2:
  int np2 = 1 << (31 - __clz(reduction_size));

  if (reduction_tid < np2) {
    if (reduction_tid + np2 < reduction_size) {
      reduction_op(
          shared_mem[linear_tid],
          shared_mem[linear_tid + np2 * reduction_stride]);
    }
  }
  __syncthreads();
  // for (int factor = np2/2; factor > contig_threads / 2; factor>>=1) {
  for (int factor = np2 / 2; factor > 0; factor >>= 1) {
    if (reduction_tid < factor) {
      reduction_op(
          shared_mem[linear_tid],
          shared_mem[linear_tid + factor * reduction_stride]);
    }
    __syncthreads();
  }

  if (should_write && read_write_pred)
    out = shared_mem[linear_tid];
}
