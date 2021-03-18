// -----------------------------------------------------------------------------------------------
//  Block Welford Primitives
// -----------------------------------------------------------------------------------------------
// Basic utility for welford update. Can be used to scan one value, or two merge
// two welford results
template <typename T, typename TN>
__inline__ __device__ void welfordCombine(
    T& a_M2,
    T& a_avg,
    TN& a_N,
    const T& b_M2,
    const T& b_avg,
    TN b_N) {
  if (b_N == 0) {
    return;
  }
  TN ab_N = a_N + b_N;
  T delta = b_avg - a_avg;
  a_avg += delta * b_N / ab_N;
  a_M2 += b_M2 + delta * delta * a_N * b_N / ab_N;
  a_N = ab_N;
}

// [Z,Y,X]_THREADS is the number of participating threads in the z, y, x
// dimension of the block.
template <
    bool X_REDUCE,
    bool Y_REDUCE,
    bool Z_REDUCE,
    typename T,
    typename TN,
    typename _dim3ti,
    typename _dim3bd>
__inline__ __device__ void blockWelford(
    T& out_M2,
    T& out_avg,
    TN& out_N,
    const T& in_M2,
    const T& in_avg,
    const TN& in_N,
    const _dim3ti& thread_idx,
    const _dim3bd& block_dim,
    T* shared_mem_M2,
    T* shared_mem_avg,
    TN* shared_mem_N,
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
    shared_mem_M2[linear_tid] = in_M2;
    shared_mem_avg[linear_tid] = in_avg;
    shared_mem_N[linear_tid] = in_N;
  } else {
    shared_mem_M2[linear_tid] = init_val;
    shared_mem_avg[linear_tid] = init_val;
    shared_mem_N[linear_tid] = 0;
  }
  __syncthreads();
  // Reduce down to nearest power of 2:
  int np2 = 1 << (31 - __clz(reduction_size));
  if (reduction_tid < np2) {
    if (reduction_tid + np2 < reduction_size) {
      welfordCombine(
          shared_mem_M2[linear_tid],
          shared_mem_avg[linear_tid],
          shared_mem_N[linear_tid],
          shared_mem_M2[linear_tid + np2 * reduction_stride],
          shared_mem_avg[linear_tid + np2 * reduction_stride],
          shared_mem_N[linear_tid + np2 * reduction_stride]);
    }
  }
  __syncthreads();

  // loop peel the final iteration to save one syncthread for the end
  for (int factor = np2 / 2; factor > 1; factor >>= 1) {
    if (reduction_tid < factor) {
      welfordCombine(
          shared_mem_M2[linear_tid],
          shared_mem_avg[linear_tid],
          shared_mem_N[linear_tid],
          shared_mem_M2[linear_tid + factor * reduction_stride],
          shared_mem_avg[linear_tid + factor * reduction_stride],
          shared_mem_N[linear_tid + factor * reduction_stride]);
    }
    __syncthreads();
  }
  if (should_write && read_write_pred) {
    T res_M2 = out_M2;
    T res_avg = out_avg;
    TN res_N = out_N;
    welfordCombine(
        res_M2,
        res_avg,
        res_N,
        shared_mem_M2[linear_tid],
        shared_mem_avg[linear_tid],
        shared_mem_N[linear_tid]);
    if (reduction_size > 1) {
      welfordCombine(
          res_M2,
          res_avg,
          res_N,
          shared_mem_M2[linear_tid + reduction_stride],
          shared_mem_avg[linear_tid + reduction_stride],
          shared_mem_N[linear_tid + reduction_stride]);
    }
    out_M2 = res_M2;
    out_avg = res_avg;
    out_N = res_N;
  }
  __syncthreads();
}
// -----------------------------------------------------------------------------------------------
//  Grid Welford Prototype
// -----------------------------------------------------------------------------------------------
namespace welford {
// Utility functions
template <typename _dim3>
__host__ __device__ __forceinline__ size_t size(const _dim3& d) {
  return (size_t)d.x * (size_t)d.y * (size_t)d.z;
}

#define isize(d) d.x* d.y* d.z

template <typename _dim3pos, typename _dim3dim>
__host__ __device__ __forceinline__ size_t
offset(const _dim3pos& pos, const _dim3dim& dim) {
  return (size_t)pos.x + (size_t)pos.y * (size_t)dim.x +
      (size_t)pos.z * (size_t)dim.x * (size_t)dim.y;
}

#define ioffset(pos, dim) pos.x + pos.y* dim.x + pos.z* dim.x* dim.y

// Returns dim3 of each reduction segment.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, typename _dim3>
__host__ __device__ dim3 dimension_of_reduction_segment(const _dim3& grid_dim) {
  return dim3{
      X_BLOCK ? grid_dim.x : 1,
      Y_BLOCK ? grid_dim.y : 1,
      Z_BLOCK ? grid_dim.z : 1};
}

// Returns the number of blocks in each reduction segment.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, typename _dim3>
__host__ __device__ size_t size_of_reduction_segment(const _dim3& grid_dim) {
  return size(
      dimension_of_reduction_segment<X_BLOCK, Y_BLOCK, Z_BLOCK>(grid_dim));
}

// Returns the total number of reduction segments.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, typename _dim3>
__host__ __device__ size_t number_of_reduction_segments(const _dim3& grid_dim) {
  return (X_BLOCK ? 1 : grid_dim.x) * (Y_BLOCK ? 1 : grid_dim.y) *
      (Z_BLOCK ? 1 : grid_dim.z);
}

// Returns the 1-D index of the segment of thread block of block_idx.
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    typename _dim3bi,
    typename _dim3gd>
__host__ __device__ size_t
index_of_reduction_segment(const _dim3bi& block_idx, const _dim3gd& grid_dim) {
  size_t seg_idx = 0;
  if (!Z_BLOCK)
    seg_idx += block_idx.z;
  if (!Y_BLOCK)
    seg_idx = seg_idx * grid_dim.y + block_idx.y;
  if (!X_BLOCK)
    seg_idx = seg_idx * grid_dim.x + block_idx.x;
  return seg_idx;
}

// Returns the offset of thread block in its reduction segment.
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    typename _dim3bi,
    typename _dim3gd>
__host__ __device__ size_t
offset_in_reduction_segment(const _dim3bi& block_idx, const _dim3gd& grid_dim) {
  size_t offset = 0;
  if (Z_BLOCK)
    offset = offset * grid_dim.z + block_idx.z;
  if (Y_BLOCK)
    offset = offset * grid_dim.y + block_idx.y;
  if (X_BLOCK)
    offset = offset * grid_dim.x + block_idx.x;
  return offset;
}

// Returns dim3 of each reduction block.
template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD, typename _dim3>
__host__ __device__ dim3 dimension_of_reduction_block(const _dim3& block_dim) {
  return dim3{
      X_THREAD ? block_dim.x : 1,
      Y_THREAD ? block_dim.y : 1,
      Z_THREAD ? block_dim.z : 1};
}

// Returns the number of threads of each reduction block.
template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD, typename _dim3>
__host__ __device__ int size_of_reduction_block(const _dim3& block_dim) {
  auto tmp_dim =
      dimension_of_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(block_dim);
  return isize(tmp_dim);
}

// Returns the linear offset of a thread in a reduction block.
template <
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    typename _dim3ti,
    typename _dim3bd>
__host__ __device__ int offset_in_reduction_block(
    const _dim3ti& thread_idx,
    const _dim3bd& block_dim) {
  int offset = 0;
  if (Z_THREAD)
    offset += thread_idx.z;
  if (Y_THREAD)
    offset = offset * block_dim.y + thread_idx.y;
  if (X_THREAD)
    offset = offset * block_dim.x + thread_idx.x;
  return offset;
}

template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD, typename T, typename TN>
__device__ void gridWelfordLastBlock(
    T& out_M2,
    T& out_avg,
    TN& out_N,
    const T* in_M2,
    const T* in_avg,
    const TN* in_N,
    const size_t in_size,
    T* shared_buf_M2,
    T* shared_buf_avg,
    TN* shared_buf_N,
    bool read_write_pred,
    T init_val) {
  const int tid = ioffset(threadIdx, blockDim);
  const int block_size = isize(blockDim);
  const int rblock_size =
      size_of_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(blockDim);

  T inp_M2 = init_val;
  T inp_avg = init_val;
  TN inp_N = 0;
  if (tid < in_size) {
    inp_M2 = in_M2[tid];
    inp_avg = in_avg[tid];
    inp_N = in_N[tid];
  }
  for (size_t i = tid + block_size; i < in_size; i += block_size) {
    welfordCombine(inp_M2, inp_avg, inp_N, in_M2[i], in_avg[i], in_N[i]);
  }
  const auto should_write = (X_THREAD || threadIdx.x == 0) &&
      (Y_THREAD || threadIdx.y == 0) && (Z_THREAD || threadIdx.z == 0);

  auto rem_size = block_size / rblock_size;

  if (rem_size > 1) {
    const int rblock_offset = tid % rblock_size;
    const int rblock_idx = tid / rblock_size;
    T inp_M2_tmp = init_val;
    T inp_avg_tmp = init_val;
    TN inp_N_tmp = 0;
    blockWelford<false, true, false>(
        inp_M2_tmp,
        inp_avg_tmp,
        inp_N_tmp,
        inp_M2,
        inp_avg,
        inp_N,
        dim3{(unsigned)rblock_offset, (unsigned)rblock_idx, 0},
        dim3{(unsigned)rblock_size, (unsigned)rem_size},
        shared_buf_M2,
        shared_buf_avg,
        shared_buf_N,
        true,
        init_val);
    __syncthreads();
    if (tid < rblock_size) {
      shared_buf_M2[tid] = inp_M2_tmp;
      shared_buf_avg[tid] = inp_avg_tmp;
      shared_buf_N[tid] = inp_N_tmp;
    }
    __syncthreads();
    if (should_write) {
      size_t offset_write =
          offset_in_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(
              threadIdx, blockDim);
      inp_M2 = shared_buf_M2[offset_write];
      inp_avg = shared_buf_avg[offset_write];
      inp_N = shared_buf_N[offset_write];
    }
  }

  if (should_write && read_write_pred) {
    welfordCombine(out_M2, out_avg, out_N, inp_M2, inp_avg, inp_N);
  }
}

//    Grid welford combine
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    typename T,
    typename TN>
__device__ bool gridWelford(
    T& out_M2,
    T& out_avg,
    TN& out_N,
    const T& inp_M2,
    const T& inp_avg,
    const TN& inp_N,
    volatile T* work_buf_M2,
    volatile T* work_buf_avg,
    volatile TN* work_buf_N,
    Tensor<int64_t, 1> sync_flags,
    T* shared_buf_M2,
    T* shared_buf_avg,
    TN* shared_buf_N,
    bool read_write_pred,
    T init_val) {
  // Number of values to reduce in the grid dimensions
  const auto seg_size =
      size_of_reduction_segment<X_BLOCK, Y_BLOCK, Z_BLOCK>(gridDim);

  // Index of the reduction we're performing out of the seg_size
  const auto seg_idx =
      index_of_reduction_segment<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);

  // Number of threads we can use in final reduction, Seems to assume all
  // threads in the block participate
  const auto rblock_size =
      size_of_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(blockDim);

  work_buf_M2 += seg_idx * seg_size * rblock_size;
  work_buf_avg += seg_idx * seg_size * rblock_size;
  work_buf_N += seg_idx * seg_size * rblock_size;

  if ((X_THREAD || threadIdx.x == 0) && (Y_THREAD || threadIdx.y == 0) &&
      (Z_THREAD || threadIdx.z == 0)) {
    auto rblock_offset = offset_in_reduction_segment<X_BLOCK, Y_BLOCK, Z_BLOCK>(
        blockIdx, gridDim);
    auto thread_offset =
        offset_in_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(
            threadIdx, blockDim);
    auto work_buf_offset = rblock_size * rblock_offset + thread_offset;
    if (read_write_pred) {
      work_buf_M2[work_buf_offset] = inp_M2;
      work_buf_avg[work_buf_offset] = inp_avg;
      work_buf_N[work_buf_offset] = inp_N;
    } else {
      work_buf_M2[work_buf_offset] = init_val;
      work_buf_avg[work_buf_offset] = init_val;
      work_buf_N[work_buf_offset] = 0;
    }
  }
  __syncthreads();

  __shared__ bool last_block;
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    __threadfence();
    auto old = (int64_t)atomicAdd((unsigned long long*)&sync_flags[seg_idx], 1);
    last_block = old + 1 == seg_size;
  }
  __syncthreads();

  if (last_block) {
    // final reduction
    gridWelfordLastBlock<X_THREAD, Y_THREAD, Z_THREAD>(
        out_M2,
        out_avg,
        out_N,
        (T*)work_buf_M2,
        (T*)work_buf_avg,
        (TN*)work_buf_N,
        seg_size * rblock_size,
        shared_buf_M2,
        shared_buf_avg,
        shared_buf_N,
        read_write_pred,
        init_val);
    return true;
  } else {
    return false;
  }
}
} // namespace welford
