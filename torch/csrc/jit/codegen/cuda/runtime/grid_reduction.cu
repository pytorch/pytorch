// Inter-block reduction.
//
// Function gridReduce performs point-wise reductions of scalars across thread
// blocks. Thread blocks are disjointly partitioned into groups of thread
// blocks, "reduction segments," that are collectively defined by boolean
// template parameters, X_BLOCK, Y_BLOCK and Z_BLOCK. Each of X/Y/Z_BLOCK
// determines whether thread blocks along the dimension should be grouped into
// the same reduction segment. Cross-block reducitons are independently done
// within each segment and generates distinctive results per segment. For
// instance, if all of X/Y/Z_BLOCK are true, reductions will be done across all
// thread blocks since there will be just a single segment consisting of all
// thread blocks. If none of them are true, each thread block will become a
// segment by itself, so no reduction will be performed.
//
// The input scalars to reduce within each segment are a certain subset of
// thread-private scalars provided as part of the gridReduce function
// parameters. Boolean template parameters, X_THREAD, Y_THREAD and Z_THREAD,
// determine which subset of the scalars should be used for inter-block
// reductions. Specifically, all the input scalars of threads along each
// dimension will be used when X/Y/Z_THREAD are true. Otherwise, only the value
// held at offset 0 of each dimension will be used. Thus, for example, if all of
// X/Y/Z_THREAD are true, the scalars of all threads in each block will
// participate in inter-block reductions. If all of them are false, only one
// scalar of the thread at threadIdx.x == threadIdx.y == threadIdx.z == 0 will
// be used. In the code below, we call the subset of threads a "reduction
// block."
//
// Inter-block reductions perform point-wise reductions of scalars of reduction
// blocks within each reduction segment. More specifically, let rb be a
// reduction block and rs be a reduction segment. Let IN(thread_idx, block_idx)
// denote the input scalar of thread at thread_idx and block_idx. The result of
// each reduction segment, OUT(thread_idx, block_idx_out), is defined only for
// each thread_idx in thread block block_idx_out in the segment as follows:
//
//   OUT(thread_idx, block_idx_out) =
//     Reduction of IN(thread_idx, block_idx) for
//       all block_idx in a reduction segment
//
// OUT is not given for all threads that are not in block_idx_out and the
// reduction block.
//
// See also the function comment of gridReduce.

namespace reduction {

// Utility functions
template <typename _dim3>
__device__ __forceinline__ size_t size(const _dim3& d) {
  return (size_t)d.x * (size_t)d.y * (size_t)d.z;
}

#define isize(d) d.x* d.y* d.z

template <typename _dim3pos, typename _dim3dim>
__device__ __forceinline__ size_t
offset(const _dim3pos& pos, const _dim3dim& dim) {
  return (size_t)pos.x + (size_t)pos.y * (size_t)dim.x +
      (size_t)pos.z * (size_t)dim.x * (size_t)dim.y;
}

#define ioffset(pos, dim) pos.x + pos.y* dim.x + pos.z* dim.x* dim.y

// Returns dim3 of each reduction segment.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, typename _dim3>
__device__ dim3 dimension_of_reduction_segment(const _dim3& grid_dim) {
  return dim3{
      X_BLOCK ? grid_dim.x : 1,
      Y_BLOCK ? grid_dim.y : 1,
      Z_BLOCK ? grid_dim.z : 1};
}

// Returns the number of blocks in each reduction segment.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, typename _dim3>
__device__ size_t size_of_reduction_segment(const _dim3& grid_dim) {
  return size(
      dimension_of_reduction_segment<X_BLOCK, Y_BLOCK, Z_BLOCK>(grid_dim));
}

// Returns the total number of reduction segments.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, typename _dim3>
__device__ size_t number_of_reduction_segments(const _dim3& grid_dim) {
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
__device__ size_t
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
__device__ size_t
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
__device__ dim3 dimension_of_reduction_block(const _dim3& block_dim) {
  return dim3{
      X_THREAD ? block_dim.x : 1,
      Y_THREAD ? block_dim.y : 1,
      Z_THREAD ? block_dim.z : 1};
}

// Returns the number of threads of each reduction block.
template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD, typename _dim3>
__device__ int size_of_reduction_block(const _dim3& block_dim) {
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
__device__ int offset_in_reduction_block(
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

// Reduces all the reduction blocks in each reduction segment.
//
// This is only used by one thread block per reduction segment. The input
// reduction blocks of the segment are stored in an intermediate buffer pointed
// by parameter in. Template parameters X/Y/Z_THREAD denote how the reduction
// block is formed.
//
// The size of a reduction block is by definition smaller or equal to the size
// of a thread block. We use the remaining threads to parallelize reductions
// across reduction blocks. For example, when X/Y/Z_THREAD = {true, false,
// false}, we use blockDim.y*blockDim.z threads for each output value. This is
// done first by loading the input values in parallel and then by reducing
// across threads of dimensions whose XYZ_THREAD are false.
//
// Note that what is done here after the loading from global memory is similar
// to what the existing blockReduce function does. The main difference is that
// the logical block to reduce is a 2D domain where the leading dimension is the
// size of a reduction block and the second dimension is the remaining factor in
// each thread block. For example, when X/Y/Z_THREAD = {false, true, false}, the
// threads are arranged as (blockDim.y, blockDim.x*blockDim.z). We do not reduce
// along the first dimension but only the second dimension. So, it is possible
// to reuse the existing blockReduce with dim3{blockDim.y,
// blockDim.x*blockDim.z} instead of blockDim and with X_THREAD and Y_THREAD
// being false and true, respectively. Also, it still need to shuffle the final
// output values to their actual corresponding threads. In the case of when
// X/Y/Z_THREAD = {false, true, false}, after the intra-block reduction, the
// final results will still be held by the first blockDim.y threads, which need
// to be transferred to threads at threadIdx.x == 0 and threadIdx.z == 0.
template <
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    typename T,
    typename Func>
__device__ void gridReduceLastBlock(
    T& out,
    const T* in,
    const size_t in_size,
    Func reduction_op,
    T* shared_buf,
    bool read_write_pred,
    T init_val) {
  const int tid = ioffset(threadIdx, blockDim);
  const int block_size = isize(blockDim);
  const int rblock_size =
      size_of_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(blockDim);

  T inp = init_val;
  if (tid < in_size) {
    inp = in[tid];
  }
  for (size_t i = tid + block_size; i < in_size; i += block_size) {
    reduction_op(inp, in[i]);
  }

  const auto should_write = (X_THREAD || threadIdx.x == 0) &&
      (Y_THREAD || threadIdx.y == 0) && (Z_THREAD || threadIdx.z == 0);

  auto rem_size = block_size / rblock_size;

  if (rem_size > 1) {
    const int rblock_offset = tid % rblock_size;
    const int rblock_idx = tid / rblock_size;
    blockReduce<false, true, false>(
        inp,
        inp,
        reduction_op,
        dim3{(unsigned)rblock_offset, (unsigned)rblock_idx, 0},
        dim3{(unsigned)rblock_size, (unsigned)rem_size},
        shared_buf,
        true,
        init_val);
    __syncthreads();
    if (tid < rblock_size) {
      shared_buf[tid] = inp;
    }
    __syncthreads();
    if (should_write) {
      inp = shared_buf[offset_in_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(
          threadIdx, blockDim)];
    }
  }

  if (should_write && read_write_pred) {
    out = inp;
  }
}

// Reduces per-thread values across thread blocks.
//
// Function parameters:
// - out: Per-thread output location
// - inp_val: Per-thread input value
// - reduction_op: Scalar reduction function
// - work_buf: Temporary buffer for cross-block reductions
// - sync_flags: A vector of integers for synchronizations
// - shared_buf: Shared memory buffer for intra-block reduction
//
// Return true when the thread block has the valid result.
//
// Template parameters:
// - X/Y/Z_BLOCK: When true, reduces across thread blocks along the X/Y/Z
//   dimensions
// - X/Y/Z_THREAD: When true, all threads along the X/Y/Z dimensions participate
//   in the cross-block reduction. Otherwise, only threads at offset 0 do.
// - T: Scalar data type of input/output data
// - Func: Type of scalara reduction function
//
// Template parameters X/Y/Z_BLOCK define a group of thread blocks that are
// reduced together. We call it a reduction segment. Some examples are:
//
// Case 1: X/Y/Z_BLOCK == true/true/true -> There is only one segment, which
// includes all thread blocks. It is effecively the same as the grid.
//
// Case 2: X/Y/Z_BLOCK == false/false/false -> Each thread block comprises an
// individual segment by itself.
//
// Case 3: X/Y/Z_BLOCK == true/false/false -> Each segment contains thread
// blocks that have the same blockDim.x. There will be blockDim.y*blockDim.z
// such segments.
//
// X/Y/Z_THREAD defines a sub region of a thread block that should be reduced
// with the sub regions of other thread blocks. We call it a reduction block.
// E.g.,
//
// Case 1: X/Y/Z_THREAD == false/false/false -> Only thread 0 participates in
// the cross-block reductions. The reduction block is 1x1x1 with thread 0.
//
// Case 2: X/Y/Z_THREAD == true/true/true-> All threads in a thread block
// participate in the cross-block reductions. The reduction block in this case
// is equivalent to the thread block.
//
// After the function completes, only one thread block per reduction segment
// gets valid reduction results. There is no guarantee which particular block
// gets the final results.
//
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    typename T,
    typename Func>
__device__ bool gridReduce(
    T& out,
    T inp_val,
    Func reduction_op,
    volatile T* work_buf,
    Tensor<int64_t, 1> sync_flags,
    T* shared_buf,
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

  // advance to the offset for this segment
  // index of reduction * size of the reduction * size of threads
  work_buf += seg_idx * seg_size * rblock_size;

  if ((X_THREAD || threadIdx.x == 0) && (Y_THREAD || threadIdx.y == 0) &&
      (Z_THREAD || threadIdx.z == 0)) {
    auto rblock_offset = offset_in_reduction_segment<X_BLOCK, Y_BLOCK, Z_BLOCK>(
        blockIdx, gridDim);
    auto thread_offset =
        offset_in_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(
            threadIdx, blockDim);
    auto work_buf_offset = rblock_size * rblock_offset + thread_offset;
    if (read_write_pred) {
      work_buf[work_buf_offset] = inp_val;
    } else {
      work_buf[work_buf_offset] = init_val;
    }
  }
  __syncthreads();

  __shared__ bool last_block;
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    __threadfence();
    // printf("%ld\n", sync_flags[seg_idx]);
    auto old = (int64_t)atomicAdd((unsigned long long*)&sync_flags[seg_idx], 1);
    last_block = old + 1 == seg_size;
    // printf("Last_block = %d + 1 == %d\n", (int)old, (int)seg_size);
  }
  __syncthreads();

  if (last_block) {
    // printf("Last block %d %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    // final reduction
    gridReduceLastBlock<X_THREAD, Y_THREAD, Z_THREAD>(
        out,
        (T*)work_buf,
        seg_size * rblock_size,
        reduction_op,
        shared_buf,
        read_write_pred,
        init_val);
    return true;
  } else {
    // printf("Not last block %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    return false;
  }
}

} // namespace reduction
