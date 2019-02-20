#include <ATen/ATen.h>
#include <ATen/native/SortingUtils.h>
#include <assert.h>
#include <c10/macros/Macros.h>
#include <stdlib.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <THC/THCDeviceUtils.cuh> // only for THCRoundUp?
#include <THC/THCNumerics.cuh>
#include <THC/THCScanUtils.cuh>
#include <THC/THCTensorMathReduce.cuh> // AddOp

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/inner_product.h>
#include <thrust/sequence.h>
#include <THC/THCThrustAllocator.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/native/cuda/SortingSort.cuh>

namespace at {
namespace native {

namespace {

template <typename scalar_t>
struct BinaryAddOp {
  __host__ __device__ inline scalar_t operator()(
      const scalar_t a,
      const scalar_t b) {
    return THCNumerics<scalar_t>::add(a, b);
  }
};

template <>
struct BinaryAddOp<uint32_t> {
  __host__ __device__ inline uint32_t operator()(
      const uint32_t a,
      const uint32_t b) {
    return a + b;
  }
};

// Used for a segmented reduction
struct ModeUnsignedBoolPair {
  uint32_t val;
  bool flag;
};

// In the kernel below, we have a common pattern of reducing (uint32_t,
// uint32_t) pairs of data
struct ModeUnsignedPair {
  uint32_t val;
  uint32_t index;
};

template <typename scalar_t>
struct MaxReduceOp {
  __host__ __device__ inline scalar_t operator()(
      const scalar_t& a,
      const scalar_t& b) {
    return b.val > a.val ? b : a;
  }
};

template <typename scalar_t>
struct MatchReduceOp {
  __host__ __device__ inline scalar_t operator()(
      const scalar_t& a,
      const scalar_t& b) {
    return b.flag ? b : a;
  }
};

// The mode kernel has the following characteristics: It uses internal shared
// memory buffers of Power2Size, which must be greater than the number of
// elements. Additionally, there is one block for every slice to calculate the
// mode for, and in each block there is one thread for every two elements.
//
// Both sorted and positions are assumed to be contiguous Tensors with the mode
// dimension as the innermost dim, such that we can get the particular slice for
// a Tensor via its linear block dimension * the slice size.
template <typename scalar_t, uint32_t Power2Size>
__global__ void computeMode(
    scalar_t* input,
    cuda::detail::TensorInfo<scalar_t, uint32_t> values,
    cuda::detail::TensorInfo<int64_t, uint32_t> indices,
    int64_t sliceSize) {
  int tidx = threadIdx.x;
  int stidx =
      blockDim.x + threadIdx.x; // Second index this thread responsible for

  // First, we need to calculate the offset into the sorted Tensor that
  // represents the start of the slice for this block to calculate the mode for.
  // This offset is a combination of the gridIndices, and the number of elements
  // in the slice.
  uint32_t blockId = getLinearBlockId<uint32_t>();
  uint32_t linearOffset = blockId * sliceSize;

  // shmem is a dynamically sized buffer we will use throughout the kernel to
  // handle computation efficiently. The size of this shmem must be
  // sizeof(scalar_t) * Power2Size + (2 * sizeof(uint32_t) * Power2Size)
  //
  // Initially, the buffer will be organized as follows:
  //
  // [smem (slice elements) | bmem (valid indices) | <scratch space>]
  extern __shared__ char shmem[];

  // smem represents a proportion of the shared memory buffer that is used to
  // store the elements from the slice:
  scalar_t* smem = reinterpret_cast<scalar_t*>(shmem);

  // Each thread loads up to two elements from the Tensor into shared memory
  if (tidx < sliceSize) {
    smem[tidx] = input[linearOffset + tidx];
  }
  if (stidx < sliceSize) {
    smem[stidx] = input[linearOffset + stidx];
  }

  // Next, we initialize a boolean region of the buffer, offset by the loaded
  // element smem region
  bool* bmem = reinterpret_cast<bool*>(&smem[Power2Size]);

  // The first use of this region stores bmem[i] = i < sliceSize to mark the
  // valid components in the smem buffer
  bmem[tidx] = tidx < sliceSize;
  bmem[stidx] = stidx < sliceSize;
  __syncthreads(); // barrier for smem, bmem initialization

  // First, sort the input slice in ascending order. smem contains the input
  // elements, and bmem marks the valid indices
  bitonicSortValues<LTComp<scalar_t>, scalar_t, uint32_t, Power2Size>(
      smem, bmem, LTComp<scalar_t>());
  __syncthreads(); // make no assumptions that the sort syncs at end

  // The next step of our algorithm is performing a block-wide comparison of
  // neighboring elements. In particular, given an sorted input slice A, we
  // produce an output slice B, such that B[i] = 1 if A[i-i] != A[i], otherwise
  // 0.
  //
  // Given the input A = [0, 0, 1, 1, 2, 2, 2, 4, 5, 6, 6, 7, 8]
  //                 B = [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
  //
  // In particular, we can think of B[i] true indicating the start of a sequence
  // of equal values in the sorted list. Similarly, we will also store the
  // negation of B, which we'll call C. In particular, we can think of C[i] =
  // true iff A[i-1] == A[i] in our original sorted slice.
  //
  //                 C = [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0]

  // We overwrite bmem, and treat the rest of shared memory as a buffer of
  // (index, flag) pairs where the index represents values from C, and the flag
  // represents values from B.
  //
  // [smem (sorted slice) | ubpmem (index, flag pairs)]

  struct ModeUnsignedBoolPair* ubpmem =
      reinterpret_cast<struct ModeUnsignedBoolPair*>(&smem[Power2Size]);

  if (tidx == 0) {
    ubpmem[0].flag = true;
    ubpmem[0].val = 0;
  }

  // Compares elements (0, 1), (2, 3), ... and sets 1, 3, ...
  ubpmem[tidx * 2 + 1].flag = THCNumerics<scalar_t>::ne(
      smem[tidx * 2], smem[tidx * 2 + 1]); // (0, 1), (1, 2), etc.
  ubpmem[tidx * 2 + 1].val = !ubpmem[tidx * 2 + 1].flag;

  // Compares elements (1, 2), (3, 4), ... and sets 2, 4, ...
  if (((tidx + 1) * 2) < Power2Size) {
    ubpmem[(tidx + 1) * 2].flag = THCNumerics<scalar_t>::ne(
        smem[((tidx + 1) * 2) - 1], smem[(tidx + 1) * 2]);
    ubpmem[(tidx + 1) * 2].val = !ubpmem[(tidx + 1) * 2].flag;
  }
  __syncthreads(); // barrier for ubpmem initialization

  // Next, we perform a segmented prefix sum on the neighboring elements, where
  // the presence of a one indicates the start of a segment. In this case B acts
  // as the segment start flags, and C is the buffer to be summed:
  //
  // Input  (C)  = [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0]
  // Flag   (B)  = [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
  // Output (C)  = [0, 1, 0, 1, 0, 1, 2, 0, 0, 0, 1, 0, 0]
  //
  // Afterwards, the (index) components of the ubpmem buffer contain the lengths
  // of the segments (minus 1), i.e. the counts of each element in the original
  // input.

  inclusivePrefixScan<
      struct ModeUnsignedBoolPair,
      struct SegmentedScanOp<
          struct ModeUnsignedBoolPair,
          BinaryAddOp<uint32_t>>,
      Power2Size>(
      ubpmem,
      SegmentedScanOp<struct ModeUnsignedBoolPair, BinaryAddOp<uint32_t>>(
          BinaryAddOp<uint32_t>()));
  // assumes scan syncs at the end

  // Next, we reinterpret the ubpmem buffer as pairs of uint32_tegers (i.e.
  // we treat the boolean flag regions as integers). We initialize these to
  // represent indices, and we'll call this buffer I
  struct ModeUnsignedPair* uupmem =
      reinterpret_cast<struct ModeUnsignedPair*>(ubpmem);

  // At this point, we need to find the maximum element in lengths buffer C.
  // This element will represent the count (-1) of the mode. Because of the
  // way we have set up the problem, the index where this mode occurs will
  // also be the location of the mode value in the sorted array, e.g.
  //
  // smem = [0, 0, 1, 1, 1, 2]
  // C    = [0, 1, 0, 1, 2, 0]
  // I    = [0, 1, 2, 3, 4, 5]
  //                     ^
  //                     maximum value, also aligned with mode = 1
  //
  // We perform a block wide max-reduction of the C buffer, but we also need the
  // indices to come along with it, so we utilize the uupmem construction.
  //
  // At the end we need to return the ModeUnsignedPair containing index = 4, val
  // = 2, which represents the max

  // In practice, we will make each thread locally reduce 2 values in its
  // registers prior to the global block-wide reduction. Note that instead of
  // tidx/stidx, we utilize tidx * 2, tidx * 2 + 1, so each thread deals with
  // adjacent elements. This is because the reduce code below relies on thread
  // elements to be adjacent.
  struct ModeUnsignedPair uup[2];
  uup[0].index = tidx * 2;
  uup[0].val = ubpmem[tidx * 2].val;
  uup[1].index = tidx * 2 + 1;
  uup[1].val = ubpmem[tidx * 2 + 1].val;
  __syncthreads();

  struct ModeUnsignedPair max = {0, 0};

  max = reduceBlockWithNThreadLocalReductions<
      struct ModeUnsignedPair,
      MaxReduceOp<struct ModeUnsignedPair>,
      2>(uupmem, uup, sliceSize, MaxReduceOp<struct ModeUnsignedPair>(), max);

  // Store the mode in shared memory for use in finding the mode in the input
  // slice
  __shared__ scalar_t mode;

  // Given the above constraints, the mode is the value at the reduced index in
  // the original sorted element buffer
  if (tidx == 0) {
    mode = smem[max.index];
  }
  __syncthreads(); // broadcast mode

  // Finally, we need to find the "an" index of the mode in the input Tensor.
  // The API does not constrain which index we pick, so it can be any of the
  // indices that contain the mode. We will do a reduction to find the index. We
  // go back to using the (index, flag) buffer arrangement. First, we mark
  // indices that are equal to the mode, i.e B[i] = true if input[i] == mode,
  // and initialize C[i] to be the index
  //
  // Again we reduce 2 elements in the thread's registers prior to the
  // block-wide reduction
  struct ModeUnsignedBoolPair ubpp[2];
  if (tidx * 2 < sliceSize) {
    ubpp[0].flag =
        THCNumerics<scalar_t>::eq(input[linearOffset + (tidx * 2)], mode);
    ubpp[0].val = tidx * 2;
  }
  if (tidx * 2 + 1 < sliceSize) {
    ubpp[1].flag =
        THCNumerics<scalar_t>::eq(input[linearOffset + (tidx * 2 + 1)], mode);
    ubpp[1].val = tidx * 2 + 1;
  }

  // Then we perform a similar reduction to the one above, except this time we
  // update the element if the element at the base position is not equal to the
  // mode and the element at the offset position is. At the end, C[0] will
  // contain an index with the mode.
  struct ModeUnsignedBoolPair match = {0, false};

  match = reduceBlockWithNThreadLocalReductions<
      struct ModeUnsignedBoolPair,
      MatchReduceOp<struct ModeUnsignedBoolPair>,
      2>(
      ubpmem,
      ubpp,
      sliceSize,
      MatchReduceOp<struct ModeUnsignedBoolPair>(),
      match);

  // Finally, we have the mode, and an index where it occurs. We use a single
  // thread to place this in the appropriate output position
  if (tidx == 0) {
    int64_t index = TH_INDEX_BASE + match.val;

    uint32_t outputOffset =
        cuda::detail::IndexToOffset<scalar_t, uint32_t, -1>::get(
            blockId, values);
    values.data[outputOffset] = mode;
    indices.data[outputOffset] = index;
  }
}

#pragma nv_exec_check_disable
template <typename scalar_t>
void calculateMode(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    Tensor& sortBuffer,
    int64_t dimension,
    const std::vector<int64_t>& position) {
  AT_ASSERT(self.is_contiguous());

  // Because the input is contiguous, we want to get a reference to the
  // location of the buffer at the innermost dimension that we are going
  // to calculate the mode for --> we do this by manually doing the stride
  // calculations to get an offset
  auto data = self.data<scalar_t>();
  // FIXME: assert that the tensor is contiguous?
  for (int i = 0; i < position.size(); ++i) {
    data += position[i] * self.stride(i);
  }

  int64_t nElement = self.size(-1);
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());

  // Wrap input data, sortBuffer, in Thrust device vectors
  auto vecPtr = thrust::device_pointer_cast(data);
  thrust::device_vector<scalar_t> iter(vecPtr, vecPtr + nElement);
  auto sbPtr = thrust::device_pointer_cast(sortBuffer.data<int64_t>());
  thrust::device_vector<int64_t> seq(sbPtr, sbPtr + nElement);

  auto stream = at::cuda::getCurrentCUDAStream();

  // Fill sortBuffer with [0, 1, 2, ... nElement - 1]
  thrust::sequence(
      thrust::cuda::par(allocator).on(stream), seq.begin(), seq.end());

  // Sort the input data. The original indices of the data are stored in seq
  thrust::sort_by_key(
      thrust::cuda::par(allocator).on(stream),
      iter.begin(),
      iter.end(),
      seq.begin());

  // Count # of unique elements via an inner product between adjacent elements.
  // Add 1 if two neighboring element are not equal.
  int unique = 1 +
      thrust::inner_product(
                   thrust::cuda::par(allocator).on(stream),
                   iter.begin(),
                   iter.end() - 1,
                   iter.begin() + 1,
                   0,
                   thrust::plus<int>(),
                   thrust::not_equal_to<scalar_t>());

  // Count frequency of each element
  thrust::device_vector<scalar_t> keys(unique);
  thrust::device_vector<int> counts(unique);
  thrust::reduce_by_key(
      thrust::cuda::par(allocator).on(stream),
      iter.begin(),
      iter.end(),
      thrust::constant_iterator<int>(1),
      keys.begin(),
      counts.begin());

  // Find index of maximum count
  auto it = thrust::max_element(
      thrust::cuda::par(allocator).on(stream), counts.begin(), counts.end());
  scalar_t mode = keys[it - counts.begin()];

  // Find first index within which it occurs
  auto positionIter = thrust::find(
      thrust::cuda::par(allocator).on(stream), iter.begin(), iter.end(), mode);

  AT_ASSERT(positionIter != iter.end());
  int64_t index = seq[positionIter - iter.begin()];

  // Place mode, index in output
  int64_t valuesOffset = values.storage_offset();
  int64_t indicesOffset = indices.storage_offset();

  for (int i = 0; i < position.size(); ++i) {
    int64_t pos = position[i];
    valuesOffset += values.stride(i) * pos;
    indicesOffset += indices.stride(i) * pos;
  }

  // FIXME: we inherited this from THC as THCStorage_set.
  // A sensible alternative with less syncing could be to create those on CPU,
  // use an accessor here and only move the finished tensors to GPU  afterwards.
  values.as_strided({1}, {1}, valuesOffset).fill_(mode);
  indices.as_strided({1}, {1}, indicesOffset).fill_(index);
}

// this probably could be a loop, not a recursive algorithm
template <typename scalar_t>
void dimApplyMode(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    Tensor& sortBuffer,
    int64_t dimension,
    std::vector<int64_t>& position,
    int64_t curDim) {
  int64_t ndim = self.dim();

  // Because we have transposed the Tensor, the data for the dimension we are
  // mode'ing along is always in the innermost dimension
  if (curDim == ndim - 1) {
    calculateMode<scalar_t>(
        values, indices, self, sortBuffer, dimension, position);
  } else {
    // Loop through the values and recurse
    for (int i = 0; i < self.size(curDim); ++i) {
      position[curDim] = i;
      dimApplyMode<scalar_t>(
          values, indices, self, sortBuffer, dimension, position, curDim + 1);
    }
  }
}

// Function that calls kernel --> note that we set the block dimensions here,
// and the amount of shared memory
template <typename scalar_t, int work_size>
void handle_mode(
    const Tensor& self_contiguous,
    cuda::detail::TensorInfo<scalar_t, uint32_t>& tiValues,
    cuda::detail::TensorInfo<int64_t, uint32_t>& tiIndices,
    int64_t slices,
    int64_t sliceSize) {
  // The number of blocks is the number of slices that we need to calculate
  // the mode for. Each block is responsible for computing a single mode
  dim3 grid;
  if (!getGridFromTiles(slices, grid)) {
    AT_ERROR("Slice to take mode of is too large");
  }
  dim3 blockSize(work_size / 2);
  int memsize =
      (sizeof(scalar_t) * work_size) + (2 * work_size * sizeof(uint32_t));
  auto stream = at::cuda::getCurrentCUDAStream();

  computeMode<scalar_t, work_size><<<grid, blockSize, memsize, stream>>>(
      self_contiguous.data<scalar_t>(), tiValues, tiIndices, sliceSize);
}

template <typename scalar_t>
void mode_cuda_template(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim_,
    bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  // FIXME: This seems bogus, I only do this because it was the old behaviour.
  //        The reductions are fine, as long as the axis being reduced along
  //        isn't of 0 elements (and the output has elements).
  AT_CHECK(
      self.numel() > 0,
      "cannot perform reduction function mode",
      " on tensor with no elements because the operation does not have an identity");
  _reduction_with_indices_allocate_or_resize_output(
      values, indices, self, dim, keepdim);
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return;
  }

  int64_t ndim = self.dim();

  int64_t sliceSize = self.size(dim);
  int64_t slices = self.numel() / sliceSize;

  // If sliceSize is 1, copy input to values and set indices
  if (sliceSize == 1) {
    values.copy_(self);
    indices.zero_();
    if (!keepdim) {
      values.squeeze_(dim);
      indices.squeeze(dim);
    }
    return;
  }

  // Requirements for fused kernel implementation:
  //
  // 1. sliceSize <= 2 * max threads per block
  // 2. uses one block per slice, so number of slices must be less than the
  // maximum number of blocks for a kernel launch
  // 3. Can use 32-bit index math for indexing (mainly just for implementation
  // conciseness, could be changed)
  if (sliceSize <= MAX_BLOCK_SIZE && slices <= MAX_GRID_SIZE &&
      cuda::detail::canUse32BitIndexMath(self)) {
    // Beginning our optimized implementation. First thing we want to do is to
    // transpose the input Tensor along the sort dimension, and then make it
    // contiguous
    auto self_contiguous = self.transpose(dim, -1).clone();

    // We also need to view the values and indices Tensors as transposed in
    // order to properly determine the offset into the underlying storage in
    // which to place the mode and index for a particular set of dimension
    // values
    auto valuesTransposed = values.transpose(dim, -1);
    auto indicesTransposed = indices.transpose(dim, -1);

    // Set-up TensorInfo structs for passing to kernel
    auto tiValues =
        cuda::detail::getTensorInfo<scalar_t, uint32_t>(valuesTransposed);
    auto tiIndices =
        cuda::detail::getTensorInfo<int64_t, uint32_t>(indicesTransposed);

    // The blocksize is two elements per thread, rounded up to the nearest power
    // of 2
    int64_t ceilPowerOf2 = nextHighestPowerOf2(sliceSize);

    // Tradeoff between compilation time and the number of specializations.
    // Ideally we would have one HANDLE_MODE for each power of 2
    switch (ceilPowerOf2) {
      case 2048:
        handle_mode<scalar_t, 2048>(
            self_contiguous, tiValues, tiIndices, slices, sliceSize);
        break;
      case 1024:
      case 512:
      case 256:
        handle_mode<scalar_t, 1024>(
            self_contiguous, tiValues, tiIndices, slices, sliceSize);
        break;
      case 128:
      case 64:
        handle_mode<scalar_t, 128>(
            self_contiguous, tiValues, tiIndices, slices, sliceSize);
        break;
      case 32:
      case 16:
      case 8:
      case 4:
      case 2:
        handle_mode<scalar_t, 32>(
            self_contiguous, tiValues, tiIndices, slices, sliceSize);
        break;
      case 1:
      default:
        assert(false);
    }
    AT_CUDA_CHECK(cudaGetLastError());
  } else {
    // Beginning our naive implementation: We don't want to mutate the input
    // Tensor, but we need to be able to sort the inputs along the dimension in
    // order to calculate the mode. Additionally, its ideal if the data along
    // the dimension is contiguous. So we transpose the dimension with the
    // innermost dimension and make a new contiguous version that we can use.
    auto self_contiguous = self.transpose(dim, -1).clone();

    // We also need to view the values and indices Tensors as transposed in
    // order to properly determine the offset into the underlying storage in
    // which to place the mode and index for a particular set of dimension
    // values
    auto valuesTransposed = values.transpose(dim, -1);
    auto indicesTransposed = indices.transpose(dim, -1);

    // Position is a Storage that will store the dimension values we are
    // processing
    std::vector<int64_t> position(ndim);

    // Sort Buffer is a Storage that will be used in the internal sort required
    // to calculate the mode efficiently
    auto sortBuffer = at::empty({sliceSize}, indices.options());

    // Call mode
    dimApplyMode<scalar_t>(
        valuesTransposed,
        indicesTransposed,
        self_contiguous,
        sortBuffer,
        dim,
        position,
        0);
  }

  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }
}


} // namespace

std::tuple<Tensor&, Tensor&> mode_out_cuda(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), "mode", [&] {
    mode_cuda_template<scalar_t>(values, indices, self, dim, keepdim);
  });
  return std::forward_as_tuple(values, indices);
}

} // namespace native
} // namespace at
