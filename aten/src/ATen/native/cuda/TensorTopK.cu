#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/TensorTopK.h>
#include <ATen/core/TensorBase.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <c10/macros/Macros.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/ScanUtils.cuh>
#include <ATen/cuda/AsmUtils.cuh>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/native/cuda/SortingRadixSelect.cuh>
#include <ATen/native/cuda/SortUtils.cuh>
#include <ATen/native/StridedRandomAccessor.h>
#include <ATen/cuda/cub.cuh>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/detail/KernelUtils.h>

#if defined(USE_ROCM)
#include <rocprim/block/block_scan.hpp>
#endif

using namespace at::native;

namespace at::native {

namespace sbtopk { // single_block_topk

template <typename T>
struct AddOp {
  __device__ __forceinline__ T operator()(T const &lhs, T const &rhs) {
    return (lhs + rhs);
  }
};

#ifndef USE_ROCM

template <typename T, typename IndexType, int Dim, bool WithKthValues>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void gatherTopK(at::cuda::detail::TensorInfo<const T, IndexType> input,
                           IndexType inputSliceSize,
                           IndexType outputSliceSize, // aka `k`
                           bool largest,

                           IndexType numInputSlices,
                           IndexType inputWithinSliceStride,

                           at::cuda::detail::TensorInfo<T, IndexType> topK,
                           IndexType topKWithinSliceStride,

                           at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
                           IndexType indicesWithinSliceStride,
                           T* kthValues) {
  // Indices are limited to integer fp precision, so counts can fit in
  // int32, regardless of IndexType
  __shared__ int smem[32]; // one per each warp, up to warp limit
  IndexType slice = getLinearBlockId<IndexType>();
  if (slice >= numInputSlices) {
    return;
  }

  // Find the start offset for our slice
  IndexType sliceStartIndex =
    at::cuda::detail::IndexToOffset<const T, IndexType, Dim>::get(slice, input);
  IndexType topKSliceStartIndex =
    at::cuda::detail::IndexToOffset<T, IndexType, Dim>::get(slice, topK);
  IndexType indicesSliceStartIndex =
    at::cuda::detail::IndexToOffset<int64_t, IndexType, Dim>::get(slice, indices);

  const T* inputSliceStart = &input.data[sliceStartIndex];
  T* topKSliceStart = &topK.data[topKSliceStartIndex];
  int64_t* indicesSliceStart = &indices.data[indicesSliceStartIndex];

  // Find the k-th highest element in our input
  T topKValue;
  if (WithKthValues){
    topKValue = kthValues[slice];
  } else {
    topKValue = static_cast<T>(0);
    radixSelect<T, typename TopKTypeConfig<T>::RadixType, IndexType>(
      inputSliceStart, outputSliceSize, largest,
      inputSliceSize, inputWithinSliceStride,
      smem, &topKValue);
  }
  const auto topKConverted = at::native::TopKTypeConfig<T>::convert(topKValue);

  // Every value that is strictly less/greater than `pattern`
  // (depending on sort dir) in sorted int format is in the top-K.
  // The top-K value itself might not be unique.
  //
  // Since there are a variable number of elements that we see that
  // are within the top-k, we don't know at what index to write out
  // the resulting values.
  // In order to get this, we perform an exclusive prefix sum of
  // `hasTopK`. This will return the resulting index into which we
  // need to write the result, if a thread has a result.

  // All threads need to participate in the loop and the prefix sum,
  // but not necessarily in the load; hence loop bounds being rounded
  // up to a multiple of the block dim.
  IndexType numIterations = round_up(inputSliceSize, (IndexType) blockDim.x);
  IndexType writeIndexStart = 0;

  for (IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < inputSliceSize);
    T v =
      inRange ? doLdg(&inputSliceStart[i * inputWithinSliceStride]) : static_cast<T>(0);
    const auto convertedV = at::native::TopKTypeConfig<T>::convert(v);
    bool hasTopK;
    if (largest) {
      hasTopK = inRange && (convertedV > topKConverted);
    } else {
      hasTopK = inRange && (convertedV < topKConverted);
    }

    int index;
    int carry;
    at::cuda::exclusiveBinaryPrefixScan<int, true>(
        smem, hasTopK, &index, &carry, AddOp<int>());

    if (hasTopK) {
      int writeIndex = writeIndexStart + index;
      CUDA_KERNEL_ASSERT(writeIndex < outputSliceSize);

      IndexType topKOffset = writeIndex * topKWithinSliceStride;
      IndexType indexOffset = writeIndex * indicesWithinSliceStride;

      topKSliceStart[topKOffset] = v;
      indicesSliceStart[indexOffset] = i;
    }

    writeIndexStart += carry;
  }

  // We need to fill in the rest with actual == top-K values.
  // The number that we need is outputSliceSize -
  // writeIndexStart. There might be more than that number available,
  // in which case we have to choose the first seen set. We do this
  // via a prefix sum to calculate indices for writing results.
  CUDA_KERNEL_ASSERT(outputSliceSize >= writeIndexStart);
  IndexType topKRemaining = (outputSliceSize - writeIndexStart);

  for (IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < inputSliceSize);
    T v =
      inRange ? doLdg(&inputSliceStart[i * inputWithinSliceStride]) : static_cast<T>(0);
    const auto convertedV = at::native::TopKTypeConfig<T>::convert(v);
    bool hasTopK = inRange && (convertedV == topKConverted);

    int index;
    int carry;
    at::cuda::exclusiveBinaryPrefixScan<int, true>(
        smem, hasTopK, &index, &carry, AddOp<int>());

    if (hasTopK && index < topKRemaining) {
      int writeIndex = writeIndexStart + index;
      CUDA_KERNEL_ASSERT(writeIndex < outputSliceSize);

      IndexType topKOffset = writeIndex * topKWithinSliceStride;
      IndexType indexOffset = writeIndex * indicesWithinSliceStride;

      topKSliceStart[topKOffset] = v;
      indicesSliceStart[indexOffset] = i;
    }

    if (carry >= topKRemaining) {
      break;
    }

    topKRemaining -= carry;
    writeIndexStart += carry;
  }

}

#else

/*
This implementation of gatherTopK is a modified version of the original gatherTopK kernel. This kernel is called
after we have found the k-th highest (or lowest, depending on the sort direction) element in our input. It gathers
values that are greater (or less than) than the k-th element (phase 1) and then adds the values that are equal to
the k-th element as long as there is space available (phase 2). In the original implementation, we call
exclusiveBinaryPrefixScan to calculate the index to write the result to. However, exclusiveBinaryPrefixScan has two
block level synchronization points, which is not efficient, specially considering that exclusiveBinaryPrefixScan is
called in a loop. In this implementation, we use warp level compaction to calculate the index to write the result
to. In both phases, each warp first counts the number of values it intends to add to the result. Then through an
atomic add, the warp reserves space for itself (atomically increases the write index variable) and then writes the
result to the corresponding indices. This requires no block level synchronization. It should be noted that we have
added a block level synchronization point after phase 1 to make sure all threads have completed phase 1. This
synchronization is cheaper than the ones in exclusiveBinaryPrefixScan because it is called only once. This
synchronization is necessary because phase 1 assumes it always has space to write all the values that are larger (or
smaller) than the k-th element but phase 2 tops off the output as long as there is space available.
*/

// helper function to reserve space for a warp in the output.
// hasTopK: boolean flag to indicate if the current thread has a value to add to the output.
// writeIndexStart: atomic variable to track the index to write the result to.
// start_index: index to write the result to. (output of function)
// my_offset: offset to write the result to. (output of function)
// warp_count: number of threads that have values to add to the output. (output of function)
__device__ __forceinline__ void reserveWarpSpace(bool hasTopK,
                                                int& writeIndexStart,
                                                int& start_index,
                                                int& my_offset,
                                                int& warp_count) {
  auto ballot = WARP_BALLOT(hasTopK); // a bitmask of threads that have hasTopK == true within the warp.
  warp_count = __popcll(ballot); // count the number of threads that have hasTopK == true within the warp.

  int lane_id = at::cuda::getLaneId();

  // if > 0 threads have hasTopK == true within the warp,
  // reserve space for them by incrementing writeIndexStart atomically + saving the old value  as start index.
  if (warp_count > 0 && lane_id == 0) {
    start_index = atomicAdd(&writeIndexStart, warp_count);
  }
  start_index = __shfl(start_index, 0); // broadcast the start index to all threads in the warp.

  uint64_t mask = (1ULL << lane_id) - 1; // a bitmask: [0, 0, 0, ..., 0, 1, 1, 1, ..., 1] with (64-lane_id) 0s and (lane_id) 1s
  my_offset = __popcll(ballot & mask);  // get number of threads that have hasTopK == true to the right of the current lane
}

// helper function to write the result to the output.
template <typename T, typename IndexType>
__device__ __forceinline__ void writeResult(T* topKSliceStart,
                                            int64_t* indicesSliceStart,
                                            IndexType topKWithinSliceStride,
                                            IndexType indicesWithinSliceStride,
                                            IndexType outputSliceSize,
                                            int writeIndex,
                                            T v,
                                            IndexType i){
  CUDA_KERNEL_ASSERT(writeIndex < outputSliceSize); // assert that the write index is within the output slice size.
  IndexType topKOffset = writeIndex * topKWithinSliceStride; // calculate the offset to the topk value in the output slice.
  IndexType indexOffset = writeIndex * indicesWithinSliceStride; // calculate the offset to the index in the output slice.
  topKSliceStart[topKOffset] = v; // write the value to the output slice.
  indicesSliceStart[indexOffset] = i; // write the index to the output slice.
}

template <typename T, typename IndexType, int Dim, bool WithKthValues>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void gatherTopK(at::cuda::detail::TensorInfo<const T, IndexType> input,
                            IndexType inputSliceSize,
                            IndexType outputSliceSize, // aka `k`
                            bool largest,

                            IndexType numInputSlices,
                            IndexType inputWithinSliceStride,

                            at::cuda::detail::TensorInfo<T, IndexType> topK,
                            IndexType topKWithinSliceStride,

                            at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
                            IndexType indicesWithinSliceStride,
                            T* kthValues) {

  // Indices are limited to integer fp precision, so counts can fit in
  // int32, regardless of IndexType

  // Maximum shared memory size for radix select (used in countRadixAggregateCounts): NUM_BUFFERS * MAX_WARPS * RADIX_SIZE.
  // HIP workgroups have at most 1024 threads. Warp size is at least 32 (can be 64 on some
  // architectures), so we use 32 for safety: 2 buffers * (1024/32) warps * 4 radix bins = 256.
  __shared__ int smem[256];
  __shared__ int writeIndexStart; // index to track where to write results. This is shared by all threads in the block. Increases atomically.

  IndexType slice = getLinearBlockId<IndexType>();
  if (slice >= numInputSlices) {
    return;
  }

  // Find the start offset for our slice
  IndexType sliceStartIndex =
    at::cuda::detail::IndexToOffset<const T, IndexType, Dim>::get(slice, input);
  IndexType topKSliceStartIndex =
    at::cuda::detail::IndexToOffset<T, IndexType, Dim>::get(slice, topK);
  IndexType indicesSliceStartIndex =
    at::cuda::detail::IndexToOffset<int64_t, IndexType, Dim>::get(slice, indices);

  const T* inputSliceStart = &input.data[sliceStartIndex];
  T* topKSliceStart = &topK.data[topKSliceStartIndex];
  int64_t* indicesSliceStart = &indices.data[indicesSliceStartIndex];

  // Find the k-th highest element in our input
  T topKValue;
  if (WithKthValues){
    topKValue = kthValues[slice];
  } else {
    topKValue = static_cast<T>(0);
    radixSelect<T, typename TopKTypeConfig<T>::RadixType, IndexType>(
      inputSliceStart, outputSliceSize, largest,
      inputSliceSize, inputWithinSliceStride,
      smem, &topKValue);
  }
  const auto topKConverted = at::native::TopKTypeConfig<T>::convert(topKValue);

  // Every value that is strictly less/greater than `pattern`
  // (depending on sort dir) in sorted int format is in the top-K.
  // The top-K value itself might not be unique.
  //
  // Since there are a variable number of elements that we see that
  // are within the top-k, we don't know at what index to write out
  // the resulting values.
  // In order to get this, we perform warp level compaction.
  // each warp counts its own number of hasTopk threads and
  // reserves space for them by incrementing writeIndexStart atomically + saving the old value as start index.

  // Initialize writeIndexStart to 0 by the first thread in the block.
  if (threadIdx.x == 0) {
    writeIndexStart = 0;
  }
  __syncthreads();
  // All threads within the warp need to participate in the loop, so rounding up to a multiple of the warp size.
  IndexType numIterations = round_up(inputSliceSize, (IndexType) warpSize);

  // phase 1: write actual > `pattern` (or < `pattern`, depending on the sort direction) values to the output.
  // prefetching data from global memory.
  T v = (threadIdx.x < inputSliceSize) ? doLdg(&inputSliceStart[threadIdx.x * inputWithinSliceStride]) : static_cast<T>(0);
  for (IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
    T v_next = (i + blockDim.x < inputSliceSize) ? doLdg(&inputSliceStart[(i + blockDim.x) * inputWithinSliceStride]) : static_cast<T>(0);

    bool hasTopK = false;
    if (i < inputSliceSize) {
      const auto convertedV = at::native::TopKTypeConfig<T>::convert(v);
      hasTopK = (largest) ? (convertedV > topKConverted) : (convertedV < topKConverted);
    }

    int start_index, my_offset, warp_count;
    reserveWarpSpace(hasTopK, writeIndexStart, start_index, my_offset, warp_count);

    // now warp has reserved space for itself. If hasTopK == true, we need to find the index to write the result to.
    if (hasTopK) {
      writeResult(topKSliceStart,
        indicesSliceStart,
        topKWithinSliceStride,
        indicesWithinSliceStride,
        outputSliceSize,
        /*writeIndex=*/start_index + my_offset,
        /*value=*/v,
        /*index=*/i);
    }

    v = v_next;
  }

  // till this point, actual > `pattern` values were being written.
  // we first need to sync to make sure all threads have completed phase 1:
  __syncthreads();

  // We need to fill in the rest with actual == top-K values.
  // The number that we need is outputSliceSize - writeIndexStart.
  // There might be more than that number available in input,
  // in which case we have to choose the first seen set. We do this
  // in a similar warp level compaction fashion as in phase 1.

  // phase 2: write actual == `pattern` values to the output.
  // prefetching data from global memory.
  T V = (threadIdx.x < inputSliceSize) ? doLdg(&inputSliceStart[threadIdx.x * inputWithinSliceStride]) : static_cast<T>(0);
  for (IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
    T V_next = (i + blockDim.x < inputSliceSize) ? doLdg(&inputSliceStart[(i + blockDim.x) * inputWithinSliceStride]) : static_cast<T>(0);
    bool hasTopK = false;
    if (i < inputSliceSize) {
      const auto convertedV = at::native::TopKTypeConfig<T>::convert(V);
      hasTopK = convertedV == topKConverted;
    }

    int start_index, my_offset, warp_count;
    reserveWarpSpace(hasTopK, writeIndexStart, start_index, my_offset, warp_count);

    if ((warp_count > 0) && (outputSliceSize <= start_index)){
      break; // there is no space to add topk values. Break out of the loop.
    }

    if (hasTopK){
      int slots_available = outputSliceSize - start_index;
      if (my_offset < slots_available){
        writeResult(topKSliceStart,
          indicesSliceStart,
          topKWithinSliceStride,
          indicesWithinSliceStride,
          outputSliceSize,
          /*writeIndex=*/start_index + my_offset,
          /*value=*/V,
          /*index=*/i);
      }
    }

    V = V_next;
  }
}

#endif

template <typename T, typename IndexType, int Dim>
void launch(
    at::cuda::detail::TensorInfo<const T, IndexType> input,
    IndexType inputSliceSize,
    IndexType outputSliceSize, // aka `k`
    bool largest,

    IndexType numInputSlices,
    IndexType inputWithinSliceStride,

    at::cuda::detail::TensorInfo<T, IndexType> topK,
    IndexType topKWithinSliceStride,

    at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
    IndexType indicesWithinSliceStride) {

    dim3 grid;
    TORCH_INTERNAL_ASSERT(getGridFromTiles(numInputSlices, grid), "Too many slices for topk");
    int warp_size = at::cuda::warp_size();
    dim3 block(std::min(at::ceil_div((int64_t)inputSliceSize, (int64_t)warp_size) * (int64_t)warp_size, (int64_t)1024));
    gatherTopK<T, IndexType, Dim, /* WithKthValues= */false><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input,
        inputSliceSize,
        outputSliceSize,
        largest,
        numInputSlices,
        inputWithinSliceStride,
        topK,
        topKWithinSliceStride,
        indices,
        indicesWithinSliceStride,
        nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace sbtopk

#if defined(USE_ROCM) && HAS_WARP_MERGE_SORT()
namespace warptopk {

constexpr int MAX_WARP_TOPK_SLICE = 512;

// Comparator for sorting with TopK semantics
// Note: For WarpMergeSort (comparison-based sorting), we use simple comparison operators
// GTOp/LTOp instead of bitwise conversion. Bitwise conversion is only needed for radix sorting.

// Kernel using WarpMergeSort for small topK operations
template <int KeyDims, int ValueDims, int sort_size, int max_block_dim_y,
          typename scalar_t, typename IndexType, bool is_descending>
__global__ void warpMergeSortTopK(
    at::cuda::detail::TensorInfo<const scalar_t, IndexType> input,
    IndexType inputSliceSize,
    IndexType k,
    IndexType numInputSlices,
    IndexType inputWithinSliceStride,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> topK,
    IndexType topKWithinSliceStride,
    at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
    IndexType indicesWithinSliceStride) {

  // Find the slice this warp is working on
  const IndexType blockIndex = getLinearBlockId<IndexType>();
  const IndexType linearIndex = blockIndex * blockDim.y + threadIdx.y;

  if (linearIndex >= numInputSlices) {
    return;
  }

  const IndexType inputStartOffset =
      at::cuda::detail::IndexToOffset<const scalar_t, IndexType, KeyDims>::get(linearIndex, input);
  const IndexType topKStartOffset =
      at::cuda::detail::IndexToOffset<scalar_t, IndexType, ValueDims>::get(linearIndex, topK);
  const IndexType indicesStartOffset =
      at::cuda::detail::IndexToOffset<int64_t, IndexType, ValueDims>::get(linearIndex, indices);

  const scalar_t* input_slice = &input.data[inputStartOffset];
  scalar_t* topK_slice = &topK.data[topKStartOffset];
  int64_t* indices_slice = &indices.data[indicesStartOffset];

  // Create strided accessors
  StridedRandomAccessor<const scalar_t, IndexType> input_iter(input_slice, inputWithinSliceStride);
  StridedRandomAccessor<scalar_t, IndexType> topK_iter(topK_slice, topKWithinSliceStride);
  StridedRandomAccessor<int64_t, IndexType> indices_iter(indices_slice, indicesWithinSliceStride);

  namespace cub = ROCM_HIPCUB(at_cuda_detail::cub);

  CUDA_KERNEL_ASSERT(blockDim.x == C10_WARP_SIZE);
  CUDA_KERNEL_ASSERT(blockDim.y <= max_block_dim_y);
  constexpr int items_per_thread = sort_size / C10_WARP_SIZE;
  static_assert(items_per_thread * C10_WARP_SIZE == sort_size,
                "sort_size must be a multiple of C10_WARP_SIZE");

  using LoadKeys = cub::WarpLoad<scalar_t, items_per_thread, cub::WARP_LOAD_TRANSPOSE>;
  using Sort = cub::WarpMergeSort<scalar_t, items_per_thread, C10_WARP_SIZE, int64_t>;
  using StoreKeys = cub::WarpStore<scalar_t, items_per_thread, cub::WARP_STORE_TRANSPOSE>;
  using StoreIndices = cub::WarpStore<int64_t, items_per_thread, cub::WARP_STORE_TRANSPOSE>;

  __shared__ union {
    typename LoadKeys::TempStorage load_keys;
    typename Sort::TempStorage sort;
    typename StoreKeys::TempStorage store_keys;
    typename StoreIndices::TempStorage store_indices;
  } tmp_storage[max_block_dim_y];

  auto& warp_storage = tmp_storage[threadIdx.y];

  // Thread-local arrays for values and indices
  scalar_t local_values[items_per_thread];
  int64_t local_indices[items_per_thread];

  // Invalid sentinel for padding
  const scalar_t invalid_value = is_descending
      ? -std::numeric_limits<scalar_t>::infinity()
      : std::numeric_limits<scalar_t>::infinity();

  // Initialize indices for this slice in blocked arrangement
  // WARP_LOAD_TRANSPOSE uses blocked layout: thread t gets items [t*items_per_thread, t*items_per_thread+items_per_thread-1]
  #pragma unroll
  for (int i = 0; i < items_per_thread; ++i) {
    local_indices[i] = threadIdx.x * items_per_thread + i;
  }

  // Load values from input
  LoadKeys(warp_storage.load_keys).Load(input_iter, local_values, inputSliceSize, invalid_value);
  WARP_SYNC();

  // Sort values with their indices using comparison-based sort
  // For largest=true (descending), use GTOp to get larger values first
  // For largest=false (ascending), use LTOp to get smaller values first
  if constexpr (is_descending) {
    Sort(warp_storage.sort).StableSort(local_values, local_indices, GTOp<scalar_t, true>(), inputSliceSize, invalid_value);
  } else {
    Sort(warp_storage.sort).StableSort(local_values, local_indices, LTOp<scalar_t, true>(), inputSliceSize, invalid_value);
  }
  WARP_SYNC();

  // Store top-k results
  StoreKeys(warp_storage.store_keys).Store(topK_iter, local_values, k);
  WARP_SYNC();
  StoreIndices(warp_storage.store_indices).Store(indices_iter, local_indices, k);
}

template <typename scalar_t, typename IndexType, int Dim>
void launch(
    at::cuda::detail::TensorInfo<const scalar_t, IndexType> input,
    IndexType inputSliceSize,
    IndexType k,
    bool largest,
    IndexType numInputSlices,
    IndexType inputWithinSliceStride,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> topK,
    IndexType topKWithinSliceStride,
    at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
    IndexType indicesWithinSliceStride) {
  TORCH_INTERNAL_ASSERT(inputSliceSize <= MAX_WARP_TOPK_SLICE,
                        "warp topk path requires slice size <= ", MAX_WARP_TOPK_SLICE);

  const auto stream = c10::cuda::getCurrentCUDAStream();
  // Use only 1 row per block to minimize shared memory usage
  // (max_block_dim_y * sizeof(union) must fit in 64KB LDS limit)
  constexpr int max_block_dim_y = 1;
  const int block_x = at::cuda::warp_size();
  dim3 block(block_x, max_block_dim_y);

  dim3 grid;
  TORCH_INTERNAL_ASSERT(getGridFromTiles(numInputSlices, grid),
                        "Too many slices for warp topk");

  // Dispatch based on sort size and sort direction
  #define LAUNCH_KERNEL(SORT_SIZE, IS_DESCENDING) \
    warpMergeSortTopK<Dim, Dim, SORT_SIZE, max_block_dim_y, scalar_t, IndexType, IS_DESCENDING> \
      <<<grid, block, 0, stream>>>( \
          input, inputSliceSize, k, numInputSlices, inputWithinSliceStride, \
          topK, topKWithinSliceStride, indices, indicesWithinSliceStride); \
    C10_CUDA_KERNEL_LAUNCH_CHECK()

  // We have specialized launches for different sizes, as sort_size affects
  // shared memory, registers per thread and occupancy. We can use 'LAUNCH_KERNEL(512, false);'
  // however, that results in lower performance.
  if (largest) {
    if (inputSliceSize <= 64) {
      LAUNCH_KERNEL(64, true);
    } else if (inputSliceSize <= 128) {
      LAUNCH_KERNEL(128, true);
    } else if (inputSliceSize <= 256) {
      LAUNCH_KERNEL(256, true);
    } else {
      // inputSliceSize <= 512
      LAUNCH_KERNEL(512, true);
    }
  } else {
    if (inputSliceSize <= 64) {
      LAUNCH_KERNEL(64, false);
    } else if (inputSliceSize <= 128) {
      LAUNCH_KERNEL(128, false);
    } else if (inputSliceSize <= 256) {
      LAUNCH_KERNEL(256, false);
    } else {
      // inputSliceSize <= 512
      LAUNCH_KERNEL(512, false);
    }
  }

  #undef LAUNCH_KERNEL
}

} // namespace warptopk
#endif // defined(USE_ROCM) && HAS_WARP_MERGE_SORT()

namespace mbtopk { // multi_block_topk

// Assumptions:
// The number of elements can be larger than UINT32_MAX, but
// the number of total blocks can not be larger than UINT32_MAX.
// So we can not have more than UINT32_MAX slices. The actual limit
// for number of slices could be a few fold smaller than UINT32_MAX,
// because we could be using multiple blocks per slice.
// Further more, the size of each input slice is also assumped to be
// smaller than UINT32_MAX

constexpr int BLOCK_THREADS = 256;

// Over what radix we are selecting values
constexpr int RADIX_BITS = 8;
constexpr int RADIX_DIGITS = 1 << RADIX_BITS; // 2 ^ RADIX_BITS
constexpr int RADIX_MASK = (RADIX_DIGITS - 1);
static_assert(RADIX_DIGITS <= BLOCK_THREADS, "RADIX_DIGITS must be <= BLOCK_THREADS");
constexpr int MIN_ITEMS_PER_THREAD = 4;
#if defined(USE_ROCM)
// AMD: Allow higher items_per_thread for large arrays to reduce blocks_per_slice
// This reduces overhead in accumulation loops (computeBlockwiseWithinKCounts, gatherTopK)
constexpr int MAX_ITEMS_PER_THREAD = 96;
#else
constexpr int MAX_ITEMS_PER_THREAD = 64;
#endif

template <typename T, typename IndexType>
__global__ void fill(T* x, T value, IndexType size) {
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (IndexType i = idx; i < size; i += gridDim.x * blockDim.x) {
    x[i] = value;
  }
}

// compute local histogram for each block
template <typename T, typename IndexType, typename Bitwise, int Dim>
C10_LAUNCH_BOUNDS_1(BLOCK_THREADS)
__global__ void computeBlockDigitCounts(
    at::cuda::detail::TensorInfo<const T, IndexType> input,
    uint32_t slice_size,
    uint32_t* ks_to_find,  // size: num_slices, unused arg but for mysterious reasons perf is better when it's present
    uint32_t num_slices,
    IndexType withinSliceStride,
    int current_bit,
    int items_per_thread,
    uint32_t blocks_per_slice,
    Bitwise desiredMask,
    Bitwise* desires,      // size: num_slices
    short* counts         // size: num_slices * blocks_per_slice * radix_digits
  ) {

  int items_per_block = items_per_thread * BLOCK_THREADS;
  int tidx = threadIdx.x;
  uint32_t block_idx = getLinearBlockId<uint32_t>();
  uint32_t slice_idx = block_idx / blocks_per_slice;
  uint32_t blk_idx_in_slice = block_idx % blocks_per_slice;
  if (slice_idx >= num_slices) {
    return;
  }

  Bitwise desired = desires[slice_idx];
  IndexType slice_start_index = at::cuda::detail::IndexToOffset<const T, IndexType, Dim>::get(slice_idx, input);
  const T* data = &input.data[slice_start_index];

  static_assert(MAX_ITEMS_PER_THREAD * BLOCK_THREADS < std::numeric_limits<short>::max(),
    "blockwise counter too large");
  union __align__(16) TempStorage {
    uint32_t digit_counters[RADIX_DIGITS];
  };
  __shared__ TempStorage temp_storage;

  // fill digit_counters with zeros
  if (tidx < RADIX_DIGITS) {
    temp_storage.digit_counters[tidx] = 0;
  }
  __syncthreads();

  items_per_thread = (blk_idx_in_slice + 1 < blocks_per_slice)
      ? items_per_thread
      : at::ceil_div((int64_t)(slice_size - blk_idx_in_slice * items_per_block), (int64_t)BLOCK_THREADS);

  // collect digit counts and store in shared memory
  for (int i = 0; i < items_per_thread; ++i) {
    // Find the start offset for this slice
    IndexType idx = blk_idx_in_slice * items_per_block + i * BLOCK_THREADS + tidx;
    if (idx < slice_size) {
      idx *= withinSliceStride;
      Bitwise val = TopKTypeConfig<T>::convert(doLdg(&data[idx]));
      bool has_val = ((val & desiredMask) == (desired & desiredMask));
      Bitwise digit = at::cuda::Bitfield<Bitwise>::getBitfield(val, current_bit, RADIX_BITS);
      if (has_val) {
        atomicAdd(&temp_storage.digit_counters[digit], 1);
      }
    }
  }

  __syncthreads();

  // load digit counter to register, one digit per thread
  static_assert(RADIX_DIGITS <= BLOCK_THREADS, "this kernel requires RADIX_DIGITS <= BLOCK_THREADS");
  uint32_t digit_count = 0;
  if (tidx < RADIX_DIGITS) {
    digit_count = temp_storage.digit_counters[tidx];
  }

  // We always write out counts regardless if blocks_per_slice == 1 because
  // it will be used to compute offsets for `gatherTopK`.
  if (tidx < RADIX_DIGITS) {
    counts[block_idx * RADIX_DIGITS + tidx] = digit_count;
  }
}

#ifndef USE_ROCM
// CUDA path: compute global histogram and cumsum for each row
__global__ void computeDigitCumSum(
  short* counts,
  uint32_t* digit_cum_sum,
  uint32_t blocks_per_slice) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int digit_idx = threadIdx.x;
  uint32_t slice_idx = blockIdx.x;

  typedef cub::BlockScan<uint32_t, RADIX_DIGITS> BlockScan;
  __shared__ typename BlockScan::TempStorage scan_storage;
  // accumulates counters from multiple blocks
  uint32_t digit_count = 0;
  if (threadIdx.x < RADIX_DIGITS) {
    constexpr int HISTO_ACCUM_TILE = 4;
    uint32_t rounds = blocks_per_slice / HISTO_ACCUM_TILE;
    for (int iter = 0; iter < rounds; iter++)  {
      int base = HISTO_ACCUM_TILE * iter;
      #pragma unroll
      for (int j = 0; j < HISTO_ACCUM_TILE; j++) {
        int blk = base + j;
        digit_count += counts[(slice_idx * blocks_per_slice + blk) * RADIX_DIGITS + digit_idx];
      }
    }
    for (int blk = HISTO_ACCUM_TILE * rounds; blk < blocks_per_slice; blk++)  {
      digit_count += counts[(slice_idx * blocks_per_slice + blk) * RADIX_DIGITS + digit_idx];
    }

  }
  // compute the block-wide inclusive prefix sum
  uint32_t digit_count_cumsum;
  BlockScan(scan_storage).InclusiveSum(digit_count, digit_count_cumsum);
  __syncthreads();
  if (threadIdx.x < RADIX_DIGITS) {
    digit_cum_sum[tidx] = digit_count_cumsum;
  }
}

// CUDA path: Assumption: k can not be larger than UINT32_MAX
template <typename Bitwise, typename T>
C10_LAUNCH_BOUNDS_1(RADIX_DIGITS)  // one thread per digit
__global__ void computeBlockwiseWithinKCounts(
  Bitwise* desires_in,          // size: num_slices
  short* counts,                // size: num_slices * blocks_per_slice * radix_digits
  uint32_t* digit_cum_sum,      // CUDA: reads pre-computed cumsum
  uint32_t* ks_to_find_in,      // size: num_slices
  uint32_t blocks_per_slice,
  int current_bit,
  bool largest,
  // outputs:
  uint32_t* withinKCounts,      // size: num_slices * blocks_per_slice == num_blocks
  T* kthValues,                 // size: num_slices, only write when current_bit reaches 0
  uint32_t* ks_to_find_out,
  Bitwise* desires_out,
  uint32_t num_blocks
) {
  // This kernel should be launched with the same number of blocks as the `computeBlockDigitCounts` kernel.
  int tidx = threadIdx.x;
  uint32_t block_idx = getLinearBlockId<uint32_t>();
  uint32_t slice_idx = block_idx / blocks_per_slice;

  // The grid is computed from `getGridFromTiles`, when there are lots of
  // elements, we will use both blockIdx.x and blockIdx.y, and maybe blockIdx.z
  // when this is the case, the number of blocks that we are launching can be
  // more than the number of blocks we need. So we need to check the range of
  // `block_idx`.
  if (block_idx >= num_blocks) {
    return;
  }


  __shared__ Bitwise desired;
  uint32_t k_to_find = ks_to_find_in[slice_idx];

  if (tidx < RADIX_DIGITS) {
    uint32_t position = slice_idx * RADIX_DIGITS + tidx;
    uint32_t digit_count_cumsum = digit_cum_sum[position];
    uint32_t digit_count_cumsum_left = (tidx == 0) ? 0 : digit_cum_sum[position - 1];

    // if not the last pass: update desired and ks_to_find
    // if last pass: write out the kth value
    // only one thread in block enters this condition
    if (digit_count_cumsum_left < k_to_find && k_to_find <= digit_count_cumsum) {
      desired = desires_in[slice_idx];
      desired = at::cuda::Bitfield<Bitwise>::setBitfield(desired, tidx, current_bit, RADIX_BITS);
      // let a single block per slice update the values
      if (block_idx == slice_idx * blocks_per_slice) {
        desires_out[slice_idx] = desired;
        if (current_bit > 0) {
          ks_to_find_out[slice_idx] = k_to_find - digit_count_cumsum_left;
        } else {
          kthValues[slice_idx] = TopKTypeConfig<T>::deconvert(desired);
        }
      }
    }
  }
  __syncthreads();

  Bitwise desired_digit = at::cuda::Bitfield<Bitwise>::getBitfield(desired, current_bit, RADIX_BITS);

  // if largest, then only threads that has tidx > desired_digit are active
  // if !largest, then only threads that has tidx < desired_digit are active
  // each active thread will read the count for its corresponding, and
  // do warp reduction followed by shared memory reduction to get the total count
  // non-active thread should not load, and non-active warp should not do reduction.
  bool warp_is_active, thread_is_active;
  int warp = tidx / C10_WARP_SIZE;
  if (largest) {
    int end_of_warp = warp * C10_WARP_SIZE + C10_WARP_SIZE - 1;
    warp_is_active = end_of_warp > desired_digit;
    thread_is_active = tidx > desired_digit;
  } else {
    int start_of_warp = warp * C10_WARP_SIZE;
    warp_is_active = start_of_warp < desired_digit;
    thread_is_active = tidx < desired_digit;
  }
  uint32_t count = 0;
  if (warp_is_active) {
    if (thread_is_active) {
      count = doLdg(counts + block_idx * RADIX_DIGITS + tidx);
    }
    for (int offset = C10_WARP_SIZE / 2; offset > 0; offset /= 2) {
      count += WARP_SHFL_DOWN(count, offset);
    }
  }

  constexpr int num_warps = RADIX_DIGITS / C10_WARP_SIZE;
  __shared__ uint32_t warp_counts[num_warps];
  if (tidx % C10_WARP_SIZE == 0) {
    warp_counts[warp] = count;
  }
  __syncthreads();
  static_assert(RADIX_DIGITS < C10_WARP_SIZE * C10_WARP_SIZE,
    "Assuming only 1 warp is needed for final reduction");
  if (warp != 0) {
    return;
  }
  count = 0;
  if (tidx < num_warps) {
    count = warp_counts[tidx];
  }
  for (int offset = num_warps / 2; offset > 0; offset /= 2) {
    count += WARP_SHFL_DOWN(count, offset);
  }
  if (tidx == 0) {
    withinKCounts[block_idx] += count;
  }
}

// CUDA path: Assumption: slice_size can not be larger than UINT32_MAX
template <typename Bitwise>
__global__ void computeBlockwiseKthCounts(
  Bitwise* desires,            // size: num_slices
  short* counts,               // size: num_slices * blocks_per_slice * radix_digits
  uint32_t num_blocks,         // the number of blocks used by `computeBlockDigitCounts` kernel
  uint32_t blocks_per_slice,
  // outputs:
  uint32_t* kthCounts          // size: num_slices * blocks_per_slice == num_blocks
) {
  CUDA_KERNEL_LOOP_TYPE(idx, num_blocks, uint32_t) {
    uint32_t slice_idx = idx / blocks_per_slice;
    Bitwise desired = doLdg(desires + slice_idx);
    Bitwise desired_digit = at::cuda::Bitfield<Bitwise>::getBitfield(desired, 0, RADIX_BITS);
    kthCounts[idx] = doLdg(counts + idx * RADIX_DIGITS + desired_digit);
  }
}

#else // USE_ROCM

// ROCm path: Assumption: k can not be larger than UINT32_MAX
template <typename Bitwise, typename T>
C10_LAUNCH_BOUNDS_1(RADIX_DIGITS)  // one thread per digit
__global__ void computeBlockwiseWithinKCounts(
  Bitwise* desires_in,          // size: num_slices
  short* counts,                // size: num_slices * blocks_per_slice * radix_digits
  uint32_t* ks_to_find_in,      // size: num_slices
  uint32_t blocks_per_slice,
  int current_bit,
  bool largest,
  // outputs:
  uint32_t* withinKCounts,      // size: num_slices * blocks_per_slice == num_blocks
  T* kthValues,                 // size: num_slices, only write when current_bit reaches 0
  uint32_t* ks_to_find_out,
  Bitwise* desires_out,
  uint32_t* kthCounts,          // ROCm: added kthCounts output
  uint32_t num_blocks
) {
  // This kernel should be launched with the same number of blocks as the `computeBlockDigitCounts` kernel.
  int tidx = threadIdx.x;
  uint32_t block_idx = getLinearBlockId<uint32_t>();
  uint32_t slice_idx = block_idx / blocks_per_slice;

  // The grid is computed from `getGridFromTiles`, when there are lots of
  // elements, we will use both blockIdx.x and blockIdx.y, and maybe blockIdx.z
  // when this is the case, the number of blocks that we are launching can be
  // more than the number of blocks we need. So we need to check the range of
  // `block_idx`.
  if (block_idx >= num_blocks) {
    return;
  }


  __shared__ Bitwise desired;
  uint32_t k_to_find = ks_to_find_in[slice_idx];

  // Use hipCUB BlockScan for efficient inclusive scan
  typedef ROCM_HIPCUB(at_cuda_detail::cub)::BlockScan<uint32_t, RADIX_DIGITS> BlockScan;
  __shared__ typename BlockScan::TempStorage scan_storage;

  // Build per-slice digit totals in shared memory and compute inclusive cumsum
  __shared__ uint32_t digit_totals[RADIX_DIGITS];
  if (tidx < RADIX_DIGITS) {
    uint32_t sum = 0;
    // Accumulate counts across all blocks in the slice for this digit

    // AMD optimization: Improve memory access pattern to reduce latency
    // Access pattern: counts[base + blk * RADIX_DIGITS + tidx]
    // For large blocks_per_slice, this loop dominates kernel time
    // Unroll by 4 to improve instruction-level parallelism and hide latency
    const short* count_ptr = counts + slice_idx * blocks_per_slice * RADIX_DIGITS + tidx;
    uint32_t blk = 0;
    // Process 4 blocks at a time to improve ILP
    for (; blk + 3 < blocks_per_slice; blk += 4) {
      uint32_t v0 = count_ptr[0 * RADIX_DIGITS];
      uint32_t v1 = count_ptr[1 * RADIX_DIGITS];
      uint32_t v2 = count_ptr[2 * RADIX_DIGITS];
      uint32_t v3 = count_ptr[3 * RADIX_DIGITS];
      sum += v0 + v1 + v2 + v3;
      count_ptr += 4 * RADIX_DIGITS;
    }
    // Handle remaining blocks
    for (; blk < blocks_per_slice; ++blk) {
      sum += count_ptr[0];
      count_ptr += RADIX_DIGITS;
    }
    digit_totals[tidx] = sum;
  }
  __syncthreads();

  // Use hipCUB BlockScan for efficient inclusive scan (replaces manual scan)
  uint32_t digit_total = (tidx < RADIX_DIGITS) ? digit_totals[tidx] : 0u;
  uint32_t digit_count_cumsum;
  BlockScan(scan_storage).InclusiveSum(digit_total, digit_count_cumsum);
  __syncthreads();
  if (tidx < RADIX_DIGITS) {
    digit_totals[tidx] = digit_count_cumsum;
  }
  __syncthreads();

  if (tidx < RADIX_DIGITS) {
    // digit_count_cumsum already contains the correct value from BlockScan
    uint32_t digit_count_cumsum_left = (tidx == 0) ? 0 : digit_totals[tidx - 1];

    // if not the last pass: update desired and ks_to_find
    // if last pass: write out the kth value
    // only one thread in block enters this condition
    if (digit_count_cumsum_left < k_to_find && k_to_find <= digit_count_cumsum) {
      desired = desires_in[slice_idx];
      desired = at::cuda::Bitfield<Bitwise>::setBitfield(desired, tidx, current_bit, RADIX_BITS);
      // let a single block per slice update the values
      if (block_idx == slice_idx * blocks_per_slice) {
        desires_out[slice_idx] = desired;
        if (current_bit > 0) {
          ks_to_find_out[slice_idx] = k_to_find - digit_count_cumsum_left;
        } else {
          kthValues[slice_idx] = TopKTypeConfig<T>::deconvert(desired);
        }
      }
    }
  }
  __syncthreads();

  Bitwise desired_digit = at::cuda::Bitfield<Bitwise>::getBitfield(desired, current_bit, RADIX_BITS);

  // if largest, then only threads that has tidx > desired_digit are active
  // if !largest, then only threads that has tidx < desired_digit are active
  // each active thread will read the count for its corresponding, and
  // do warp reduction followed by shared memory reduction to get the total count
  // non-active thread should not load, and non-active warp should not do reduction.
  bool warp_is_active, thread_is_active;
  int warp = tidx / C10_WARP_SIZE;
  if (largest) {
    int end_of_warp = warp * C10_WARP_SIZE + C10_WARP_SIZE - 1;
    warp_is_active = end_of_warp > desired_digit;
    thread_is_active = tidx > desired_digit;
  } else {
    int start_of_warp = warp * C10_WARP_SIZE;
    warp_is_active = start_of_warp < desired_digit;
    thread_is_active = tidx < desired_digit;
  }
  uint32_t count = 0;
  if (warp_is_active) {
    if (thread_is_active) {
      count = doLdg(counts + block_idx * RADIX_DIGITS + tidx);
    }
    for (int offset = C10_WARP_SIZE / 2; offset > 0; offset /= 2) {
      count += WARP_SHFL_DOWN(count, offset);
    }
  }

  constexpr int num_warps = RADIX_DIGITS / C10_WARP_SIZE;
  __shared__ uint32_t warp_counts[num_warps];
  if (tidx % C10_WARP_SIZE == 0) {
    warp_counts[warp] = count;
  }
  __syncthreads();

  CUDA_KERNEL_ASSERT(RADIX_DIGITS < C10_WARP_SIZE * C10_WARP_SIZE);
  if (warp != 0) {
    return;
  }
  count = 0;
  if (tidx < num_warps) {
    count = warp_counts[tidx];
  }
  for (int offset = num_warps / 2; offset > 0; offset /= 2) {
    count += WARP_SHFL_DOWN(count, offset);
  }
  if (tidx == 0) {
    withinKCounts[block_idx] += count;
  }

  // On the last pass (current_bit == 0), write out block-wise kthCounts directly
  if (tidx == 0 && current_bit == 0) {
    Bitwise desired_digit0 = at::cuda::Bitfield<Bitwise>::getBitfield(desired, 0, RADIX_BITS);
    kthCounts[block_idx] = doLdg(counts + block_idx * RADIX_DIGITS + desired_digit0);
  }
}

#endif // USE_ROCM

template <typename T, typename IndexType, int Dim>
C10_LAUNCH_BOUNDS_1(BLOCK_THREADS)
__global__ void gatherTopK(at::cuda::detail::TensorInfo<const T, IndexType> input,
                           IndexType inputSliceSize,
                           IndexType outputSliceSize, // aka `k`
                           bool largest,

                           uint32_t numInputSlices,
                           IndexType inputWithinSliceStride,

                           at::cuda::detail::TensorInfo<T, IndexType> topK,
                           IndexType topKWithinSliceStride,

                           at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
                           IndexType indicesWithinSliceStride,

                           uint32_t items_per_thread,
                           uint32_t blocks_per_slice,

                           T *kthValues,
                           uint32_t* withinKCounts,
                           uint32_t* kthCounts,
                           uint32_t num_blocks) {

  uint32_t items_per_block = items_per_thread * BLOCK_THREADS;
  uint32_t tidx = threadIdx.x;
  uint32_t block_idx = getLinearBlockId<uint32_t>();

  // The grid is computed from `getGridFromTiles`, when there are lots of
  // elements, we will use both blockIdx.x and blockIdx.y, and maybe blockIdx.z
  // when this is the case, the number of blocks that we are launching can be
  // more than the number of blocks we need. So we need to check the range of
  // `block_idx`.
  if (block_idx >= num_blocks) {
    return;
  }

  uint32_t slice_idx = block_idx / blocks_per_slice;
  uint32_t blk_idx_in_slice = block_idx % blocks_per_slice;

  items_per_thread = (blk_idx_in_slice + 1 < blocks_per_slice)
      ? items_per_thread
      : at::ceil_div((int64_t)(inputSliceSize - blk_idx_in_slice * items_per_block), (int64_t)BLOCK_THREADS);

  // Find the start offset for our slice
  IndexType sliceStartIndex =
    at::cuda::detail::IndexToOffset<const T, IndexType, Dim>::get(slice_idx, input);
  IndexType topKSliceStartIndex =
    at::cuda::detail::IndexToOffset<T, IndexType, Dim>::get(slice_idx, topK);
  IndexType indicesSliceStartIndex =
    at::cuda::detail::IndexToOffset<int64_t, IndexType, Dim>::get(slice_idx, indices);

  const T* inputSliceStart = &input.data[sliceStartIndex];
  T* topKSliceStart = &topK.data[topKSliceStartIndex];
  int64_t* indicesSliceStart = &indices.data[indicesSliceStartIndex];

  // Find the k-th highest element in our input
  T kthValue = kthValues[slice_idx];
  const auto kthValueConverted = at::native::TopKTypeConfig<T>::convert(kthValue);

  // Find the start index in output tensor of this block
#if !defined(USE_ROCM)
  // CUDA path: Use pre-computed prefix sums from CUB
  uint32_t startWithinK = 0;
  if (blk_idx_in_slice > 0) {
    startWithinK = withinKCounts[block_idx - 1];
  }
  uint32_t startKth = withinKCounts[slice_idx * blocks_per_slice + blocks_per_slice - 1];
  if (blk_idx_in_slice > 0) {
    startKth += kthCounts[block_idx - 1];
  }
#else
  // ROCm path: Compute prefix sums inline with unrolled loop
  __shared__ uint32_t slice_prefix_within;
  __shared__ uint32_t slice_prefix_kth;
  __shared__ uint32_t slice_total_within;
  if (threadIdx.x == 0) {
    uint32_t prefix_within = 0;
    uint32_t prefix_kth = 0;
    uint32_t total_within = 0;
    uint32_t slice_offset = slice_idx * blocks_per_slice;
    // AMD optimization: Unroll prefix sum loop to improve ILP
    // This loop iterates up to 245 times for 1M case, causing memory latency
    const uint32_t* within_ptr = withinKCounts + slice_offset;
    const uint32_t* kth_ptr = kthCounts + slice_offset;
    uint32_t blk = 0;
    // Process 4 blocks at a time
    for (; blk + 3 < blocks_per_slice; blk += 4) {
      uint32_t w0 = within_ptr[0];
      uint32_t w1 = within_ptr[1];
      uint32_t w2 = within_ptr[2];
      uint32_t w3 = within_ptr[3];
      total_within += w0 + w1 + w2 + w3;
      if (blk < blk_idx_in_slice) {
        prefix_within += w0;
        prefix_kth += kth_ptr[0];
      }
      if (blk + 1 < blk_idx_in_slice) {
        prefix_within += w1;
        prefix_kth += kth_ptr[1];
      }
      if (blk + 2 < blk_idx_in_slice) {
        prefix_within += w2;
        prefix_kth += kth_ptr[2];
      }
      if (blk + 3 < blk_idx_in_slice) {
        prefix_within += w3;
        prefix_kth += kth_ptr[3];
      }
      within_ptr += 4;
      kth_ptr += 4;
    }
    // Handle remaining blocks
    for (; blk < blocks_per_slice; ++blk) {
      uint32_t within_val = *within_ptr++;
      total_within += within_val;
      if (blk < blk_idx_in_slice) {
        prefix_within += within_val;
        prefix_kth += *kth_ptr;
      }
      kth_ptr++;
    }
    slice_prefix_within = prefix_within;
    slice_prefix_kth = prefix_kth;
    slice_total_within = total_within;
  }
  __syncthreads();

  uint32_t startWithinK = slice_prefix_within;
  uint32_t startKth = slice_total_within + slice_prefix_kth;
#endif // USE_ROCM

  // Read input, select topk out and write
  typedef cub::BlockScan<uint32_t, BLOCK_THREADS> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  for (int i = 0; i < items_per_thread; ++i) {
    // Find the start offset for this slice
    IndexType idx = blk_idx_in_slice * items_per_block + i * BLOCK_THREADS + tidx;
    T val;
    int withinK = 0;
    int kth = 0;
    if (idx < inputSliceSize) {
      val = doLdg(inputSliceStart + idx * inputWithinSliceStride);
      const auto valConverted = at::native::TopKTypeConfig<T>::convert(val);
      withinK = (largest ? valConverted > kthValueConverted : valConverted < kthValueConverted);
      kth = (valConverted == kthValueConverted);
    }

    uint32_t withinKIndex;
    uint32_t numWithinK;
    BlockScan(temp_storage).ExclusiveSum(withinK, withinKIndex, numWithinK);
    __syncthreads();
    if (withinK) {
      uint32_t offset = withinKIndex + startWithinK;
      topKSliceStart[offset * topKWithinSliceStride] = val;
      indicesSliceStart[offset * indicesWithinSliceStride] = idx;
    }
    startWithinK += numWithinK;

    if (startKth < outputSliceSize) {
      uint32_t kthIndex;
      uint32_t numKth;
      BlockScan(temp_storage).ExclusiveSum(kth, kthIndex, numKth);
      __syncthreads();
      if (kth) {
        uint32_t offset = kthIndex + startKth;
        if (offset < outputSliceSize) {
          topKSliceStart[offset * topKWithinSliceStride] = val;
          indicesSliceStart[offset * indicesWithinSliceStride] = idx;
        }
      }
      startKth += numKth;
    }
  }
}

int get_items_per_thread(uint64_t num_slices, uint64_t slice_size) {
  // Occupancy of this kernel is limited by registers per thread
  // Platform-specific tuning for optimal register pressure
#if defined(USE_ROCM)
  // AMD RDNA/CDNA architecture: measured via rocprof for mbtopk kernels
  // MI250X has different register file organization than NVIDIA
  // - More VGPRs available per thread (256 vs 255 on NVIDIA)
  // - Wave64 execution model requires different occupancy tuning
  // Empirically tuned for large 1D TopK (1M elements, k=8 case)
  constexpr int REGS_PER_THREAD = 48;  // Higher register usage acceptable on AMD
#else
  constexpr int REGS_PER_THREAD = 40;  // from nsight launch statistics (NVIDIA)
#endif
  constexpr int REGS_PER_BLOCK = REGS_PER_THREAD * BLOCK_THREADS;

  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  int mpc = prop->multiProcessorCount;
  int regs_per_mp = prop->regsPerMultiprocessor;
  int max_blocks_per_mp = prop->maxBlocksPerMultiProcessor;
  int blocks_per_mp = std::min(regs_per_mp / REGS_PER_BLOCK, max_blocks_per_mp);

  // Calculate items_per_thread to maximize GPU utilization
  int64_t items_per_thread = at::ceil_div((int64_t)(slice_size * num_slices), (int64_t)(mpc * blocks_per_mp * BLOCK_THREADS));

#if defined(USE_ROCM)
  // AMD-specific optimization: For large 1D slices, use higher items_per_thread
  // to significantly reduce blocks_per_slice, which reduces overhead in:
  // - computeBlockwiseWithinKCounts accumulation loop (lines 730-750)
  // - gatherTopK prefix sum loop (lines 940-981)
  // Goal: Keep blocks_per_slice under 100 for optimal performance
  if (num_slices <= 4 && slice_size >= 800000) {
    // Very large arrays (800k+): Aggressively increase items_per_thread
    // For 1M elements: items_per_thread=32 → blocks_per_slice=123
    // For 1M elements: items_per_thread=48 → blocks_per_slice=82
    // For 1M elements: items_per_thread=64 → blocks_per_slice=62
    items_per_thread = std::max(items_per_thread, (int64_t)48);
  } else if (num_slices <= 4 && slice_size >= 500000) {
    // Large arrays (500k-800k): Moderately increase items_per_thread
    items_per_thread = std::max(items_per_thread, (int64_t)32);
  } else if (num_slices <= 4 && slice_size >= 250000) {
    // Medium-large arrays (250k-500k): Slightly increase items_per_thread
    items_per_thread = std::max(items_per_thread, (int64_t)24);
  }
#endif

  // Clamp to valid range [MIN, MAX] (AMD: [4, 96], CUDA: [4, 64])
  items_per_thread = std::max(MIN_ITEMS_PER_THREAD, std::min((int)items_per_thread, MAX_ITEMS_PER_THREAD));
  return items_per_thread;
}

class BlockIdxToKey {
  uint32_t blocks_per_slice;
public:
  BlockIdxToKey(uint32_t blocks_per_slice): blocks_per_slice(blocks_per_slice) {}
  __device__ __forceinline__ uint32_t operator()(uint32_t blk) const {
    return blk / blocks_per_slice;
  }
};

template <typename T, typename IndexType, int Dim>
void launch(
    at::cuda::detail::TensorInfo<const T, IndexType> input,
    IndexType inputSliceSize,
    IndexType outputSliceSize, // aka `k`
    bool largest,

    uint32_t numInputSlices,
    IndexType inputWithinSliceStride,

    at::cuda::detail::TensorInfo<T, IndexType> topK,
    IndexType topKWithinSliceStride,

    at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
    IndexType indicesWithinSliceStride) {
  auto stream = c10::cuda::getCurrentCUDAStream();

  // configure items_per_thread based on device architecture and input size
  int items_per_thread = get_items_per_thread(numInputSlices, inputSliceSize);
  int items_per_block = items_per_thread * BLOCK_THREADS;

  using Bitwise = typename TopKTypeConfig<T>::RadixType;
  uint32_t blocks_per_slice = at::ceil_div((int64_t)inputSliceSize, (int64_t)items_per_block);
  uint32_t num_blocks = numInputSlices * blocks_per_slice;

  // temporary storage
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();

  auto kthValues_buffer = allocator.allocate(numInputSlices * sizeof(T));
  T* kthValues = reinterpret_cast<T*>(kthValues_buffer.get());

  TORCH_CHECK(blocks_per_slice <= std::numeric_limits<uint32_t>::max(), "blocks_per_slice larger than uint32 maximum is not supported");


  auto ks_to_find_buffer = allocator.allocate(2 * numInputSlices * sizeof(uint32_t));
  uint32_t* ks_to_find = reinterpret_cast<uint32_t*>(ks_to_find_buffer.get());
  uint32_t k_to_find = largest ? inputSliceSize - outputSliceSize + 1: outputSliceSize;
  fill<uint32_t><<<std::min(((int64_t)numInputSlices + 511) / 512, (int64_t)1073741824), 512, 0, stream>>>(
    ks_to_find, k_to_find, numInputSlices);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto desired_buffer = allocator.allocate(2 * numInputSlices * sizeof(Bitwise));
  Bitwise* desired = reinterpret_cast<Bitwise*>(desired_buffer.get());

  auto counts_buffer = allocator.allocate(num_blocks * RADIX_DIGITS * sizeof(short));
  short* counts = reinterpret_cast<short*>(counts_buffer.get());
  static_assert(MAX_ITEMS_PER_THREAD * BLOCK_THREADS < std::numeric_limits<short>::max(),
    "blockwise counter too large");

#if !defined(USE_ROCM)
  // CUDA path: Allocate digit_cum_sum buffer
  auto digit_cum_sum_buffer = allocator.allocate(numInputSlices * RADIX_DIGITS * sizeof(uint32_t));
  uint32_t* digit_cum_sum = reinterpret_cast<uint32_t*>(digit_cum_sum_buffer.get());
  AT_CUDA_CHECK(cudaMemsetAsync(digit_cum_sum, 0, numInputSlices * RADIX_DIGITS * sizeof(uint32_t), stream));
#else
  // ROCm path: No separate digit cumsum buffer; fused into computeBlockwiseWithinKCounts
#endif

  auto withinKCounts_buffer = allocator.allocate(num_blocks * sizeof(uint32_t));
  uint32_t* withinKCounts = reinterpret_cast<uint32_t*>(withinKCounts_buffer.get());
  AT_CUDA_CHECK(cudaMemsetAsync(withinKCounts, 0, num_blocks * sizeof(uint32_t), stream));

  auto kthCounts_buffer = allocator.allocate(num_blocks * sizeof(uint32_t));
  uint32_t* kthCounts = reinterpret_cast<uint32_t*>(kthCounts_buffer.get());

  Bitwise desiredMask = 0;
  dim3 grid;
  TORCH_INTERNAL_ASSERT(getGridFromTiles(num_blocks, grid), "Too many slices for topk");
  dim3 block(BLOCK_THREADS);

  uint32_t * ks_to_find_in = ks_to_find;
  uint32_t * ks_to_find_out = ks_to_find + numInputSlices;
  Bitwise * desired_in = desired;
  Bitwise * desired_out = desired + numInputSlices;

  // iterate radix bits for multiple passes
  for (int current_bit = sizeof(T) * 8 - RADIX_BITS; current_bit >= 0; current_bit -= RADIX_BITS) {
    computeBlockDigitCounts<T, IndexType, Bitwise, Dim><<<grid, block, 0, stream>>>(
        input,
        inputSliceSize,
        ks_to_find_in, // unused arg
        numInputSlices,
        inputWithinSliceStride,
        current_bit,
        items_per_thread,
        blocks_per_slice,
        desiredMask,
        desired_in,
        counts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

#if !defined(USE_ROCM)
    // CUDA path: Separate kernel for digit cumsum
    computeDigitCumSum<<<numInputSlices, RADIX_DIGITS, 0, stream>>>(counts, digit_cum_sum, blocks_per_slice);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // CUDA path: Call computeBlockwiseWithinKCounts with digit_cum_sum, without kthCounts
    computeBlockwiseWithinKCounts<Bitwise, T><<<grid, RADIX_DIGITS, 0, stream>>>(
      desired_in, counts, digit_cum_sum, ks_to_find_in, blocks_per_slice, current_bit, largest,
      withinKCounts, kthValues, ks_to_find_out, desired_out, num_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
    // ROCm path: Fused version (no digit_cum_sum kernel, includes kthCounts)
    // we unconditionally call this kernel to update desired/ks_to_find/kthValues
    // if cub supports scan_by_key we additionally do k counts
    computeBlockwiseWithinKCounts<Bitwise, T><<<grid, RADIX_DIGITS, 0, stream>>>(
      desired_in, counts, ks_to_find_in, blocks_per_slice, current_bit, largest,
      withinKCounts, kthValues, ks_to_find_out, desired_out, kthCounts, num_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
    // swap desired/ks_to_find in and out for next iter
    auto tmp_desired = desired_in;
    desired_in = desired_out;
    desired_out = tmp_desired;
    auto tmp_ks = ks_to_find_in;
    ks_to_find_in = ks_to_find_out;
    ks_to_find_out = tmp_ks;
    desiredMask = at::cuda::Bitfield<Bitwise>::setBitfield(desiredMask, RADIX_MASK, current_bit, RADIX_BITS);
  }
  desired = desired_in;

#if !defined(USE_ROCM)
  // CUDA path: Separate kernel for kthCounts
  computeBlockwiseKthCounts<Bitwise><<<std::min(((int64_t)numInputSlices + 255) / 256, (int64_t)1073741824), 256, 0, stream>>>(
    desired, counts, num_blocks, blocks_per_slice, kthCounts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // CUDA path: Do a prefix scan of withinKCounts and kthCounts using slice_idx as keys to get the starting index of each block
  using counting_iter_t = ATEN_CUB_COUNTING_ITERATOR(uint32_t, uint32_t);
  using slice_idx_iter_t = ATEN_CUB_TRANSFORM_ITERATOR(uint32_t, BlockIdxToKey, counting_iter_t);
  slice_idx_iter_t slice_idx_iter(counting_iter_t(0), BlockIdxToKey(blocks_per_slice));
  at::cuda::cub::inclusive_sum_by_key(slice_idx_iter, withinKCounts, withinKCounts, num_blocks);
  at::cuda::cub::inclusive_sum_by_key(slice_idx_iter, kthCounts, kthCounts, num_blocks);
#else
  // ROCm path: kthCounts already produced in computeBlockwiseWithinKCounts at last pass
  // No CUB scans; prefix computation fused into gatherTopK
#endif

  // copy topk values to output tensor
  gatherTopK<T, IndexType, Dim><<<grid, block, 0, stream>>>(
    input, inputSliceSize, outputSliceSize, largest, numInputSlices, inputWithinSliceStride,
    topK, topKWithinSliceStride, indices, indicesWithinSliceStride, items_per_thread,
    blocks_per_slice, kthValues, withinKCounts, kthCounts, num_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace mbtopk

bool should_use_multiblock(int64_t num_slices, int64_t slice_size) {
  if (num_slices > std::numeric_limits<uint32_t>::max() ||
      slice_size > std::numeric_limits<uint32_t>::max()) return false;
  // This heuristics is based on the experiment in https://github.com/pytorch/pytorch/pull/74267
  return (num_slices <= 20 && slice_size >= 20000) ||
      (num_slices > 20 && num_slices <= 40 && slice_size >= 10000) ||
      (num_slices > 40 && num_slices <= 80 && slice_size >= 8000) ||
      (num_slices > 80 && num_slices < 200 && slice_size >= 5000) ||
      (num_slices >= 200 && num_slices < 800 && slice_size >= 3000) ||
      (num_slices >= 800 && num_slices <= 4000 && slice_size >= 800) ||
      (num_slices > 4000 && slice_size >= 400);
}

bool should_use_warp_topk(int64_t slice_size, int64_t k) {
#if !defined(USE_ROCM) || !HAS_WARP_MERGE_SORT()
  return false;
#else
  if (slice_size <= 0 || k <= 0 || slice_size > warptopk::MAX_WARP_TOPK_SLICE) {
    return false;
  }
  // Use WarpMergeSort for small slices
  // WarpMergeSort has O(n log n) complexity and is efficient for small sizes
  // Conservative threshold: slice_size <= 256 shows consistent improvements
  return slice_size <= 256;
#endif
}

void launch_gather_topk_kernel(
    const TensorBase& self, int64_t k, int64_t dim, bool largest,
    const TensorBase& values, const TensorBase& indices) {
  int numDims = self.dim();
  numDims = numDims == 0 ? 1 : numDims;
  TORCH_CHECK(numDims <= MAX_DIMS, "input tensor has too many dimensions");
  int64_t sliceSize = self.dim() == 0 ? 1 : self.size(dim);

  auto input = self.contiguous();
  // static_cast is required to ensure that the correct type (INDEX_T)
  // is provided to the kernel for the arguments.
#define RUN_K(INDEX_T, DIM, LAUNCH_FUNCTION_NAME)                       \
  LAUNCH_FUNCTION_NAME<scalar_t, INDEX_T, DIM>(                         \
      inputInfo,                                                        \
      static_cast<INDEX_T>(sliceSize),                                  \
      static_cast<INDEX_T>(k),                                          \
      largest,                                                          \
      static_cast<INDEX_T>(numInputSlices),                             \
      /* The actual dimension that the k-selection is running in */     \
      /* may have changed from collapseDims() */                        \
      static_cast<INDEX_T>(inputInfo.strides[collapseInputDim]),        \
      topKInfo,                                                         \
      static_cast<INDEX_T>(topKInfo.strides[collapseTopKDim]),          \
      indicesInfo,                                                      \
      static_cast<INDEX_T>(indicesInfo.strides[collapseIndicesDim]));

#if defined(USE_ROCM) && HAS_WARP_MERGE_SORT()
#define RUN_MB(INDEX_T, DIM)                                              \
  if (should_use_warp_topk(sliceSize, k)) {                               \
    RUN_K(INDEX_T, DIM, warptopk::launch);                                \
  } else if (should_use_multiblock(numInputSlices, sliceSize)) {          \
    RUN_K(INDEX_T, DIM, mbtopk::launch);                                  \
  } else {                                                                \
    RUN_K(INDEX_T, DIM, sbtopk::launch);                                  \
  }
#else
#define RUN_MB(INDEX_T, DIM)                                              \
  if (should_use_multiblock(numInputSlices, sliceSize)) {                 \
    RUN_K(INDEX_T, DIM, mbtopk::launch);                                  \
  } else {                                                                \
    RUN_K(INDEX_T, DIM, sbtopk::launch);                                  \
  }
#endif

#define RUN_DIM(INDEX_T)                        \
  if (allDims == 1) {                           \
    RUN_MB(INDEX_T, 1);                         \
  } else if (allDims == 2) {                    \
    RUN_MB(INDEX_T, 2);                         \
  } else if (allDims == 3) {                    \
    RUN_MB(INDEX_T, 3);                         \
  } else {                                      \
    RUN_MB(INDEX_T, -1);                        \
  }

#define RUN_T(INDEX_T)                                                    \
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "topk_out_cuda", [&] { \
    at::cuda::detail::TensorInfo<const scalar_t, INDEX_T> inputInfo =     \
      at::cuda::detail::getTensorInfo<const scalar_t, INDEX_T>(input);    \
    at::cuda::detail::TensorInfo<scalar_t, INDEX_T> topKInfo =            \
      at::cuda::detail::getTensorInfo<scalar_t, INDEX_T>(values);         \
    at::cuda::detail::TensorInfo<int64_t, INDEX_T> indicesInfo =          \
      at::cuda::detail::getTensorInfo<int64_t, INDEX_T>(indices);         \
    /* tensorInfoLegacyIfScalar*/                                         \
    if (!input.dim()) {                                                   \
      inputInfo.dims = 1;                                                 \
      inputInfo.sizes[0] = 1;                                             \
      inputInfo.strides[0] = 1;                                           \
      topKInfo.dims = 1;                                                  \
      topKInfo.sizes[0] = 1;                                              \
      topKInfo.strides[0] = 1;                                            \
      indicesInfo.dims = 1;                                               \
      indicesInfo.sizes[0] = 1;                                           \
      indicesInfo.strides[0] = 1;                                         \
    }                                                                     \
    /* We use these structures solely to find the offset to */            \
    /* each slice we are operating on */                                  \
    inputInfo.sizes[dim] = 1;                                             \
    topKInfo.sizes[dim] = 1;                                              \
    indicesInfo.sizes[dim] = 1;                                           \
    /* stash the stride of dim because it can be accidentally collapsed */ \
    auto strideInput = inputInfo.strides[dim];                            \
    auto strideTopK = topKInfo.strides[dim];                              \
    auto strideIndices = indicesInfo.strides[dim];                        \
    /* Collapse all other dims */                                         \
    int collapseInputDim = inputInfo.collapseDims(dim);                   \
    int collapseTopKDim = topKInfo.collapseDims(dim);                     \
    int collapseIndicesDim = indicesInfo.collapseDims(dim);               \
    /* restore stride in case it was collapsed */                         \
    inputInfo.strides[collapseInputDim] = strideInput;                    \
    topKInfo.strides[collapseTopKDim] = strideTopK;                       \
    indicesInfo.strides[collapseIndicesDim] = strideIndices;              \
    int64_t numInputSlices = 1;                                           \
    for (int i = 0; i < inputInfo.dims; ++i) {                            \
      numInputSlices *= inputInfo.sizes[i];                               \
    }                                                                     \
                                                                          \
    /* This is used as a template parameter to calculate indices. */      \
    /* We only specialize it if all collapsed dim sizes are the */        \
    /* same; otherwise, we use -1 which is the specialization */          \
    /* parameter for arbitrary dimensions */                              \
    int allDims = inputInfo.dims;                                         \
    if (topKInfo.dims != allDims || indicesInfo.dims != allDims) {        \
      allDims = -1;                                                       \
    }                                                                     \
                                                                          \
    RUN_DIM(INDEX_T);                                                     \
  });

  // the below is safe with 0-dimensional tensors because it is based on
  // TensorInfo which implicitly expands to 1-dimensional.
  if (input.numel() > 0) {
    if (at::cuda::detail::canUse32BitIndexMath(input) &&
        at::cuda::detail::canUse32BitIndexMath(values) &&
        at::cuda::detail::canUse32BitIndexMath(indices)) {
      RUN_T(uint32_t);
    } else {
      RUN_T(uint64_t);
    }
  }
#undef RUN_T
#undef RUN_DIM
#undef RUN_MB
#undef RUN_K
}

} // at::native
