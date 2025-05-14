#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/TensorTopK.h>
#include <ATen/core/TensorBase.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/ScanUtils.cuh>
#include <ATen/cuda/AsmUtils.cuh>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/native/cuda/SortingRadixSelect.cuh>
#include <ATen/cuda/cub.cuh>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/detail/KernelUtils.h>

#include <c10/macros/Macros.h>

using namespace at::native;

namespace at::native {

// TODO: remove this when CUDA <11.6 is no longer supported
bool disable_sort_for_topk() {
  return CUB_SUPPORTS_SCAN_BY_KEY();
}

namespace sbtopk { // single_block_topk

template <typename T>
struct AddOp {
  __device__ __forceinline__ T operator()(T const &lhs, T const &rhs) {
    return (lhs + rhs);
  }
};

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
#if defined(USE_ROCM)
  __shared__ int smem[64];
#else
  __shared__ int smem[32]; // one per each warp, up to warp limit
#endif
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

};

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
static_assert(RADIX_DIGITS <= BLOCK_THREADS, "radixFindKthValues kernel requires RADIX_DIGITS <= BLOCK_THREADS");
constexpr int MIN_ITEMS_PER_THREAD = 4;
constexpr int MAX_ITEMS_PER_THREAD = 64;

template <typename T, typename IndexType>
__global__ void fill(T* x, T value, IndexType size) {
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (IndexType i = idx; i < size; i += gridDim.x * blockDim.x) {
    x[i] = value;
  }
}

// find the kth smallest value,
// for largest topk, k_to_find = slice_size - k + 1
template <typename T, typename IndexType, typename Bitwise, int Dim>
C10_LAUNCH_BOUNDS_1(BLOCK_THREADS)
__global__ void radixFindKthValues(
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

// Assumption: k can not be larger than UINT32_MAX
template <typename Bitwise, typename T>
C10_LAUNCH_BOUNDS_1(RADIX_DIGITS)  // one thread per digit
__global__ void computeBlockwiseWithinKCounts(
  Bitwise* desires_in,          // size: num_slices
  short* counts,             // size: num_slices * blocks_per_slice * radix_digits
  uint32_t* ks_to_find_in,  // size: num_slices
  uint32_t blocks_per_slice,
  int current_bit,
  bool largest,
  // outputs:
  uint32_t* withinKCounts,  // size: num_slices * blocks_per_slice == num_blocks
  T* kthValues,           // size: num_slices, only write when current_bit reaches 0
  uint32_t* ks_to_find_out,
  Bitwise* desires_out,
  uint32_t num_blocks
) {
  // This kernel should be launched with the same number of blocks as the `radixFindKthValues` kernel.
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
  typedef cub::BlockScan<uint32_t, BLOCK_THREADS> BlockScan;
  union __align__(16) TempStorage {
    uint32_t digit_count_cumsum[RADIX_DIGITS]; // only used if this it the last block for this slice
    typename BlockScan::TempStorage scan_storage;
  };
  __shared__ TempStorage temp_storage;

  // accumulates counters from multiple blocks
  uint32_t digit_count = 0;
  if (tidx < RADIX_DIGITS) {
    for (int blk = 0; blk < blocks_per_slice; ++blk) {
      digit_count += counts[(slice_idx * blocks_per_slice + blk) * RADIX_DIGITS + tidx];
    }
  }

  // compute the block-wide inclusive prefix sum
  uint32_t digit_count_cumsum;
  BlockScan(temp_storage.scan_storage).InclusiveSum(digit_count, digit_count_cumsum);
  __syncthreads();
  // every thread also need the perfix_sum of it's left value for comparison, so save a copy in shared mem
  if (tidx < RADIX_DIGITS) {
    temp_storage.digit_count_cumsum[tidx] = digit_count_cumsum;
  }
  __syncthreads();

  __shared__ Bitwise desired;
  uint32_t k_to_find = ks_to_find_in[slice_idx];

  if (tidx < RADIX_DIGITS) {
    uint32_t digit_count_cumsum_left = (tidx == 0) ? 0 : temp_storage.digit_count_cumsum[tidx - 1];

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

#if !CUB_SUPPORTS_SCAN_BY_KEY()
  return;
#endif

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

#if CUB_SUPPORTS_SCAN_BY_KEY()
// Assumption: slice_size can not be larger than UINT32_MAX
template <typename Bitwise>
__global__ void computeBlockwiseKthCounts(
  Bitwise* desires,            // size: num_slices
  short* counts,               // size: num_slices * blocks_per_slice * radix_digits
  uint32_t num_blocks,         // the number of blocks used by `radixFindKthValues` kernel
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
  uint32_t startWithinK = 0;
  if (blk_idx_in_slice > 0) {
    startWithinK = withinKCounts[block_idx - 1];
  }
  uint32_t startKth = withinKCounts[slice_idx * blocks_per_slice + blocks_per_slice - 1];
  if (blk_idx_in_slice > 0) {
    startKth += kthCounts[block_idx - 1];
  }

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
#endif

int get_items_per_thread(uint64_t num_slices, uint64_t slice_size) {
  // occupancy of this kernel is limited by registers per threads
  constexpr int REGS_PER_THREAD = 40; // from nsight launch statistics
  constexpr int REGS_PER_BLOCK = REGS_PER_THREAD * BLOCK_THREADS;
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  int mpc = prop->multiProcessorCount;
  int regs_per_mp = prop->regsPerMultiprocessor;
  int max_blocks_per_mp = prop->maxBlocksPerMultiProcessor;
  int blocks_per_mp = std::min(regs_per_mp / REGS_PER_BLOCK, max_blocks_per_mp);
  int64_t items_per_thread = at::ceil_div((int64_t)(slice_size * num_slices), (int64_t)(mpc * blocks_per_mp * BLOCK_THREADS));
  items_per_thread = std::max(MIN_ITEMS_PER_THREAD, std::min((int)items_per_thread, MAX_ITEMS_PER_THREAD)); // clamp to (4, 64)
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
  auto semaphores_buffer = allocator.allocate(numInputSlices * sizeof(uint32_t));
  uint32_t* semaphores = reinterpret_cast<uint32_t*>(semaphores_buffer.get());
  AT_CUDA_CHECK(cudaMemsetAsync(semaphores, 0, numInputSlices * sizeof(uint32_t), stream));

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

#if CUB_SUPPORTS_SCAN_BY_KEY()
  auto withinKCounts_buffer = allocator.allocate(num_blocks * sizeof(uint32_t));
  uint32_t* withinKCounts = reinterpret_cast<uint32_t*>(withinKCounts_buffer.get());
  AT_CUDA_CHECK(cudaMemsetAsync(withinKCounts, 0, num_blocks * sizeof(uint32_t), stream));

  auto kthCounts_buffer = allocator.allocate(num_blocks * sizeof(uint32_t));
  uint32_t* kthCounts = reinterpret_cast<uint32_t*>(kthCounts_buffer.get());
#else
  uint32_t* withinKCounts = nullptr;
#endif

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
    radixFindKthValues<T, IndexType, Bitwise, Dim><<<grid, block, 0, stream>>>(
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
    // we unconditionally call this kernel to update desired/ks_to_find/kthValues
    // if cub supports scan_by_key we additionally do k counts
    computeBlockwiseWithinKCounts<Bitwise, T><<<grid, RADIX_DIGITS, 0, stream>>>(
      desired_in, counts, ks_to_find_in, blocks_per_slice, current_bit, largest, withinKCounts, kthValues, ks_to_find_out, desired_out, num_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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

#if CUB_SUPPORTS_SCAN_BY_KEY()
  computeBlockwiseKthCounts<Bitwise><<<std::min(((int64_t)numInputSlices + 255) / 256, (int64_t)1073741824), 256, 0, stream>>>(
    desired, counts, num_blocks, blocks_per_slice, kthCounts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // Do a prefix scan of withinKCounts and kthCounts using slice_idx as keys to get the starting index of each block
  using counting_iter_t = ATEN_CUB_COUNTING_ITERATOR(uint32_t, uint32_t);
  using slice_idx_iter_t = ATEN_CUB_TRANSFORM_ITERATOR(uint32_t, BlockIdxToKey, counting_iter_t);
  slice_idx_iter_t slice_idx_iter(counting_iter_t(0), BlockIdxToKey(blocks_per_slice));
  at::cuda::cub::inclusive_sum_by_key(slice_idx_iter, withinKCounts, withinKCounts, num_blocks);
  at::cuda::cub::inclusive_sum_by_key(slice_idx_iter, kthCounts, kthCounts, num_blocks);
  // copy topk values to output tensor
  gatherTopK<T, IndexType, Dim><<<grid, block, 0, stream>>>(
    input, inputSliceSize, outputSliceSize, largest, numInputSlices, inputWithinSliceStride,
    topK, topKWithinSliceStride, indices, indicesWithinSliceStride, items_per_thread,
    blocks_per_slice, kthValues, withinKCounts, kthCounts, num_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  // Find topk values based on kth values
  {
    dim3 grid;
    TORCH_INTERNAL_ASSERT(getGridFromTiles(numInputSlices, grid), "Too many slices for topk");
    int warp_size = at::cuda::warp_size();
    dim3 block(std::min(at::ceil_div((int64_t)inputSliceSize, (int64_t)warp_size) * (int64_t)warp_size, (int64_t)1024));
    sbtopk::gatherTopK<T, IndexType, Dim, /* WithKthValues= */true><<<grid, block, 0, stream>>>(
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
        kthValues);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
#endif
}

} // namespace mbtopk

bool should_use_multiblock(int64_t num_slices, int64_t slice_size) {
  if (num_slices > std::numeric_limits<uint32_t>::max() ||
      slice_size > std::numeric_limits<uint32_t>::max()) return false;
#if CUB_SUPPORTS_SCAN_BY_KEY()
  // This heuristics is based on the experiment in https://github.com/pytorch/pytorch/pull/74267
  return (num_slices <= 20 && slice_size >= 20000) ||
      (num_slices > 20 && num_slices <= 40 && slice_size >= 10000) ||
      (num_slices > 40 && num_slices <= 80 && slice_size >= 8000) ||
      (num_slices > 80 && num_slices < 200 && slice_size >= 5000) ||
      (num_slices >= 200 && num_slices < 800 && slice_size >= 3000) ||
      (num_slices >= 800 && num_slices <= 4000 && slice_size >= 800) ||
      (num_slices > 4000 && slice_size >= 400);
#else
  // This heuristics is based on the experiment in https://github.com/pytorch/pytorch/pull/71081
  return (num_slices <= 400 && slice_size >= 5000) ||
      (num_slices > 400 && num_slices < 4000 && slice_size >= 1000) ||
      (num_slices >= 4000 && slice_size >= 300);
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

#define RUN_MB(INDEX_T, DIM)                                            \
  if (should_use_multiblock(numInputSlices, sliceSize)) {               \
    RUN_K(INDEX_T, DIM, mbtopk::launch);                                \
  } else {                                                              \
    RUN_K(INDEX_T, DIM, sbtopk::launch);                                \
  }

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
    auto strideTopK = topKInfo.strides[dim];                              \
    auto strideIndices = indicesInfo.strides[dim];                        \
    /* Collapse all other dims */                                         \
    int collapseInputDim = inputInfo.collapseDims(dim);                   \
    int collapseTopKDim = topKInfo.collapseDims(dim);                     \
    int collapseIndicesDim = indicesInfo.collapseDims(dim);               \
    /* restore stride in case it was collapsed */                         \
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
    // Based on required index size, run the algorithm with the
    // appropriate index type
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
#undef RUN_K
}

} // at::native
