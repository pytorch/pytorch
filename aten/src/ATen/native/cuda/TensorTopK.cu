#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/TensorTopK.h>
#include <ATen/core/TensorBase.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/ScanUtils.cuh>
#include <ATen/cuda/AsmUtils.cuh>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/native/cuda/SortingRadixSelect.cuh>
#include <ATen/native/cuda/SortUtils.cuh>
#include <ATen/cuda/cub.cuh>
#include <c10/cuda/CUDACachingAllocator.h>

#include <c10/macros/Macros.h>

using namespace at::native;

namespace at {
namespace native {
namespace sbtopk { // single_block_topk

template <typename T>
struct AddOp {
  __device__ __forceinline__ T operator()(T const &lhs, T const &rhs) {
    return (lhs + rhs);
  }
};

template <typename T, typename IndexType, int Dim, bool WithKthValues>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void gatherTopK(at::cuda::detail::TensorInfo<T, IndexType> input,
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
    at::cuda::detail::IndexToOffset<T, IndexType, Dim>::get(slice, input);
  IndexType topKSliceStartIndex =
    at::cuda::detail::IndexToOffset<T, IndexType, Dim>::get(slice, topK);
  IndexType indicesSliceStartIndex =
    at::cuda::detail::IndexToOffset<int64_t, IndexType, Dim>::get(slice, indices);

  T* inputSliceStart = &input.data[sliceStartIndex];
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
    at::cuda::detail::TensorInfo<T, IndexType> input,
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

constexpr int BLOCK_THREADS = 256;

// Over what radix we are selecting values
constexpr int RADIX_BITS = 8;
constexpr int RADIX_DIGITS = 1 << RADIX_BITS; // 2 ^ RADIX_BITS
constexpr int RADIX_MASK = (RADIX_DIGITS - 1);
static_assert(RADIX_DIGITS <= BLOCK_THREADS, "radixFindKthValues kernel requires RADIX_DIGITS <= BLOCK_THREADS");

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
    at::cuda::detail::TensorInfo<T, IndexType> input,
    IndexType slice_size,
    IndexType* ks_to_find, // size: num_slices

    IndexType num_slices,
    IndexType withinSliceStride,

    int current_bit,
    int items_per_thread,
    IndexType blocks_per_slice,
    Bitwise desiredMask,

    // outputs
    uint32_t* semaphores,  // size: num_slices
    Bitwise* desires,      // size: num_slices
    IndexType* counts,     // size: num_slices * blocks_per_slice * radix_digits
    T* kthValues           // size: num_slices, only write when current_bit reaches 0
  ) {

  int items_per_block = items_per_thread * BLOCK_THREADS;
  int tidx = threadIdx.x;
  IndexType block_idx = getLinearBlockId<IndexType>();
  IndexType slice_idx = block_idx / blocks_per_slice;
  IndexType blk_idx_in_slice = block_idx % blocks_per_slice;
  if (slice_idx >= num_slices) {
    return;
  }

  Bitwise desired = desires[slice_idx];
  IndexType k_to_find = ks_to_find[slice_idx];
  IndexType slice_start_index = at::cuda::detail::IndexToOffset<T, IndexType, Dim>::get(slice_idx, input);
  T* data = &input.data[slice_start_index];

  typedef cub::BlockScan<IndexType, BLOCK_THREADS> BlockScan;
  union __align__(16) TempStorage {
    uint32_t digit_counters[RADIX_DIGITS];
    IndexType digit_count_cumsum[RADIX_DIGITS]; // only used if this it the last block for this slice
    typename BlockScan::TempStorage scan_storage;
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

  // collect digit counts and store in shared memorey
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
  IndexType digit_count = 0;
  if (tidx < RADIX_DIGITS) {
    digit_count = temp_storage.digit_counters[tidx];
  }

  // if blocks_per_slice == 1, there is no need to do cross-block reduction
  // in this case counts saved at registers instead of global memory
  if (blocks_per_slice > 1) {

    if (tidx < RADIX_DIGITS) {
      counts[block_idx * RADIX_DIGITS + tidx] = digit_count;
    }
    __threadfence(); // make sure writes are globally visible
    __syncthreads(); // make sure all writes are finished before update semaphores
  }

  // the last block of each slice accumulates counters from multiple blocks and updates desired and ks_to_find
  __shared__ bool s_is_last_block_done;

  if (tidx == 0) {
    if (blocks_per_slice == 1) {
      s_is_last_block_done = true;
    } else {
      uint32_t blocks_finished_old = atomicAdd(&semaphores[slice_idx], 1);
      s_is_last_block_done = (blocks_finished_old == blocks_per_slice - 1);
    }
  }

  __syncthreads();

  if (!s_is_last_block_done)
    return;

  // accumulates counters from multiple blocks
  if (tidx < RADIX_DIGITS && blocks_per_slice > 1) {
    digit_count = 0;
    for (int blk = 0; blk < blocks_per_slice; ++blk) {
      digit_count += counts[(slice_idx * blocks_per_slice + blk) * RADIX_DIGITS + tidx];
    }
  }

  // compute the block-wide inclusive prefix sum
  IndexType digit_count_cumsum;
  BlockScan(temp_storage.scan_storage).InclusiveSum(digit_count, digit_count_cumsum);
  __syncthreads();
  // every thread also need the perfix_sum of it's left value for comparison, so save a copy in shared mem
  if (tidx < RADIX_DIGITS) {
    temp_storage.digit_count_cumsum[tidx] = digit_count_cumsum;
  }
  __syncthreads();

  if (tidx < RADIX_DIGITS) {
    IndexType digit_count_cumsum_left = (tidx == 0) ? 0 : temp_storage.digit_count_cumsum[tidx - 1];

    // if not the last pass: update desired and ks_to_find
    // if last pass: write out the kth value
    if (digit_count_cumsum_left < k_to_find && k_to_find <= digit_count_cumsum) {
      desired = at::cuda::Bitfield<Bitwise>::setBitfield(desired, tidx, current_bit, RADIX_BITS);
      if (current_bit > 0) {
        desires[slice_idx] = desired;
        ks_to_find[slice_idx] = k_to_find - digit_count_cumsum_left;
      } else {
        kthValues[slice_idx] = TopKTypeConfig<T>::deconvert(desired);
      }
    }
  }

  // reset semaphores for the next pass
  if (tidx == 0) {
    semaphores[slice_idx] = 0;
  }
};

int get_items_per_thread(uint64_t num_slices, uint64_t slice_size) {
  // occupancy of this kernel is limited by registers per threads
  constexpr int REGS_PER_THREAD = 40; // from nsight launch statistics
  constexpr int REGS_PER_BLOCK = REGS_PER_THREAD * BLOCK_THREADS;
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  int mpc = prop->multiProcessorCount;
#if defined(USE_ROCM)
  int regs_per_mp = prop->regsPerBlock;
  int max_blocks_per_mp = 32;
#else
  int regs_per_mp = prop->regsPerMultiprocessor;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  int max_blocks_per_mp = prop->maxBlocksPerMultiProcessor;
#else
  int max_blocks_per_mp = 32;
#endif
#endif
  int blocks_per_mp = std::min(regs_per_mp / REGS_PER_BLOCK, max_blocks_per_mp);
  int64_t items_per_thread = at::ceil_div((int64_t)(slice_size * num_slices), (int64_t)(mpc * blocks_per_mp * BLOCK_THREADS));
  items_per_thread = std::max(4, std::min((int)items_per_thread, 64)); // clamp to (4, 64)
  return items_per_thread;
}

template <typename T, typename IndexType, int Dim>
void launch(
    at::cuda::detail::TensorInfo<T, IndexType> input,
    IndexType inputSliceSize,
    IndexType outputSliceSize, // aka `k`
    bool largest,

    IndexType numInputSlices,
    IndexType inputWithinSliceStride,

    at::cuda::detail::TensorInfo<T, IndexType> topK,
    IndexType topKWithinSliceStride,

    at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
    IndexType indicesWithinSliceStride) {

  // configure items_per_thread based on device architecture and input size
  int items_per_thread = get_items_per_thread(numInputSlices, inputSliceSize);
  int items_per_block = items_per_thread * BLOCK_THREADS;

  using Bitwise = typename TopKTypeConfig<T>::RadixType;
  int64_t blocks_per_slice = at::ceil_div((int64_t)inputSliceSize, (int64_t)items_per_block);
  int64_t num_blocks = numInputSlices * blocks_per_slice;

  // temporary storage
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();

  auto kthValues_buffer = allocator.allocate(numInputSlices * sizeof(T));
  T* kthValues = reinterpret_cast<T*>(kthValues_buffer.get());

  TORCH_CHECK(blocks_per_slice <= std::numeric_limits<uint32_t>::max(), "blocks_per_slice larger than uint32 maximum is not supported");
  auto semaphores_buffer = allocator.allocate(numInputSlices * sizeof(uint32_t));
  uint32_t* semaphores = reinterpret_cast<uint32_t*>(semaphores_buffer.get());
  AT_CUDA_CHECK(cudaMemsetAsync(semaphores, 0, numInputSlices * sizeof(uint32_t), c10::cuda::getCurrentCUDAStream()));

  auto ks_to_find_buffer = allocator.allocate(numInputSlices * sizeof(IndexType));
  IndexType* ks_to_find = reinterpret_cast<IndexType*>(ks_to_find_buffer.get());
  IndexType k_to_find = largest ? inputSliceSize - outputSliceSize + 1: outputSliceSize;
  fill<IndexType><<<std::min((numInputSlices + 511) / 512, (IndexType)65535), 512, 0, c10::cuda::getCurrentCUDAStream()>>>(
    ks_to_find, k_to_find, numInputSlices);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto desired_buffer = allocator.allocate(numInputSlices * sizeof(Bitwise));
  Bitwise* desired = reinterpret_cast<Bitwise*>(desired_buffer.get());

  auto counts_buffer = allocator.allocate(num_blocks * RADIX_DIGITS * sizeof(IndexType));
  IndexType* counts = reinterpret_cast<IndexType*>(counts_buffer.get());

  Bitwise desiredMask = 0;
  dim3 grid;
  TORCH_INTERNAL_ASSERT(getGridFromTiles(num_blocks, grid), "Too many slices for topk");
  dim3 block(BLOCK_THREADS);

  // iterate radix bits for multiple passes
  for (int current_bit = sizeof(T) * 8 - RADIX_BITS; current_bit >= 0; current_bit -= RADIX_BITS) {
    radixFindKthValues<T, IndexType, Bitwise, Dim><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input,
        inputSliceSize,
        ks_to_find,
        numInputSlices,
        inputWithinSliceStride,
        current_bit,
        items_per_thread,
        blocks_per_slice,
        desiredMask,
        semaphores,
        desired,
        counts,
        kthValues);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    desiredMask = at::cuda::Bitfield<Bitwise>::setBitfield(desiredMask, RADIX_MASK, current_bit, RADIX_BITS);
  }

  // Find topk values based on kth values
  {
    dim3 grid;
    TORCH_INTERNAL_ASSERT(getGridFromTiles(numInputSlices, grid), "Too many slices for topk");
    int warp_size = at::cuda::warp_size();
    dim3 block(std::min(at::ceil_div((int64_t)inputSliceSize, (int64_t)warp_size) * (int64_t)warp_size, (int64_t)1024));
    sbtopk::gatherTopK<T, IndexType, Dim, /* WithKthValues= */true><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
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
}

} // namespace mbtopk

bool should_use_multiblock(int64_t num_slices, int64_t slice_size) {
  // This heuristics is based on the experiment in https://github.com/pytorch/pytorch/pull/71081
  return (num_slices <= 400 && slice_size >= 5000) ||
      (num_slices >= 400 && num_slices < 4000 && slice_size >= 1000) ||
      (num_slices >= 4000 && slice_size >= 300);
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
    at::cuda::detail::TensorInfo<scalar_t, INDEX_T> inputInfo =           \
      at::cuda::detail::getTensorInfo<scalar_t, INDEX_T>(input);          \
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
} // at
