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
namespace sbtopk {  // single_block_topk

template <typename T>
struct AddOp {
  __device__ __forceinline__ T operator()(T const &lhs, T const &rhs) {
    return (lhs + rhs);
  }
};

template <typename T, typename IndexType, int Dim, bool Order, bool WithKthValues>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void gatherTopK(at::cuda::detail::TensorInfo<T, IndexType> input,
                           IndexType inputSliceSize,
                           IndexType outputSliceSize, // aka `k`

                           IndexType numInputSlices,
                           IndexType inputWithinSliceStride,

                           at::cuda::detail::TensorInfo<T, IndexType> topK,
                           IndexType numTopKSlices,
                           IndexType topKWithinSliceStride,

                           at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
                           IndexType indicesWithinSliceStride,
                           T* kthValues) {
  (void)numTopKSlices; // Suppress unused variable warning
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
    radixSelect<T, typename TopKTypeConfig<T>::RadixType, IndexType, Order>(
      inputSliceStart, outputSliceSize,
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
    if (Order) {
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

template <typename T, typename IndexType, int Dim, bool Order>
void launch(
    at::cuda::detail::TensorInfo<T, IndexType> input,
    IndexType inputSliceSize,
    IndexType outputSliceSize, // aka `k`

    IndexType numInputSlices,
    IndexType inputWithinSliceStride,

    at::cuda::detail::TensorInfo<T, IndexType> topK,
    IndexType numTopKSlices, // TODO never used
    IndexType topKWithinSliceStride,

    at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
    IndexType indicesWithinSliceStride) {

    dim3 grid;
    TORCH_INTERNAL_ASSERT(getGridFromTiles(numInputSlices, grid), "Too many slices to sort");
    dim3 block(std::min(
        at::ceil_div((int64_t)inputSliceSize, (int64_t)C10_WARP_SIZE) * (int64_t)C10_WARP_SIZE, (int64_t)1024));
    gatherTopK<T, IndexType, Dim, Order, /* WithKthValues= */false><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input,
        inputSliceSize,
        outputSliceSize,
        numInputSlices,
        inputWithinSliceStride,
        topK,
        numTopKSlices,
        topKWithinSliceStride,
        indices,
        indicesWithinSliceStride,
        nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace at::native::sbtopk

namespace mbtopk {  // multi_block_topk

constexpr int BLOCK_THREADS = 128;
// in principle, we could write at most 255 into digit counter (in shared mem) with unsigned char type
// TODO tune this, maybe smaller
constexpr int MAX_ITEMS_PER_THREAD = 64;
constexpr int ITEMS_PER_BLOCK = BLOCK_THREADS * MAX_ITEMS_PER_THREAD;

// Over what radix we are selecting values
constexpr int RADIX_BITS = 6; // digits are base-(2 ^ RADIX_BITS)
constexpr int RADIX_DIGITS = 1 << RADIX_BITS; // 2 ^ RADIX_BITS
constexpr int RADIX_MASK = (RADIX_DIGITS - 1);

// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
struct BlockPrefixCallbackOp {
  // Running prefix
  int running_total;
  // Constructor
  __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide scan.
  __device__ int operator()(int block_aggregate) {
    int old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

template <typename T>
__global__ void fill(T* x, T value, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
    x[i] = value;
  }
}

template <typename T, typename IndexType, typename Bitwise, int Dim, bool Order, int RADIX_BITS>
C10_LAUNCH_BOUNDS_1(BLOCK_THREADS)
__global__ void radixFindKthValues(
    at::cuda::detail::TensorInfo<T, IndexType> input,
    IndexType inputSliceSize,
    IndexType* ksToFind, // aka `k`

    IndexType numInputSlices,
    IndexType withinSliceStride,

    int current_bit,
    IndexType blocks_per_slice,
    Bitwise desiredMask,

    // outputs
    int* semaphores,
    Bitwise* desires,
    IndexType* counts,
    T* kthValues // only writes when current_bit reaches 0
) {
  constexpr int RADIX_DIGITS = 1 << RADIX_BITS; // 2 ^ RADIX_BITS
  constexpr int PACKING_RATIO = sizeof(int) / sizeof(unsigned char);
  constexpr int COUNTER_LANES = RADIX_DIGITS / PACKING_RATIO;

  int tidx = threadIdx.x;
  IndexType block_idx = getLinearBlockId<IndexType>();
  IndexType slice_idx = block_idx / blocks_per_slice;
  IndexType blk_idx_in_slice = block_idx % blocks_per_slice;
  if (slice_idx >= numInputSlices) {
    return;
  }

  Bitwise desired = desires[slice_idx];
  IndexType kToFind = ksToFind[slice_idx];
  IndexType sliceStartIndex = at::cuda::detail::IndexToOffset<T, IndexType, Dim>::get(slice_idx, input);
  T* data = &input.data[sliceStartIndex];

  typedef cub::BlockScan<IndexType, BLOCK_THREADS> BlockScan;
  union __align__(16) TempStorage {
    unsigned char thread_counters[COUNTER_LANES][BLOCK_THREADS]
                                 [PACKING_RATIO]; // threads in a warp is guaranteed to access different banks
    uint32_t packed_thread_counters[COUNTER_LANES][BLOCK_THREADS];
    struct {
      IndexType digit_count_cumsum[RADIX_DIGITS];
      typename BlockScan::TempStorage temp_storage;
    } scan_storage;
  };
  __shared__ TempStorage temp_storage;

  // reset temp_storage
  for (int i = 0; i < COUNTER_LANES; ++i) {
    temp_storage.packed_thread_counters[i][tidx] = 0;
  }
  __syncthreads();

  int items_per_thread = (blk_idx_in_slice + 1 < blocks_per_slice)
      ? MAX_ITEMS_PER_THREAD
      : at::ceil_div((int64_t)(inputSliceSize - blk_idx_in_slice * ITEMS_PER_BLOCK), (int64_t)BLOCK_THREADS);

  // collect counts and store in shared memorey for each thread
  for (int i = 0; i < items_per_thread; ++i) {
    // Find the start offset for our slice
    IndexType idx = (tidx + i * BLOCK_THREADS + blk_idx_in_slice * ITEMS_PER_BLOCK) * withinSliceStride;
    if (idx < inputSliceSize) {
      T val_ori = doLdg(&data[idx]);
      Bitwise val = TopKTypeConfig<T>::convert(doLdg(&data[idx]));
      bool hasVal = ((val & desiredMask) == (desired & desiredMask));
      Bitwise digit = at::cuda::Bitfield<Bitwise>::getBitfield(val, current_bit, RADIX_BITS);
      if (hasVal) {
        temp_storage.thread_counters[digit / PACKING_RATIO][tidx][digit % PACKING_RATIO]++; // threads in a warp is guaranteed to access different banks
      }
    }
  }

  __syncthreads();

  // extract counts and write count out
  for (int i = 0; i < (RADIX_DIGITS + BLOCK_THREADS - 1) / BLOCK_THREADS; ++i) {
    // every thread collects one overall digit count stored in shared mem for each thread
    int digit_count = 0;
    int digit = i * BLOCK_THREADS + tidx;
    if (digit < RADIX_DIGITS) {
      for (int j = 0, idx = tidx; j < BLOCK_THREADS; ++j, idx = (idx + 1) % BLOCK_THREADS) { // every thread access different bank
        digit_count += temp_storage.thread_counters[digit / PACKING_RATIO][idx][digit % PACKING_RATIO];
      }
      counts[block_idx * RADIX_DIGITS + digit] = digit_count;
    }
  }

  __threadfence(); // make sure writes are globally visible
  __syncthreads();

  __shared__ bool s_is_last_block_done;
  __shared__ bool s_desired_found;

  if (tidx == 0) {
    int blocks_finished_old = atomicAdd(&semaphores[slice_idx], 1);
    s_is_last_block_done = (blocks_finished_old == blocks_per_slice - 1);
    s_desired_found = false;
  }

  __syncthreads();

  // last block for each slice accumulate counts from blocks and update desired
  if (s_is_last_block_done) {
    // sum block counts
    BlockPrefixCallbackOp prefix_op(0);

    auto post_process = [&](int digit) {
      IndexType digit_count = 0;
      IndexType& digit_count_cumsum = digit_count;
      for (int blk = 0; blk < blocks_per_slice; ++blk) {
        digit_count += counts[(slice_idx * blocks_per_slice + blk) * RADIX_DIGITS + digit];
      }

      // Collectively compute the block-wide exclusive prefix sum
      BlockScan(temp_storage.scan_storage.temp_storage).InclusiveSum(digit_count, digit_count_cumsum, prefix_op);
      __syncthreads();
      temp_storage.scan_storage.digit_count_cumsum[digit] = digit_count_cumsum;
      __syncthreads();

      // update desired
      IndexType digit_count_cumsum_left;
      if (Order) {
        digit_count_cumsum_left =
            (digit == RADIX_DIGITS - 1) ? 0 : temp_storage.scan_storage.digit_count_cumsum[digit + 1];
      } else {
        digit_count_cumsum_left = (digit == 0) ? 0 : temp_storage.scan_storage.digit_count_cumsum[digit - 1];
      }
      if (digit_count_cumsum_left < kToFind && kToFind <= digit_count_cumsum) {
        desired = at::cuda::Bitfield<Bitwise>::setBitfield(desired, digit, current_bit, RADIX_BITS);
        if (current_bit > 0) {
          desires[slice_idx] = desired;
          ksToFind[slice_idx] = kToFind - digit_count_cumsum_left;
        } else {
          kthValues[slice_idx] = TopKTypeConfig<T>::deconvert(desired);
        }
        s_desired_found = true;
      }
      __syncthreads();
    };

    if (Order) {
      for (int digit = RADIX_DIGITS - tidx - 1; digit >= 0 && !s_desired_found; digit -= BLOCK_THREADS) {
        post_process(digit);
      }
    } else {
      for (int digit = tidx; digit < RADIX_DIGITS && !s_desired_found; digit += BLOCK_THREADS) {
        post_process(digit);
      }
    }

    if (tidx == 0) {
      semaphores[slice_idx] = 0;
    }
  }
};

template <typename T, typename IndexType, int Dim, bool Order>
void launch(
    at::cuda::detail::TensorInfo<T, IndexType> input,
    IndexType inputSliceSize,
    IndexType outputSliceSize, // aka `k`

    IndexType numInputSlices,
    IndexType inputWithinSliceStride,

    at::cuda::detail::TensorInfo<T, IndexType> topK,
    IndexType numTopKSlices, // TODO never used
    IndexType topKWithinSliceStride,

    at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
    IndexType indicesWithinSliceStride) {
  using Bitwise = typename TopKTypeConfig<T>::RadixType;
  int64_t blocks_per_slice = at::ceil_div((int64_t)inputSliceSize, (int64_t)ITEMS_PER_BLOCK);
  int64_t num_blocks = numInputSlices * blocks_per_slice;

  // temporary storage
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();

  auto kthValues_buffer = allocator.allocate(numInputSlices * sizeof(T));
  T* kthValues = reinterpret_cast<T*>(kthValues_buffer.get());
  AT_CUDA_CHECK(cudaMemsetAsync(kthValues, 0, numInputSlices * sizeof(T), c10::cuda::getCurrentCUDAStream()));

  auto semaphores_buffer = allocator.allocate(numInputSlices * sizeof(int));
  int* semaphores = reinterpret_cast<int*>(semaphores_buffer.get());
  AT_CUDA_CHECK(cudaMemsetAsync(semaphores, 0, numInputSlices * sizeof(int), c10::cuda::getCurrentCUDAStream()));

  auto ksToFind_buffer = allocator.allocate(numInputSlices * sizeof(IndexType));
  IndexType* ksToFind = reinterpret_cast<IndexType*>(ksToFind_buffer.get());
  fill<IndexType><<<std::min((numInputSlices + 511) / 512, (IndexType)65535), 512, 0, c10::cuda::getCurrentCUDAStream()>>>(
          ksToFind, outputSliceSize, numInputSlices);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto desired_buffer = allocator.allocate(numInputSlices * sizeof(Bitwise));
  Bitwise* desired = reinterpret_cast<Bitwise*>(desired_buffer.get());

  auto counts_buffer = allocator.allocate(num_blocks * RADIX_DIGITS * sizeof(IndexType));
  IndexType* counts = reinterpret_cast<IndexType*>(counts_buffer.get());

  Bitwise desiredMask = 0;
  dim3 grid;
  TORCH_INTERNAL_ASSERT(getGridFromTiles(num_blocks, grid), "Too many slices to sort");
  dim3 block(BLOCK_THREADS);

#define RUN_K(BIT)                                             \
  radixFindKthValues<T, IndexType, Bitwise, Dim, Order, BIT>   \
      <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>( \
          input,                                               \
          inputSliceSize,                                      \
          ksToFind,                                            \
          numInputSlices,                                      \
          inputWithinSliceStride,                              \
          current_bit,                                         \
          blocks_per_slice,                                    \
          desiredMask,                                         \
          semaphores,                                          \
          desired,                                             \
          counts,                                              \
          kthValues);                                          \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define RUN_BIT()                                              \
  if (radix_bits == 6) {                                       \
    RUN_K(6);                                                  \
  } else if (radix_bits == 4) {                                \
    RUN_K(4);                                                  \
  } else if (radix_bits == 2) {                                \
    RUN_K(2);                                                  \
  } else {                                                     \
    TORCH_INTERNAL_ASSERT(                                     \
        false, "RADIX_BIT ", radix_bits, " is not supported"); \
  }

  int current_bit = sizeof(T) * 8 - RADIX_BITS;
  int radix_bits = RADIX_BITS;
  for (; current_bit > 0; current_bit -= RADIX_BITS) {
    RUN_BIT();
    desiredMask = at::cuda::Bitfield<Bitwise>::setBitfield(
        desiredMask, RADIX_MASK, current_bit, RADIX_BITS);
  }
  radix_bits = current_bit + RADIX_BITS;
  current_bit = 0;
  RUN_BIT();

#undef RUN_BIT
#undef RUN_K

  // Find topk values based on kth value
  {
    dim3 grid;
    TORCH_INTERNAL_ASSERT(getGridFromTiles(numInputSlices, grid), "Too many slices to sort");
    dim3 block(std::min(at::ceil_div((int64_t)inputSliceSize, (int64_t)C10_WARP_SIZE) * (int64_t)C10_WARP_SIZE, (int64_t)1024));
    sbtopk::gatherTopK<T, IndexType, Dim, Order, /* WithKthValues= */true><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
            input,
            inputSliceSize,
            outputSliceSize,
            numInputSlices,
            inputWithinSliceStride,
            topK,
            numTopKSlices,
            topKWithinSliceStride,
            indices,
            indicesWithinSliceStride,
            kthValues);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

} // namespace at::native::mbtopk

void launch_gather_topk_kernel(
    const TensorBase& self, int64_t k, int64_t dim, bool largest, bool sorted,
    const TensorBase& values, const TensorBase& indices) {
  int numDims = self.dim();
  numDims = numDims == 0 ? 1 : numDims;
  TORCH_CHECK(numDims <= MAX_DIMS, "input tensor has too many dimensions");
  int64_t sliceSize = self.dim() == 0 ? 1 : self.size(dim);

  auto input = self.contiguous();
  // static_cast is required to ensure that the correct type (INDEX_T)
  // is provided to the kernel for the arguments.

  bool should_use_multiblock_per_slice = true;  // TODO logic to disbatch based on heuristics result
#define RUN_K(INDEX_T, DIM, DIR, LAUNCH_FUNCTION_NAME)                  \
  LAUNCH_FUNCTION_NAME<scalar_t, INDEX_T, DIM, DIR>(                    \
      inputInfo,                                                        \
      static_cast<INDEX_T>(sliceSize),                                  \
      static_cast<INDEX_T>(k),                                          \
      static_cast<INDEX_T>(inputSlices),                                \
      /* The actual dimension that the k-selection is running in */     \
      /* may have changed from collapseDims() */                        \
      static_cast<INDEX_T>(inputInfo.strides[collapseInputDim]),        \
      topKInfo,                                                         \
      static_cast<INDEX_T>(topKSlices),                                 \
      static_cast<INDEX_T>(topKInfo.strides[collapseTopKDim]),          \
      indicesInfo,                                                      \
      static_cast<INDEX_T>(indicesInfo.strides[collapseIndicesDim]));

#define RUN_MB(INDEX_T, DIM, DIR)               \
  if (should_use_multiblock_per_slice) {        \
    RUN_K(INDEX_T, DIM, DIR, mbtopk::launch);   \
  } else {                                      \
    RUN_K(INDEX_T, DIM, DIR, sbtopk::launch);   \
  }

#define RUN_DIR(INDEX_T, DIM)                   \
  if (largest) {                                \
    RUN_MB(INDEX_T, DIM, true);                 \
  } else {                                      \
    RUN_MB(INDEX_T, DIM, false);                \
  }

#define RUN_DIM(INDEX_T)                        \
  if (allDims == 1) {                           \
    RUN_DIR(INDEX_T, 1);                        \
  } else if (allDims == 2) {                    \
    RUN_DIR(INDEX_T, 2);                        \
  } else if (allDims == 3) {                    \
    RUN_DIR(INDEX_T, 3);                        \
  } else {                                      \
    RUN_DIR(INDEX_T, -1);                       \
  }

#define RUN_T(INDEX_T)                                                  \
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
    int64_t inputSlices = 1;                                              \
    for (int i = 0; i < inputInfo.dims; ++i) {                            \
      inputSlices *= inputInfo.sizes[i];                                  \
    }                                                                     \
    int64_t topKSlices = 1;                                               \
    for (int i = 0; i < topKInfo.dims; ++i) {                             \
      topKSlices *= topKInfo.sizes[i];                                    \
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
#undef RUN_DIR
#undef RUN_K
}

} // at::native
} // at
