#include "THCReduceApplyUtils.cuh"
#include "THCTensorMath.h"
#include "THCTensorSort.h"
#include "THCAsmUtils.cuh"
#include "THCScanUtils.cuh"
#include "THCTensorTypeUtils.cuh"
#include <algorithm> // for std::min

#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

// Converts a float to an integer representation with the same
// sorting; i.e., for floats f1, f2:
// if f1 < f2 then convert(f1) < convert(f2)
// We use this to enable radix selection of floating-point values.
// This also gives a relative order for NaNs, but that's ok, as they
// will all be adjacent
struct FloatToSortedInt {
  inline __host__ __device__ FloatToSortedInt() {}

  inline __device__ unsigned int convert(float v) const {
    unsigned int x = __float_as_int(v);
    unsigned int mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;

    return (x ^ mask);
  }

  inline __device__ float deconvert(unsigned int v) const {
    unsigned int mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;

    return __int_as_float(v ^ mask);
  }
};

// This function counts the distribution of all input values in a
// slice we are selecting by radix digit at `radixDigitPos`, but only
// those that pass the filter `((v & desiredMask) == desired)`.
// This produces and broadcasts the seen counts for a single block only.
// `smem` must have at least `RadixSize` elements.
template <typename DataType, typename BitDataType,
          typename IndexType, typename CountType,
          typename RadixConverter, int RadixSize, int RadixBits>
__device__ void countRadixUsingMask(const RadixConverter& conv,
                                    CountType counts[RadixSize],
                                    CountType* smem,
                                    BitDataType desired,
                                    BitDataType desiredMask,
                                    int radixDigitPos,
                                    IndexType sliceSize,
                                    IndexType withinSliceStride,
                                    DataType* data) {
  // Clear out per-thread counts from a previous round
#pragma unroll
  for (int i = 0; i < RadixSize; ++i) {
    counts[i] = 0;
  }

  if (threadIdx.x < RadixSize) {
    smem[threadIdx.x] = 0;
  }
  __syncthreads();

  // Scan over all the data. Upon a read, the warp will accumulate
  // counts per each digit in the radix using warp voting.
  for (IndexType i = threadIdx.x; i < sliceSize; i += blockDim.x) {
    BitDataType val = conv.convert(doLdg(&data[i * withinSliceStride]));

    bool hasVal = ((val & desiredMask) == desired);
    unsigned int digitInRadix = getBitfield(val, radixDigitPos, RadixBits);

#pragma unroll
    for (unsigned int j = 0; j < RadixSize; ++j) {
      bool vote = hasVal && (digitInRadix == j);
      counts[j] += __popc(__ballot(vote));
    }
  }

  // Now, for each warp, sum values
  if (getLaneId() == 0) {
#pragma unroll
    for (unsigned int i = 0; i < RadixSize; ++i) {
      atomicAdd(&smem[i], counts[i]);
    }
  }

  __syncthreads();

  // For each thread, read in the total counts
#pragma unroll
  for (unsigned int i = 0; i < RadixSize; ++i) {
    counts[i] = smem[i];
  }

  __syncthreads();
}

// Over what radix we are selecting values
#define RADIX_BITS 2 // digits are base-(2 ^ RADIX_BITS)
#define RADIX_SIZE 4 // 2 ^ RADIX_BITS
#define RADIX_MASK (RADIX_SIZE - 1)

// This finds the unique value `v` that matches the pattern
// ((v & desired) == desiredMask) in our sorted int format
template <typename DataType, typename IndexType, typename RadixConverter>
__device__ float findPattern(const RadixConverter& conv,
                             DataType* smem,
                             DataType* data,
                             IndexType sliceSize,
                             IndexType withinSliceStride,
                             unsigned int desired,
                             unsigned int desiredMask) {
  if (threadIdx.x < 32) {
    smem[threadIdx.x] = (DataType) 0;
  }
  __syncthreads();

  // All threads participate in the loop, in order to sync on the flag
  IndexType numIterations = THCRoundUp(sliceSize, (IndexType) blockDim.x);
  for (IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < sliceSize);
    DataType v = inRange ? doLdg(&data[i * withinSliceStride]) : (DataType) 0;

    if (inRange && ((conv.convert(v) & desiredMask) == desired)) {
      // There should not be conflicts if we are using findPattern,
      // since the result is unique
      smem[0] = (DataType) 1;
      smem[1] = v; // can't use val as the flag, since it could be 0
    }

    __syncthreads();

    DataType found = smem[0];
    DataType val = smem[1];

    __syncthreads();

    // Check to see if a thread found the value
    if (found != (DataType) 0) {
      // all threads return this value
      return val;
    }
  }

  // should not get here
  assert(false);
  return (DataType) 0;
}

// Returns the top-Kth element found in the data using radix selection
template <typename DataType, typename BitDataType, typename IndexType,
          typename RadixConverter, bool Order>
__device__ void radixSelect(const RadixConverter& conv,
                            DataType* data,
                            IndexType k,
                            IndexType sliceSize,
                            IndexType withinSliceStride,
                            int* smem,
                            DataType* topK) {
  // Per-thread buckets into which we accumulate digit counts in our
  // radix
  int counts[RADIX_SIZE];

  // We only consider elements x such that (x & desiredMask) == desired
  // Initially, we consider all elements of the array, so the above
  // statement is true regardless of input.
  unsigned int desired = 0;
  unsigned int desiredMask = 0;

  // We are looking for the top kToFind-th element when iterating over
  // digits; this count gets reduced by elimination when counting
  // successive digits
  int kToFind = k;

  // We start at the most significant digit in our radix, scanning
  // through to the least significant digit
#pragma unroll
  for (int digitPos = sizeof(BitDataType) * 8 - RADIX_BITS;
       digitPos >= 0;
       digitPos -= RADIX_BITS) {

    // Count radix distribution for the current position and reduce
    // across all threads
    countRadixUsingMask<DataType, BitDataType,
                        IndexType, int, RadixConverter,
                        RADIX_SIZE, RADIX_BITS>(
                          conv, counts, smem,
                          desired, desiredMask, digitPos,
                          sliceSize, withinSliceStride, data);

    // All threads participate in the comparisons below to know the
    // final result

#define CHECK_RADIX(i)                                                  \
    int count = counts[i];                                              \
                                                                        \
    /* All threads have the same value in counts here, so all */        \
    /* threads will return from the function. */                        \
    if (count == 1 && kToFind == 1) {                                   \
      /* There is a unique answer. */                                   \
      desired = setBitfield(desired, i, digitPos, RADIX_BITS);          \
      desiredMask =                                                     \
        setBitfield(desiredMask, RADIX_MASK, digitPos, RADIX_BITS);     \
                                                                        \
      /* The answer is now the unique element v such that: */           \
      /* (v & desiredMask) == desired */                                \
      /* However, we do not yet know what the actual element is. We */  \
      /* need to perform a search through the data to find the */       \
      /* element that matches this pattern. */                          \
      *topK = findPattern<DataType, IndexType, RadixConverter>(         \
        conv, (float*) smem, data, sliceSize,                           \
        withinSliceStride, desired, desiredMask);                       \
      return;                                                           \
    }                                                                   \
                                                                        \
    if (count >= kToFind) {                                             \
      desired = setBitfield(desired, i, digitPos, RADIX_BITS);          \
      desiredMask =                                                     \
        setBitfield(desiredMask, RADIX_MASK, digitPos, RADIX_BITS);     \
                                                                        \
      /* The top-Kth element v must now be one such that: */            \
      /* (v & desiredMask == desired) */                                \
      /* but we haven't narrowed it down; we must check the next */     \
      /* least-significant digit */                                     \
      break;                                                            \
    }                                                                   \
                                                                        \
    kToFind -= count                                                    \

    if (Order) {
      // Process in descending order
#pragma unroll
      for (int i = RADIX_SIZE - 1; i >= 0; --i) {
        CHECK_RADIX(i);
      }
    } else {
      // Process in ascending order
#pragma unroll
      for (int i = 0; i < RADIX_SIZE; ++i) {
        CHECK_RADIX(i);
      }
    }
#undef CHECK_RADIX
  } // end digitPos for

  // There is no unique result, but there is a non-unique result
  // matching `desired` exactly
  *topK = conv.deconvert(desired);
}

template <typename IndexType, int Dim, bool Order>
__global__ void gatherTopK(TensorInfo<float, IndexType> input,
                           IndexType inputSliceSize,
                           IndexType outputSliceSize, // aka `k`

                           IndexType numInputSlices,
                           IndexType inputWithinSliceStride,

                           TensorInfo<float, IndexType> topK,
                           IndexType numTopKSlices,
                           IndexType topKWithinSliceStride,

                           TensorInfo<float, IndexType> indices,
                           IndexType indicesWithinSliceStride) {
  // Indices are limited to integer fp precision, so counts can fit in
  // int32, regardless of IndexType
  __shared__ int smem[32]; // one per each warp, up to warp limit

  IndexType slice = getLinearBlockId<IndexType>();
  if (slice >= numInputSlices) {
    return;
  }

  // Find the start offset for our slice
  IndexType sliceStartIndex =
    IndexToOffset<float, IndexType, Dim>::get(slice, input);
  IndexType topKSliceStartIndex =
    IndexToOffset<float, IndexType, Dim>::get(slice, topK);
  IndexType indicesSliceStartIndex =
    IndexToOffset<float, IndexType, Dim>::get(slice, indices);

  float* inputSliceStart = &input.data[sliceStartIndex];
  float* topKSliceStart = &topK.data[topKSliceStartIndex];
  float* indicesSliceStart = &indices.data[indicesSliceStartIndex];

  // Find the k-th highest element in our input
  float topKValue = -1.0f;
  radixSelect<float, unsigned int, IndexType, FloatToSortedInt, Order>(
    FloatToSortedInt(),
    inputSliceStart, outputSliceSize,
    inputSliceSize, inputWithinSliceStride,
    smem, &topKValue);

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
  IndexType numIterations = THCRoundUp(inputSliceSize, (IndexType) blockDim.x);
  IndexType writeIndexStart = 0;

  for (IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < inputSliceSize);
    float v =
      inRange ? doLdg(&inputSliceStart[i * inputWithinSliceStride]) : 0.0f;
    bool hasTopK;
    if (Order) {
      hasTopK = inRange && (v > topKValue);
    } else {
      hasTopK = inRange && (v < topKValue);
    }

    int index;
    int carry;
    exclusiveBinaryPrefixSum<int, true>(smem, hasTopK, &index, &carry);

    if (hasTopK) {
      int writeIndex = writeIndexStart + index;
      assert(writeIndex < outputSliceSize);

      IndexType topKOffset = writeIndex * topKWithinSliceStride;
      IndexType indexOffset = writeIndex * indicesWithinSliceStride;

      topKSliceStart[topKOffset] = v;
      indicesSliceStart[indexOffset] = i + 1; // to Lua index
    }

    writeIndexStart += carry;
  }

  // We need to fill in the rest with actual == top-K values.
  // The number that we need is outputSliceSize -
  // writeIndexStart. There might be more than that number available,
  // in which case we have to choose the first seen set. We do this
  // via a prefix sum to calculate indices for writing results.
  assert(outputSliceSize >= writeIndexStart);
  IndexType topKRemaining = (outputSliceSize - writeIndexStart);

  for (IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < inputSliceSize);
    float v =
      inRange ? doLdg(&inputSliceStart[i * inputWithinSliceStride]) : 0.0f;
    bool hasTopK = inRange && (v == topKValue);

    int index;
    int carry;
    exclusiveBinaryPrefixSum<int, true>(smem, hasTopK, &index, &carry);

    if (hasTopK && index < topKRemaining) {
      int writeIndex = writeIndexStart + index;
      assert(writeIndex < outputSliceSize);

      IndexType topKOffset = writeIndex * topKWithinSliceStride;
      IndexType indexOffset = writeIndex * indicesWithinSliceStride;

      topKSliceStart[topKOffset] = v;
      indicesSliceStart[indexOffset] = i + 1; // to Lua index
    }

    if (carry >= topKRemaining) {
      break;
    }

    topKRemaining -= carry;
    writeIndexStart += carry;
  }
}

#undef RADIX_BITS
#undef RADIX_SIZE
#undef RADIX_MASK

THC_API void THCudaTensor_topk(THCState* state,
                               THCudaTensor *topK,
                               THCudaTensor *indices,
                               THCudaTensor *input,
                               long k, int dim, int dir, int sorted) {
  THAssert(topK != NULL && indices != NULL && input != NULL);
  THAssert(THCudaTensor_checkGPU(state, 3, topK, indices, input));
  THCCheckTensorDims(state, topK, 2);
  THCCheckTensorDims(state, indices, 2);
  THCCheckTensorDims(state, input, 2);

  int numDims = THCudaTensor_nDimension(state, input);
  THArgCheck(dim >= 0 && dim < numDims, 3, "dim not in range");

  long sliceSize = THCudaTensor_size(state, input, dim);
  THArgCheck(k > 0 && k <= sliceSize, 2, "k not in range for dimension");

  // We're using THCudaTensor to write out indices, so if the slice
  // size that we're selecting has more elements than can be
  // represented in fp32, warn the user
  // FIXME: this isn't a real restriction of either our code or of
  // Thrust, but we have to switch to a CUDA long tensor to support
  // larger slice sizes. Otherwise the indices will contain garbage.
  THArgCheck(sliceSize <= (long) FLOAT32_MAX_CONSECUTIVE_INT, 1,
             "The dimension to be selected exceeds single-precision float "
             "consecutive integer precision size (2^24), since float "
             "is used for indices");

  // Build the output size, which is the dim being selected set to
  // size k
  THLongStorage* topKSize = THCudaTensor_newSizeOf(state, input);
  THLongStorage_set(topKSize, dim, k);
  THCudaTensor_resize(state, topK, topKSize, NULL);
  THCudaTensor_resize(state, indices, topKSize, NULL);
  THLongStorage_free(topKSize);

#define RUN_K(INDEX_T, DIM, DIR)                                        \
  gatherTopK<INDEX_T, DIM, DIR>                                         \
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(             \
      inputInfo,                                                        \
      sliceSize,                                                        \
      k,                                                                \
      inputSlices,                                                      \
      /* The actual dimension that the k-selection is running in */     \
      /* may have changed from collapseDims() */                        \
      inputInfo.strides[collapseInputDim],                              \
      topKInfo,                                                         \
      topKSlices,                                                       \
      topKInfo.strides[collapseTopKDim],                                \
      indicesInfo,                                                      \
      indicesInfo.strides[collapseIndicesDim])

#define RUN_DIR(INDEX_T, DIM)                   \
  if (dir) {                                    \
    RUN_K(INDEX_T, DIM, true);                  \
  } else {                                      \
    RUN_K(INDEX_T, DIM, false);                 \
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
  TensorInfo<float, INDEX_T> inputInfo =                                \
    getTensorInfo<THCudaTensor, INDEX_T>(state, input);                 \
  TensorInfo<float, INDEX_T> topKInfo =                                 \
    getTensorInfo<THCudaTensor, INDEX_T>(state, topK);                  \
  TensorInfo<float, INDEX_T> indicesInfo =                              \
    getTensorInfo<THCudaTensor, INDEX_T>(state, indices);               \
                                                                        \
  /* We use these structures solely to find the offset to */            \
  /* each slice we are operating on */                                  \
  inputInfo.sizes[dim] = 1;                                             \
  topKInfo.sizes[dim] = 1;                                              \
  indicesInfo.sizes[dim] = 1;                                           \
                                                                        \
  /* Collapse all other dims */                                         \
  int collapseInputDim = inputInfo.collapseDims(dim);                   \
  int collapseTopKDim = topKInfo.collapseDims(dim);                     \
  int collapseIndicesDim = indicesInfo.collapseDims(dim);               \
                                                                        \
  long inputSlices = 1;                                                 \
  long topKSlices = 1;                                                  \
  for (int i = 0; i < numDims; ++i) {                                   \
    inputSlices *= inputInfo.sizes[i];                                  \
    topKSlices *= topKInfo.sizes[i];                                    \
  }                                                                     \
                                                                        \
  dim3 grid;                                                            \
  if (!THC_getGridFromTiles(inputSlices, grid)) {                       \
    THError("Slice to sort is too large");                              \
  }                                                                     \
                                                                        \
  dim3 block(std::min(THCRoundUp(sliceSize, 32L), 1024L));              \
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
  RUN_DIM(INDEX_T);

  // Based on required index size, run the algorithm with the
  // appropriate index type
  if (TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, input) &&
      TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, topK) &&
      TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, indices)) {
    RUN_T(unsigned int);
  } else {
    RUN_T(unsigned long);
  }
#undef RUN_T
#undef RUN_DIM
#undef RUN_DIR
#undef RUN_K

  // Sort the results if the user wants them sorted, since our
  // selection routine does not ensure sorting
  if (sorted) {
    // FIXME: the k/v inplace sort along slice only works for size <=
    // 2048 at the moment
    if (sliceSize <= 2048) {
      // This avoids any memory allocations and performs all sorting
      // work inplace along the slice
      THCudaTensor_sortKeyValueInplace(state, topK, indices, dim, dir);
    } else {
      // Depend upon the backup sort that returns indices, which we
      // can use in conjunction with gather to produce the original
      // indices.
      // This is not the most efficient implementation, especially since
      // there are memory allocations performed here. If the user desires
      // greater performance, they should torch.gather() the results
      // themselves using the reported indices, providing previously
      // allocated tensors to receive the results.
      THCudaTensor* sortedTopK = THCudaTensor_new(state);
      THCudaTensor* sortedIndices = THCudaTensor_new(state);
      THCudaTensor_sort(state, sortedTopK, sortedIndices, topK, dim, dir);

      THCudaTensor* sortedTopKIndices = THCudaTensor_new(state);

      THCudaTensor_resizeAs(state, sortedTopKIndices, indices);
      THCudaTensor_gather(state, sortedTopKIndices, indices, dim, sortedIndices);

      THCudaTensor_freeCopyTo(state, sortedTopK, topK);
      THCudaTensor_freeCopyTo(state, sortedTopKIndices, indices);
      THCudaTensor_free(state, sortedIndices);
    }
  }

  THCudaCheck(cudaGetLastError());
}
