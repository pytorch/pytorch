#ifndef CAFFE2_OPERATORS_TOP_K_RADIX_SELECTION_H_
#define CAFFE2_OPERATORS_TOP_K_RADIX_SELECTION_H_

#include "caffe2/core/common_gpu.h"
#include "caffe2/utils/GpuDefs.cuh"
#include "caffe2/utils/GpuScanUtils.cuh"
#include "caffe2/utils/math.h"
#include <cuda_runtime.h>

namespace caffe2 {

// From the cutorch library

template <typename T>
struct AddOp {
  __device__ __forceinline__ T operator()(T &lhs, T &rhs) {
    return lhs + rhs;
  }
};

template <typename T>
struct TopKTypeConfig {};

template <>
struct TopKTypeConfig<float> {
  typedef unsigned int RadixType;

  // Converts a float to an integer representation with the same
  // sorting; i.e., for floats f1, f2:
  // if f1 < f2 then convert(f1) < convert(f2)
  // We use this to enable radix selection of floating-point values.
  // This also gives a relative order for NaNs, but that's ok, as they
  // will all be adjacent
  static inline __device__ RadixType convert(float v) {
    RadixType x = __float_as_int(v);
    RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;

    return (x ^ mask);
  }

  static inline __device__ float deconvert(RadixType v) {
    RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;

    return __int_as_float(v ^ mask);
  }
};

template <>
struct TopKTypeConfig<unsigned char> {
  typedef unsigned int RadixType;

  static inline __device__ RadixType convert(unsigned char v) {
    return v;
  }

  static inline __device__ unsigned char deconvert(RadixType v) {
    return v;
  }
};

template <>
struct TopKTypeConfig<char> {
  typedef unsigned int RadixType;

  static inline __device__ RadixType convert(char v) {
    return 128u + v;
  }

  static inline __device__ char deconvert(RadixType v) {
    return v - 128;
  }
};

template <>
struct TopKTypeConfig<short> {
  typedef unsigned int RadixType;

  static inline __device__ RadixType convert(short v) {
    static_assert(sizeof(short) == 2, "");
    return 32768u + v;
  }

  static inline __device__ short deconvert(RadixType v) {
    return v - 32768;
  }
};

template <>
struct TopKTypeConfig<int> {
  typedef unsigned int RadixType;

  static inline __device__ RadixType convert(int v) {
    static_assert(sizeof(int) == 4, "");
    return 2147483648u + v;
  }

  static inline __device__ int deconvert(RadixType v) {
    return v - 2147483648u;
  }
};

template <>
struct TopKTypeConfig<int64_t> {
  typedef unsigned long long int RadixType;

  static inline __device__ RadixType convert(int64_t v) {
    //static_assert fails on windows, so leave it as CUDA_KERNEL_ASSERT
    static_assert(sizeof(int64_t) == 8, "");
    return 9223372036854775808ull + v;
  }

  static inline __device__ int64_t deconvert(RadixType v) {
    return v - 9223372036854775808ull;
  }
};

template <>
struct TopKTypeConfig<double> {
  typedef unsigned long long int RadixType;

  static inline __device__ RadixType convert(double v) {
    RadixType x = __double_as_longlong(v);
    RadixType mask = -((x >> 63)) | 0x8000000000000000;
    return (x ^ mask);
  }

  static inline __device__ double deconvert(RadixType v) {
    RadixType mask = ((v >> 63) - 1) | 0x8000000000000000;
    return __longlong_as_double(v ^ mask);
  }
};

// This function counts the distribution of all input values in a
// slice we are selecting by radix digit at `radixDigitPos`, but only
// those that pass the filter `((v & desiredMask) == desired)`.
// This produces and broadcasts the seen counts for a single block only.
// `smem` must have at least `RadixSize` elements.
template <typename DataType,
          typename BitDataType,
          typename CountType,
          int RadixSize,
          int RadixBits>
__device__ void countRadixUsingMask(CountType counts[RadixSize],
                                    CountType* smem,
                                    BitDataType desired,
                                    BitDataType desiredMask,
                                    int radixDigitPos,
                                    int sliceSize,
                                    const DataType* data) {
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
  for (int i = threadIdx.x; i < sliceSize; i += blockDim.x) {
    BitDataType val = TopKTypeConfig<DataType>::convert(data[i]);

    bool hasVal = ((val & desiredMask) == desired);
    BitDataType digitInRadix = Bitfield<BitDataType>::getBitfield(val, radixDigitPos, RadixBits);

#pragma unroll
    for (unsigned int j = 0; j < RadixSize; ++j) {
      bool vote = hasVal && (digitInRadix == j);
#if defined(__HIP_PLATFORM_HCC__)
      counts[j] += __popcll(__ballot(vote));
#else
      counts[j] += __popc(__ballot_sync(__activemask(), vote));
#endif  // __HIP_PLATFORM_HCC__
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
template <typename DataType, typename BitDataType>
__device__ DataType findPattern(DataType* smem,
                                const DataType* data,
                                int sliceSize,
                                BitDataType desired,
                                BitDataType desiredMask) {
  if (threadIdx.x < kWarpSize) {
    smem[threadIdx.x] = (DataType) 0;
  }
  __syncthreads();

  // All threads participate in the loop, in order to sync on the flag
  int numIterations = math::RoundUp(sliceSize, (int) blockDim.x);
  for (int i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < sliceSize);
    DataType v = inRange ? data[i] : (DataType)0;

    if (inRange && ((TopKTypeConfig<DataType>::convert(v) & desiredMask) == desired)) {
      // There should not be conflicts if we are using findPattern,
      // since the result is unique
      smem[0] = (DataType)1;
      smem[1] = v; // can't use val as the flag, since it could be 0
    }

    __syncthreads();

    DataType found = smem[0];
    DataType val = smem[1];

    __syncthreads();

    // Check to see if a thread found the value
    if (found != (DataType)0) {
      // all threads return this value
      return val;
    }
  }

  // should not get here
  CUDA_KERNEL_ASSERT(false);
  return (DataType)0;
}

// Returns the top-Kth element found in the data using radix selection
template <typename DataType, typename BitDataType, bool Order>
__device__ void radixSelect(const DataType* data,
                            int k,
                            int sliceSize,
                            int* smem,
                            DataType* topK) {
  // Per-thread buckets into which we accumulate digit counts in our
  // radix
  int counts[RADIX_SIZE];

  // We only consider elements x such that (x & desiredMask) == desired
  // Initially, we consider all elements of the array, so the above
  // statement is true regardless of input.
  BitDataType desired = 0;
  BitDataType desiredMask = 0;

  // We are looking for the top kToFind-th element when iterating over
  // digits; this count gets reduced by elimination when counting
  // successive digits
  int kToFind = k <= sliceSize ? k : sliceSize;

  // We start at the most significant digit in our radix, scanning
  // through to the least significant digit
#pragma unroll
  for (int digitPos = sizeof(DataType) * 8 - RADIX_BITS;
       digitPos >= 0;
       digitPos -= RADIX_BITS) {

    // Count radix distribution for the current position and reduce
    // across all threads
    countRadixUsingMask<DataType, BitDataType,
                        int,
                        RADIX_SIZE, RADIX_BITS>(
                          counts, smem,
                          desired, desiredMask, digitPos,
                          sliceSize, data);

    // All threads participate in the comparisons below to know the
    // final result


#define CHECK_RADIX(i)                                                  \
    int count = counts[i];                                              \
                                                                        \
    /* All threads have the same value in counts here, so all */        \
    /* threads will return from the function. */                        \
    if (count == 1 && kToFind == 1) {                                   \
      /* There is a unique answer. */                                   \
      desired = Bitfield<BitDataType>::setBitfield(desired, i, digitPos, RADIX_BITS);          \
      desiredMask =                                                     \
        Bitfield<BitDataType>::setBitfield(desiredMask, RADIX_MASK, digitPos, RADIX_BITS);     \
                                                                        \
      /* The answer is now the unique element v such that: */           \
      /* (v & desiredMask) == desired */                                \
      /* However, we do not yet know what the actual element is. We */  \
      /* need to perform a search through the data to find the */       \
      /* element that matches this pattern. */                          \
      *topK = findPattern<DataType, BitDataType>(                       \
        (DataType*) smem, data, sliceSize,                              \
        desired, desiredMask);                                          \
      return;                                                           \
    }                                                                   \
                                                                        \
    if (count >= kToFind) {                                             \
      desired = Bitfield<BitDataType>::setBitfield(desired, i, digitPos, RADIX_BITS);          \
      desiredMask =                                                     \
        Bitfield<BitDataType>::setBitfield(desiredMask, RADIX_MASK, digitPos, RADIX_BITS);     \
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
  *topK = TopKTypeConfig<DataType>::deconvert(desired);
}

template <typename T, bool Order, typename IndicesType>
__global__ void gatherTopK(const T* inputPtr,
                           int inputSliceSize,
                           int outputSliceSize, // aka `k`
                           int numInputSlices,
                           T* topKPtr,
                           IndicesType* indicesPtr) {
  __shared__ int smem[kWarpSize]; // one per each warp, up to warp limit

  int slice = blockIdx.x;
  if (slice >= numInputSlices) {
    return;
  }

  // Find the start offset for our slice
  const T* inputSliceStart = &inputPtr[slice * inputSliceSize];
  T* topKSliceStart = &topKPtr[slice * outputSliceSize];
  IndicesType* indicesSliceStart = &indicesPtr[slice * outputSliceSize];

  // Find the k-th highest element in our input
  T topKValue = (T)0;
  radixSelect<T, typename TopKTypeConfig<T>::RadixType, Order>(
    inputSliceStart, outputSliceSize,
    inputSliceSize,
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
  int numIterations = math::RoundUp(inputSliceSize, (int) blockDim.x);
  int writeIndexStart = 0;

  for (int i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < inputSliceSize);
    T v = inRange ? inputSliceStart[i] : (T)0;
    bool hasTopK;
    if (Order) {
      hasTopK = inRange && (v > topKValue);
    } else {
      hasTopK = inRange && (v < topKValue);
    }

    int index;
    int carry;
    exclusiveBinaryPrefixScan<int, true>(smem, hasTopK, &index, &carry, AddOp<int>());

    if (hasTopK) {
      int writeIndex = writeIndexStart + index;
      CUDA_KERNEL_ASSERT(writeIndex < outputSliceSize);

      int topKOffset = writeIndex;
      int indexOffset = writeIndex;

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
  int topKRemaining = (outputSliceSize - writeIndexStart);

  for (int i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < inputSliceSize);
    T v = inRange ? inputSliceStart[i] : (T)0;
    bool hasTopK = inRange && (v == topKValue);

    int index;
    int carry;
    exclusiveBinaryPrefixScan<int, true>(smem, hasTopK, &index, &carry, AddOp<int>());

    if (hasTopK && index < topKRemaining) {
      int writeIndex = writeIndexStart + index;
      CUDA_KERNEL_ASSERT(writeIndex < outputSliceSize);

      int topKOffset = writeIndex;
      int indexOffset = writeIndex;

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

#undef RADIX_BITS
#undef RADIX_SIZE
#undef RADIX_MASK

} // namespace caffe2

#endif // CAFFE2_OPERATORS_TOP_K_RADIX_SELECTION_H_
