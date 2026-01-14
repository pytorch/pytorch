#include <ATen/ceil_div.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/cuda/AsmUtils.cuh>
#include <c10/macros/Macros.h>

namespace at::native {

template <typename scalar_t>
struct TopKTypeConfig {};

template <>
struct TopKTypeConfig<float> {
  typedef uint32_t RadixType;

  // Converts a float to an integer representation with the same
  // sorting; i.e., for floats f1, f2:
  // if f1 < f2 then convert(f1) < convert(f2)
  // We use this to enable radix selection of floating-point values.
  // This also gives a relative order for NaNs, but that's ok, as they
  // will all be adjacent
  // neg inf: signbit=1 exp=ff fraction=0 --> radix = 0 00 ff..
  // pos inf: signbit=0 exp=ff fraction=0 --> radix = 1 ff 00..
  // pos nan: signbit=0 exp=ff fraction>0 --> radix = 1 ff x>0
  // neg nan: signbit=1 exp=ff fraction>0 --> radix = 0 00 x<ff...
  static inline __device__ RadixType convert(float v) {
    RadixType x = __float_as_int(v);
    RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;

    return (v == v) ? (x ^ mask) : 0xffffffff;
  }

  static inline __device__ float deconvert(RadixType v) {
    RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;

    return __int_as_float(v ^ mask);
  }
};

template <>
struct TopKTypeConfig<uint8_t> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType convert(uint8_t v) {
    return v;
  }

  static inline __device__ uint8_t deconvert(RadixType v) {
    return v;
  }
};

template <>
struct TopKTypeConfig<int8_t> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType convert(int8_t v) {
    return 128u + v;
  }

  static inline __device__ int8_t deconvert(RadixType v) {
    return v - 128;
  }
};

template <>
struct TopKTypeConfig<int16_t> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType convert(int16_t v) {
    static_assert(sizeof(short) == 2, "");
    return 32768u + v;
  }

  static inline __device__ int16_t deconvert(RadixType v) {
    return v - 32768;
  }
};

template <>
struct TopKTypeConfig<int32_t> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType convert(int32_t v) {
    static_assert(sizeof(int) == 4, "");
    return 2147483648u + v;
  }

  static inline __device__ int32_t deconvert(RadixType v) {
    return v - 2147483648u;
  }
};

template <>
struct TopKTypeConfig<int64_t> {
  typedef uint64_t RadixType;

  static inline __device__ RadixType convert(int64_t v) {
    static_assert(sizeof(int64_t) == 8, "");
    return 9223372036854775808ull + v;
  }

  static inline __device__ int64_t deconvert(RadixType v) {
    return v - 9223372036854775808ull;
  }
};

template <>
struct TopKTypeConfig<double> {
  typedef uint64_t RadixType;

  static inline __device__ RadixType convert(double v) {
    RadixType x = __double_as_longlong(v);
    RadixType mask = -((x >> 63)) | 0x8000000000000000;
    return (v == v) ? (x ^ mask) : 0xffffffffffffffff;
  }

  static inline __device__ double deconvert(RadixType v) {
    RadixType mask = ((v >> 63) - 1) | 0x8000000000000000;
    return __longlong_as_double(v ^ mask);
  }
};

template <>
struct TopKTypeConfig<at::Half> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType convert(at::Half v) {
    RadixType x = __half_as_ushort(v);
    RadixType mask = (x & 0x00008000) ? 0x0000ffff : 0x00008000;
    return (v == v) ? (x ^ mask) : 0xffff;
  }

  static inline __device__ at::Half deconvert(RadixType v) {
    RadixType mask = (v & 0x00008000) ? 0x00008000 : 0x0000ffff;
    return __ushort_as_half(v ^ mask);
  }
};

template <>
struct TopKTypeConfig<at::BFloat16> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType convert(at::BFloat16 v) {
    RadixType x = v.x;
    RadixType mask = (x & 0x00008000) ? 0x0000ffff : 0x00008000;
    return (v == v) ? (x ^ mask) : 0xffff;
  }

  static inline __device__ at::BFloat16 deconvert(RadixType v) {
    RadixType mask = (v & 0x00008000) ? 0x00008000 : 0x0000ffff;
    at::BFloat16 r;
    r.x = (v ^ mask);
    return r;
  }
};

// Over what radix we are selecting values
constexpr int RADIX_BITS = 2; // digits are base-(2 ^ RADIX_BITS)
constexpr int RADIX_SIZE = 4; // 2 ^ RADIX_BITS
constexpr int RADIX_MASK = (RADIX_SIZE - 1);

#ifndef USE_ROCM
// This function counts the distribution of all input values in a
// slice we are selecting by radix digit at `radixDigitPos`, but only
// those that pass the filter `((v & desiredMask) == desired)`.
// This produces and broadcasts the seen counts for a single block only.
// `smem` must have at least `RadixSize` elements.
template <
    typename scalar_t,
    typename bitwise_t,
    typename index_t,
    typename CountType,
    int RadixSize,
    int RadixBits>
__device__ void countRadixUsingMask(
    CountType counts[RadixSize],
    CountType* smem,
    bitwise_t desired,
    bitwise_t desiredMask,
    int radixDigitPos,
    index_t sliceSize,
    index_t withinSliceStride,
    const scalar_t* data) {
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
  // Must be called outside of loop to ensure all threads participate
  unsigned mask = WARP_BALLOT(threadIdx.x < sliceSize);
  for (index_t i = threadIdx.x; i < sliceSize;) {
    bitwise_t val =
        TopKTypeConfig<scalar_t>::convert(doLdg(&data[i * withinSliceStride]));

    bool hasVal = ((val & desiredMask) == desired);
    bitwise_t digitInRadix = at::cuda::Bitfield<bitwise_t>::getBitfield(
        val, radixDigitPos, RadixBits);

#pragma unroll
    for (uint32_t j = 0; j < RadixSize; ++j) {
      bool vote = hasVal && (digitInRadix == j);
      counts[j] += __popc(WARP_BALLOT(vote, mask));
    }
    i += blockDim.x;
    mask = WARP_BALLOT(i < sliceSize, mask);
  }

  // Now, for each warp, sum values
  if (at::cuda::getLaneId() == 0) {
#pragma unroll
    for (uint32_t i = 0; i < RadixSize; ++i) {
      gpuAtomicAddNoReturn(&smem[i], counts[i]);
    }
  }

  __syncthreads();

  // For each thread, read in the total counts
#pragma unroll
  for (uint32_t i = 0; i < RadixSize; ++i) {
    counts[i] = smem[i];
  }

  __syncthreads();
}

// This finds the unique value `v` that matches the pattern
// ((v & desired) == desiredMask) in our sorted int format
template <typename scalar_t, typename bitwise_t, typename index_t>
__device__ scalar_t findPattern(
    scalar_t* smem,
    const scalar_t* data,
    index_t sliceSize,
    index_t withinSliceStride,
    bitwise_t desired,
    bitwise_t desiredMask) {
  if (threadIdx.x < 2) {
    smem[threadIdx.x] = static_cast<scalar_t>(0);
  }
  __syncthreads();

  // All threads participate in the loop, in order to sync on the flag
  index_t numIterations =
      round_up(sliceSize, static_cast<index_t>(blockDim.x));
  for (index_t i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < sliceSize);
    scalar_t v = inRange ? doLdg(&data[i * withinSliceStride])
                         : static_cast<scalar_t>(0);

    if (inRange &&
        ((TopKTypeConfig<scalar_t>::convert(v) & desiredMask) == desired)) {
      // There should not be conflicts if we are using findPattern,
      // since the result is unique
      smem[0] = static_cast<scalar_t>(1);
      smem[1] = v; // can't use val as the flag, since it could be 0
    }

    __syncthreads();

    scalar_t found = smem[0];
    scalar_t val = smem[1];

    __syncthreads();

    // Check to see if a thread found the value
    if (found != static_cast<scalar_t>(0)) {
      // all threads return this value
      return val;
    }
  }

  // should not get here
  CUDA_KERNEL_ASSERT(false);
  return static_cast<scalar_t>(0);
}


#else

// this is the main loop of the countRadixUsingMask function that counts the distribution
// of the bits in the radix digit at `radixDigitPos` to `radixDigitPos`+RADIX_BITS-1. This works for generic data accessors
// so that it works when reading from global memory or shared memory.
template <typename scalar_t, typename bitwise_t, typename index_t, typename CountType, int RadixSize, int RadixBits, typename DataAccessor>
__device__ __forceinline__ void countRadixLoop(
    CountType counts[RadixSize], // counts[i] will be the number of matching elements ((val & desiredMask) == desired) and that have the digits [radixDigitPos, radixDigitPos+RADIX_BITS-1] set to i.
    bitwise_t desired, // combined with desiredMask to filter relevant elements. An element is relevant if ((val & desiredMask) == desired).
    bitwise_t desiredMask, // combined with desired to filter relevantelements. An element is relevant if ((val & desiredMask) == desired).
    int radixDigitPos, // the position of the radix digit.
    index_t loopBound, // the upper bound of the loop.
    DataAccessor&& getData){ // a function that returns the input data value at index i.

  scalar_t v      = threadIdx.x < loopBound ? getData(threadIdx.x) : static_cast<scalar_t>(0);
  for (index_t i = threadIdx.x; i < round_up(static_cast<index_t>(loopBound), static_cast<index_t>(warpSize)); i += blockDim.x) {
    scalar_t v_next = i + blockDim.x < loopBound ? getData(i + blockDim.x) : static_cast<scalar_t>(0); // prefetch the next value.

    bool hasVal = false;
    bitwise_t digitInRadix = static_cast<bitwise_t>(0);
    if (i < loopBound) {
      bitwise_t val = TopKTypeConfig<scalar_t>::convert(v);
      hasVal = ((val & desiredMask) == desired);
      digitInRadix = at::cuda::Bitfield<bitwise_t>::getBitfield(val, radixDigitPos, RadixBits);
    }
    #pragma unroll
    for (uint32_t j = 0; j < RadixSize; ++j) {
      bool vote = hasVal && (digitInRadix == j); // true if: (value is relevant) && (the digits [radixDigitPos, radixDigitPos+RADIX_BITS-1] set to j).
      counts[j] += __popcll(WARP_BALLOT(vote)); // how many threads in this warp found digitInRadix == j while matching the desired pattern?
    }

    v = v_next;
  }
}


// This function counts the distribution of all input values in a
// slice we are selecting by radix digit at `radixDigitPos`, but only
// those that pass the filter `((v & desiredMask) == desired)`.
// This produces and broadcasts the seen counts for a single block only.
// `smem` must have at least `RadixSize` elements.
// this is an smem-friendly version of the countRadixUsingMask function.
// it works when data is in global memory or in shared memory.
template <
    typename scalar_t,
    typename bitwise_t,
    typename index_t,
    typename CountType,
    int RadixSize,
    int RadixBits>
__device__ void countRadixUsingMaskDataSmem(
    CountType counts[RadixSize], // counts[i] will be the number of matching elements ((val & desiredMask) == desired) that have the digits [radixDigitPos, radixDigitPos+RADIX_BITS-1] set to i.
    CountType* smem, // shared memory for inter-warp reduction of counts.
    bitwise_t desired, // combined with desiredMask to filter relevant elements. An element is relevant if ((val & desiredMask) == desired).
    bitwise_t desiredMask, // combined with desired to filter relevant elements. An element is relevant if ((val & desiredMask) == desired).
    int radixDigitPos, // position of the radix digit.
    index_t sliceSize, // size of the input slice.
    index_t withinSliceStride, // stride of the input slice.
    const scalar_t* data, // input data.
    const scalar_t* dataSmem, // input data stored in shared memory.
    int dataSmemSize) { // input data size stored in shared memory.
    // Clear out per-thread counts from a previous round
#pragma unroll
  for (int i = 0; i < RadixSize; ++i) {
    counts[i] = 0; // initialize counts to 0.
  }

  if (threadIdx.x < RadixSize) {
    smem[threadIdx.x] = 0; // initialize smem to 0.
  }
  __syncthreads();

  // Scan over all the data. Upon a read, the warp will accumulate
  // counts per each digit in the radix using warp voting.

  if (dataSmemSize > 0) { // if data is in shared memory, use dataSmem as the input data.
    countRadixLoop<scalar_t, bitwise_t, index_t, int, RadixSize, RadixBits>(
      counts, desired, desiredMask, radixDigitPos, dataSmemSize, [&](index_t i) -> scalar_t { return dataSmem[i]; });
  } else { // if data is in global memory, use data as the input data.
    countRadixLoop<scalar_t, bitwise_t, index_t, int, RadixSize, RadixBits>(
      counts, desired, desiredMask, radixDigitPos, sliceSize, [&](index_t i) -> scalar_t { return doLdg(&data[i * withinSliceStride]); });
  }

    // Now, for each warp, sum values
  if (at::cuda::getLaneId() == 0) {
  #pragma unroll
      for (uint32_t i = 0; i < RadixSize; ++i) {
        gpuAtomicAddNoReturn(&smem[i], counts[i]); // thread0 in warp atomically adds the counts to smem.
      }
    }

    __syncthreads(); // wait for all threads in the block to finish counting.

    // For each thread, read in the total counts
  #pragma unroll
    for (uint32_t i = 0; i < RadixSize; ++i) {
      counts[i] = smem[i];
    }

    __syncthreads(); // wait for all threads in the block to finish reading the counts.
}

// This is the main loop of the findPattern function that finds the unique value `v` that matches the pattern ((v & desired) == desiredMask)
// in the input data. This works for generic data accessors so that it works when reading from global memory or shared memory.
template <typename scalar_t, typename bitwise_t, typename index_t, typename DataAccessor>
__device__ __forceinline__ scalar_t findPatternLoop(
    scalar_t* smem, // shared memory for inter-thread communication of the found value.
    bitwise_t desired, // combined with desiredMask to filter relevant elements. An element is relevant if ((val & desiredMask) == desired).
    bitwise_t desiredMask, // combined with desired to filter relevant elements. An element is relevant if ((val & desiredMask) == desired).
    index_t loopBound, // the upper bound of the loop.
    DataAccessor&& getData){ // a function that returns the input data value at index i.
  

  for (index_t i = threadIdx.x; i < round_up(loopBound, static_cast<index_t>(blockDim.x)); i += blockDim.x) {
    bool inRange = (i < loopBound);
    scalar_t v = inRange ? getData(i) : static_cast<scalar_t>(0);

    if (inRange &&
        ((TopKTypeConfig<scalar_t>::convert(v) & desiredMask) == desired)) {
      // There should not be conflicts if we are using findPattern,
      // since the result is unique
      smem[0] = static_cast<scalar_t>(1);
      smem[1] = v; // can't use val as the flag, since it could be 0
    }

    __syncthreads();

    scalar_t found = smem[0];
    scalar_t val = smem[1];

    __syncthreads();

    // Check to see if a thread found the value
    if (found != static_cast<scalar_t>(0)) {
      // all threads return this value
      return val;
    }
  }

    // should not get here
  CUDA_KERNEL_ASSERT(false);
  return static_cast<scalar_t>(0);
}

// This finds the unique value `v` that matches the pattern ((v & desired) == desiredMask)
// in the input data. This is an smem-friendly version of the findPattern function.
// This works for generic data accessors so that it works when reading from global memory or shared memory.
template <typename scalar_t, typename bitwise_t, typename index_t>
__device__ scalar_t findPatternDataSmem(
    scalar_t* smem, // shared memory for inter-thread communication of the found value.
    const scalar_t* data, // input data.
    index_t sliceSize, // size of the input slice.
    index_t withinSliceStride, // stride of the input slice.
    bitwise_t desired, // combined with desiredMask to filter relevant elements. An element is relevant if ((val & desiredMask) == desired).
    bitwise_t desiredMask, // combined with desired to filter relevant elements. An element is relevant if ((val & desiredMask) == desired).
    const scalar_t* dataSmem, // input data stored in shared memory.
    index_t dataSmemSize) { // input data size stored in shared memory.
  if (threadIdx.x < 2) {
    smem[threadIdx.x] = static_cast<scalar_t>(0); // initialize smem to 0.
  }
  __syncthreads();

  if (dataSmemSize > 0) { // if data is in shared memory, use dataSmem as the input data.
    return findPatternLoop<scalar_t, bitwise_t, index_t>(
      smem, desired, desiredMask, dataSmemSize, [&](index_t i) -> scalar_t { return dataSmem[i]; });
  } else { // if data is in global memory, use data as the input data.
    return findPatternLoop<scalar_t, bitwise_t, index_t>(
      smem, desired, desiredMask, sliceSize, [&](index_t i) -> scalar_t { return doLdg(&data[i * withinSliceStride]); });
  }

  return static_cast<scalar_t>(0);
}


// This function fills the shared memory with the input data.
// this function is called before the main loop of the radixSelect function.
// if dataSmemSize > 0, the shared memory is already filled, so we return.
// otherwise we check if the input data is small enough to fit into shared memory.
// if it is, all the data is put into shared memory. If not, but dataSizeRemaining <= dataSmemCap,
// it means that the data has been filtered (through having (val & desiredMask) == desired) so much 
// that it fits into the shared memory. So we put the filtered data into shared memory.
template <typename scalar_t, typename bitwise_t, typename index_t>
__device__ __forceinline__ void fillDataSmem(
    scalar_t* dataSmem, // shared memory to store the input data.
    index_t dataSmemCap, // max number of elements that can be stored in dataSmem.
    index_t dataSizeRemaining, // number of relevant elements remaining. We put data on dataSmem once dataSizeRemaining <= dataSmemCap.
    index_t& dataSmemSize, // actual number of elements in dataSmem.
    index_t sliceSize, // size of the input slice.
    index_t withinSliceStride, // stride of the input slice.
    const scalar_t* data, // input data.
    bitwise_t desired, // combined with desiredMask to filter relevant elements. An element is relevant if ((val & desiredMask) == desired).
    bitwise_t desiredMask, // combined with desired to filter relevant elements. An element is relevant if ((val & desiredMask) == desired).
    int& DataSmemWriteIndex // index used to write data to dataSmem. Incremented atomically.
    ) {
  
  if (dataSmemSize > 0) return; // already filled
  
  if (sliceSize <= dataSmemCap){ // if the input data is small enough, put all of it into shared memory.
    
    // reading from global memory. Prefetching to improve performance.
    scalar_t v = static_cast<scalar_t>(0);
    if (threadIdx.x < sliceSize) v = doLdg(&data[threadIdx.x * withinSliceStride]);
    for (index_t i = threadIdx.x; i < sliceSize; i += blockDim.x) {
      scalar_t v_next = (i+blockDim.x)<sliceSize ? doLdg(&data[(i + blockDim.x) * withinSliceStride]) : static_cast<scalar_t>(0);
      dataSmem[i] = v;
      v = v_next;
    }

    __syncthreads(); // wait for all threads in the block to finish writing to dataSmem.
    if (threadIdx.x == 0){
      dataSmemSize = sliceSize; // thread 0 updates dataSmemSize to the size of the input slice.
    }
    __syncthreads(); // wait for all threads in the block to see the updated dataSmemSize.

  } else if (dataSizeRemaining <= dataSmemCap){ // if data did not fit originally, but now it does.
    // if this is the case, data needs to be filtered so only the relevant data is stored in dataSmem.
    // Each warp performs an internal counting of the number of elements that match the desired pattern.
    // Then reserves slots in dataSmem for the matching elements by atomically incrementing DataSmemWriteIndex.
    // Finally, each thread within the warp writes its value to the appropriate slot in dataSmem.
    // This is done to minimize the amount of time each warp spends waiting for others.
    
    constexpr index_t MAX_WARPS = 1024/32; // max number of warps in a block.
    __shared__ index_t warp_bases[MAX_WARPS]; // to store the base index in dataSmem for each warp.
    
    int warp_bits = __builtin_ctz(warpSize); // = log2(WARP_SIZE).
    
    int warp_id = threadIdx.x >> warp_bits; // = threadIdx.x / WARP_SIZE
    int lane_id = at::cuda::getLaneId(); // = threadIdx.x % WARP_SIZE
  
    for (index_t i = threadIdx.x; i < round_up(static_cast<index_t>(sliceSize), static_cast<index_t>(warpSize)); i += blockDim.x) {        
      scalar_t v = static_cast<scalar_t>(0);
      bool match = false;
      if (i < sliceSize) {
        v = doLdg(&data[i * withinSliceStride]);
        match = ((TopKTypeConfig<scalar_t>::convert(v) & desiredMask) == desired);
      }

      // Warp-level ballot
      uint64_t ballot = WARP_BALLOT(match); // what threads in this warp match the desired pattern?
      int warp_count = __popcll(ballot); // how many threads in this warp match the desired pattern?
  
      if (lane_id == 0 && warp_count > 0) {
        warp_bases[warp_id] = atomicAdd(&DataSmemWriteIndex, warp_count); // reserve warp_count slots in dataSmem for this warp, and get the base index.
      }

      if (match) { // if the current thread matches the desired pattern, store the value in dataSmem.
        uint64_t my_mask = (1ULL << lane_id) - 1; // a bitmask: [0, 0, 0, ..., 0, 1, 1, 1, ..., 1] with (64-lane_id) 0s and lane_id 1s.
        int my_offset = __popcll(ballot & my_mask); // count the number of threads that have match to the right of the current thread in bitmask.
        dataSmem[warp_bases[warp_id] + my_offset] = v;
      }

    }
    __syncthreads();
    if (threadIdx.x == 0) {
      dataSmemSize = DataSmemWriteIndex;
    }
    __syncthreads();
  }
}

#endif

// Returns the top-Kth element found in the data using radix selection
template <typename scalar_t, typename bitwise_t, typename index_t>
__device__ void radixSelect(
    const scalar_t* data,
    index_t k,
    bool largest,
    index_t sliceSize,
    index_t withinSliceStride,
    int* smem,
    scalar_t* topK) {
  // Per-thread buckets into which we accumulate digit counts in our
  // radix
  int counts[RADIX_SIZE];
#ifdef USE_ROCM
  // this kernel reads all the data at most (sizeof(scalar_t)*2/RADIX_BITS + 1) times.
  // if data fits into shared memory, we can avoid reading data from global memory.
  // if not, we may still be able to put the filtered data, after a few iterations,
  // into shared memory. after every pass, relevant data is likely reduced by a factor of RADIX_SIZE.
  // dataSmem is used to store the relavant data.
  constexpr index_t DATA_SMEM_BYTES = 3 * 1024; // 3KB is a good compromise between memory usage and performance.
  constexpr index_t dataSmemCap = DATA_SMEM_BYTES / sizeof(scalar_t); // max number of elements that can be stored in dataSmem.
  __shared__ scalar_t dataSmem[dataSmemCap]; 
  __shared__ index_t dataSmemSize;  // actual number of elements in dataSmem.
  __shared__ index_t dataSizeRemaining; // number of relevant elements remaining. We put data on dataSmem once dataSizeRemaining <= dataSmemCap.
  __shared__ int DataSmemWriteIndex; // index used to write data to dataSmem.
  if (threadIdx.x == 0) {
    dataSmemSize = 0;
    DataSmemWriteIndex = 0;
    dataSizeRemaining = sliceSize;
  }
  __syncthreads();
#endif

  // We only consider elements x such that (x & desiredMask) == desired
  // Initially, we consider all elements of the array, so the above
  // statement is true regardless of input.
  bitwise_t desired = 0;
  bitwise_t desiredMask = 0;

  // We are looking for the top kToFind-th element when iterating over
  // digits; this count gets reduced by elimination when counting
  // successive digits
  int kToFind = k;

  // We start at the most significant digit in our radix, scanning
  // through to the least significant digit
  for (int digitPos = sizeof(scalar_t) * 8 - RADIX_BITS; digitPos >= 0;
       digitPos -= RADIX_BITS) {
    // Count radix distribution for the current position and reduce
    // across all threads
#ifdef USE_ROCM
    fillDataSmem<scalar_t, bitwise_t, index_t>(
      dataSmem,
      dataSmemCap,
      dataSizeRemaining,
      dataSmemSize,
      sliceSize,
      withinSliceStride,
      data,
      desired,
      desiredMask,
      DataSmemWriteIndex);

    countRadixUsingMaskDataSmem<
        scalar_t,
        bitwise_t,
        index_t,
        int,
        RADIX_SIZE,
        RADIX_BITS>(
        counts,
        smem,
        desired,
        desiredMask,
        digitPos,
        sliceSize,
        withinSliceStride,
        data,
        dataSmem,
        dataSmemSize);
#else
    countRadixUsingMask<
        scalar_t,
        bitwise_t,
        index_t,
        int,
        RADIX_SIZE,
        RADIX_BITS>(
        counts,
        smem,
        desired,
        desiredMask,
        digitPos,
        sliceSize,
        withinSliceStride,
        data);

#endif
    auto found_unique = [&](int i, int count) -> bool {
      /* All threads have the same value in counts here, so all */
      /* threads will return from the function. */
      if (count == 1 && kToFind == 1) {
        /* There is a unique answer. */
        desired = at::cuda::Bitfield<bitwise_t>::setBitfield(
            desired, i, digitPos, RADIX_BITS);
        desiredMask = at::cuda::Bitfield<bitwise_t>::setBitfield(
            desiredMask, RADIX_MASK, digitPos, RADIX_BITS);

        /* The answer is now the unique element v such that: */
        /* (v & desiredMask) == desired */
        /* However, we do not yet know what the actual element is. We */
        /* need to perform a search through the data to find the */
        /* element that matches this pattern. */
#ifndef USE_ROCM
        *topK = findPattern<scalar_t, bitwise_t, index_t>(
            (scalar_t*)smem,
            data,
            sliceSize,
            withinSliceStride,
            desired,
            desiredMask);
#else
        *topK = findPatternDataSmem<scalar_t, bitwise_t, index_t>(
            (scalar_t*)smem,
            data,
            sliceSize,
            withinSliceStride,
            desired,
            desiredMask,
            dataSmem,
            dataSmemSize);
#endif
        return true;
      }
      return false;
    };
    auto found_non_unique = [&](int i, int count) -> bool {
      if (count >= kToFind) {
        desired =
            at::cuda::Bitfield<bitwise_t>::setBitfield(
                desired, i, digitPos, RADIX_BITS);
        desiredMask = at::cuda::Bitfield<bitwise_t>::setBitfield(
            desiredMask, RADIX_MASK, digitPos, RADIX_BITS);

#ifdef USE_ROCM
            if (dataSmemSize == 0){ // we only care about updating dataSizeRemaining when dataSmem is empty.
              if (threadIdx.x == 0) {
                  // this bucket has count >= kToFind elements. This means topK is in this bucket and
                  // the number of elements with value & desiredMask == desired (which is the relevant data) equals count.
                  // so we update dataSizeRemaining to count.
                  dataSizeRemaining = count;
              }
              __syncthreads();
            }
#endif
        /* The top-Kth element v must now be one such that: */
        /* (v & desiredMask == desired) */
        /* but we haven't narrowed it down; we must check the next */
        /* least-significant digit */
        return true;
      }
      kToFind -= count;
      return false; // continue the loop
    };

    // All threads participate in the comparisons below to know the
    // final result
    if (largest) {
      // Process in descending order
#pragma unroll
      for (int i = RADIX_SIZE - 1; i >= 0; --i) {
        int count = counts[i];
        if (found_unique(i, count)) {
          return;
        }
        if (found_non_unique(i, count)) {
          break;
        }
      }
    } else {
      // Process in ascending order
#pragma unroll
      for (int i = 0; i < RADIX_SIZE; ++i) {
        int count = counts[i];
        if (found_unique(i, count)) {
          return;
        }
        if (found_non_unique(i, count)) {
          break;
        }
      }
    }
  } // end digitPos for

  // There is no unique result, but there is a non-unique result
  // matching `desired` exactly
  *topK = TopKTypeConfig<scalar_t>::deconvert(desired);
}
} // namespace at::native
