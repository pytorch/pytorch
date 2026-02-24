#include <ATen/ceil_div.h>
#include <c10/macros/Macros.h>
#include <ATen/cuda/AsmUtils.cuh>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/DeviceUtils.cuh>

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
  index_t numIterations = round_up(sliceSize, static_cast<index_t>(blockDim.x));
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

/*
This implementation of radixSelect optimizes the k-th element selection
algorithm by dynamically utilizing shared memory to cache input data when
possible, significantly reducing global memory traffic during the iterative bit
discovery process. The radixSelect algorithm finds the k-th element by
iteratively uncovering its bit pattern through multiple passes over the data.
Each pass determines 2 bits of the target value's bitmap (up to 16 passes for
float32 inputs). As iterations progress, the number of relevant values decreases
by approximately 4× per pass, assuming uniform bit distribution. While initially
the input data may be too large to fit in shared memory, it often becomes
cacheable after a few filtering iterations as the data size shrinks. This
implementation introduces dynamic shared memory caching that checks at each
iteration whether the filtered data fits within available LDS (a few KB's
allocated for this purpose). When the data fits, it is cached to shared memory,
eliminating redundant global memory reads in subsequent operations within that
iteration. New kernel functions countRadixUsingMaskDataSmem and findPatternSmem
were introduced to seamlessly handle both cached (LDS) and non-cached (global
memory) data paths. These variants maintain backward compatibility with the
original algorithm and automatically fall back to global memory access when data
exceeds LDS capacity.
*/

// this is the main loop of the countRadixUsingMask function that counts the
// distribution of the bits in the radix digit at `radixDigitPos` to
// `radixDigitPos`+RADIX_BITS-1. DataAccessor is a function that returns the
// input data value at index i. It could potentially be a global memory accessor
// or a shared memory accessor.
template <
    typename scalar_t,
    typename bitwise_t,
    typename index_t,
    typename CountType,
    int RadixSize,
    int RadixBits,
    bool prefetch,
    typename DataAccessor>
__device__ __forceinline__ void countRadixLoop(
    CountType counts[RadixSize], // counts[i] will be the number of matching
                                 // elements ((val & desiredMask) == desired)
                                 // that have the digits [radixDigitPos,
                                 // radixDigitPos+RADIX_BITS-1] set to i.
    bitwise_t
        desired, // combined with desiredMask to filter relevant elements. A
                 // value is relevant if ((val & desiredMask) == desired).
    bitwise_t
        desiredMask, // combined with desired to filter relevant elements. A
                     // value is relevant if ((val & desiredMask) == desired).
    int radixDigitPos, // the position of the radix digit.
    index_t loopBound, // the upper bound of the loop.
    DataAccessor&& getData) { // a function that returns the input data value at
                              // index i. It could potentially be a global
                              // memory accessor or a shared memory accessor.

  // the kernel consists of two parts:
  // phase 1: processing 4 elements at an iteration.
  // phase 2: processing 1 element at an iteration.

  constexpr index_t unroll_factor = 4;
  index_t unroll_segment =
      (loopBound / (blockDim.x * unroll_factor)) * blockDim.x * unroll_factor;

  // phase 1: processing 4 elements at an iteration.

  for (index_t i = threadIdx.x * unroll_factor; i < unroll_segment;
       i += blockDim.x * unroll_factor) {

    // prefetch 4 elements.
    scalar_t v0 = getData(i);
    scalar_t v1 = getData(i + 1);
    scalar_t v2 = getData(i + 2);
    scalar_t v3 = getData(i + 3);

    // convert the values to bitwise_t.
    bitwise_t val0 = TopKTypeConfig<scalar_t>::convert(v0);
    bitwise_t val1 = TopKTypeConfig<scalar_t>::convert(v1);
    bitwise_t val2 = TopKTypeConfig<scalar_t>::convert(v2);
    bitwise_t val3 = TopKTypeConfig<scalar_t>::convert(v3);

    // check if the values match the desired pattern.
    bool hasVal0 = ((val0 & desiredMask) == desired);
    bool hasVal1 = ((val1 & desiredMask) == desired);
    bool hasVal2 = ((val2 & desiredMask) == desired);
    bool hasVal3 = ((val3 & desiredMask) == desired);

    // get the bits [radixDigitPos, radixDigitPos+RADIX_BITS-1] of the values.
    bitwise_t digitInRadix0 = at::cuda::Bitfield<bitwise_t>::getBitfield(
        val0, radixDigitPos, RadixBits);
    bitwise_t digitInRadix1 = at::cuda::Bitfield<bitwise_t>::getBitfield(
        val1, radixDigitPos, RadixBits);
    bitwise_t digitInRadix2 = at::cuda::Bitfield<bitwise_t>::getBitfield(
        val2, radixDigitPos, RadixBits);
    bitwise_t digitInRadix3 = at::cuda::Bitfield<bitwise_t>::getBitfield(
        val3, radixDigitPos, RadixBits);

// counting across the warp.
#pragma unroll
    for (uint32_t j = 0; j < RadixSize; ++j) {
      // checking pattern match & digit match.
      bool vote0 = hasVal0 && (digitInRadix0 == j);
      bool vote1 = hasVal1 && (digitInRadix1 == j);
      bool vote2 = hasVal2 && (digitInRadix2 == j);
      bool vote3 = hasVal3 && (digitInRadix3 == j);

      // how many threads in this warp found digitInRadix == j while matching
      // the desired pattern?
      counts[j] += __popcll(WARP_BALLOT(vote0)) + __popcll(WARP_BALLOT(vote1)) +
          __popcll(WARP_BALLOT(vote2)) + __popcll(WARP_BALLOT(vote3));
    }
  }

  // phase 2: processing 1 element at an iteration.

  // prefetching pattern if prefetch is true.
  // prefetching pattern is only useful for global memory access.
  scalar_t v_curr;
  if constexpr (prefetch) {
    v_curr = unroll_segment + threadIdx.x < loopBound
        ? getData(unroll_segment + threadIdx.x)
        : static_cast<scalar_t>(0);
  }
  for (index_t i = unroll_segment + threadIdx.x;
       i < loopBound;
       i += blockDim.x) {
        scalar_t v_local; // the current element.
        scalar_t v_next; // the next element. Used for prefetching.

        if constexpr (prefetch) {
          // prefetch the next element.
          v_local = v_curr;
          v_next = i + blockDim.x < loopBound ? getData(i + blockDim.x)
                                              : static_cast<scalar_t>(0);
        }
        else {
          v_local = getData(i); // if no prefetching, just get the current element.
        }

        bitwise_t val = TopKTypeConfig<scalar_t>::convert(v_local);
        // check if bit pattern matches the pattern we have already discovered for
        // topk value v.
        bool hasVal = ((val & desiredMask) == desired);
        // get the bits [radixDigitPos, radixDigitPos+RADIX_BITS-1] of the value
        // v.
        bitwise_t digitInRadix = at::cuda::Bitfield<bitwise_t>::getBitfield(
            val, radixDigitPos, RadixBits);

// counting across the warp.
#pragma unroll
    for (uint32_t j = 0; j < RadixSize; ++j) {
      // checking pattern match & digit match.
      bool vote = hasVal && (digitInRadix == j);
      // how many threads in this warp found digitInRadix == j while matching
      // the desired pattern?
      counts[j] += __popcll(WARP_BALLOT(vote));
    }

    if constexpr (prefetch) {
      v_curr = v_next; // closing the prefetching loop.
    }
  }
}

// Aggregates radix matches across all warps and distributes results back to all threads.
// Uses double-buffering via buffer_index (0 or 1) to alternate between two smem segments,
// preventing race conditions between concurrent iterations. Since countRadixUsingMaskDataSmem
// performs __syncthreads() internally, at most two loop iterations can be in flight
// simultaneously, so two buffers are sufficient. buffer_index is toggled after each
// countRadixUsingMaskDataSmem invocation.
template <
    typename CountType,
    int RadixSize,
    int RadixBits>
__device__ __forceinline__ void countRadixAggregateCounts(
    CountType counts[RadixSize], // counts[i] will be the number of matching
                                 // elements ((val & desiredMask) == desired)
                                 // that have the digits [radixDigitPos,
                                 // radixDigitPos+RADIX_BITS-1] set to i.
    CountType* smem, // shared memory for inter-warp reduction of counts.
    int buffer_index){ // buffer index for smem.

  // Maximum number of warps per workgroup. HIP workgroups have at most 1024 threads.
  // Warp size is at least 32 (can be 64 on some architectures), so we use 32 for safety.
  // This sizes shared memory buffers to accommodate all possible warps: 1024/32 = 32.
  constexpr uint MAX_WARPS = 1024/32;
  const int buffer_offset = buffer_index * MAX_WARPS * RadixSize; // offset of the buffer in smem.
  const uint WARP_BITS = __builtin_ctz(warpSize);

  const uint num_warps = blockDim.x >> WARP_BITS;  // Actual number of warps in this block
  const uint warp_id = threadIdx.x >> WARP_BITS; // = threadIdx.x / warpSize
  const int lane_id = at::cuda::getLaneId(); // = threadIdx.x % warpSize

  // Stage 1: Each warp's lane 0 stores its counts in smem.
  // Layout after Stage 1: [warp0: all radix bins], [warp1: all radix bins], ...
  // this layout starts from index buffer_offset.
  if (lane_id == 0) {
#pragma unroll
    for (uint32_t i = 0; i < RadixSize; ++i) {
      smem[
            buffer_offset
          + warp_id * RadixSize
          + i
          ] = counts[i];
    }
  }

  __syncthreads(); // wait for all warps to finish storing their counts to smem.

  // Stage 2: Warp0 performs reduction for all bins.
  // Layout after Stage 2: [final radix0 sum], [final radix1 sum], ..., [final radix(RadixSize-1) sum]
  // this layout starts from index buffer_offset.
  if (warp_id == 0 && lane_id < RadixSize) {
    CountType sum = 0;
#pragma unroll
    for (int w = 0; w < num_warps; ++w) {
      sum += smem[
                    buffer_offset
                  + w * RadixSize
                  + lane_id
                  ];
    }
    smem[buffer_offset + lane_id] = sum;
  }

  __syncthreads(); // Wait for warp 0 to finish reduction.

  // Stage 3: Each thread reads the final counts from smem.
#pragma unroll
  for (uint32_t i = 0; i < RadixSize; ++i) {
    counts[i] = smem[buffer_offset + i];
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
    CountType
        counts[RadixSize], // counts[i] will be the number of matching elements
                           // ((val & desiredMask) == desired) that have the
                           // digits [radixDigitPos, radixDigitPos+RADIX_BITS-1]
                           // set to i in the warp.
    CountType* smem, // shared memory for inter-warp reduction of counts.
    int buffer_index, // buffer index for smem.
    bitwise_t
        desired, // combined with desiredMask to filter relevant elements. An
                 // element is relevant if ((val & desiredMask) == desired).
    bitwise_t
        desiredMask, // combined with desired to filter relevant elements. An
                     // element is relevant if ((val & desiredMask) == desired).
    int radixDigitPos, // position of the radix digit.
    index_t sliceSize, // size of the input slice.
    index_t withinSliceStride, // stride of the input slice.
    const scalar_t* data, // input data. This is global memory.
    const scalar_t*
        dataSmem, // input data stored in shared memory. This is shared memory.
                  // It is not initialized if dataSmemSize == 0.
    int dataSmemSize) { // input data size stored in shared memory. dataSmemSize
                        // > 0 if dataSmem is filled.

// Clear out per-thread counts from a previous round
#pragma unroll
  for (int i = 0; i < RadixSize; ++i) {
    counts[i] = 0; // initialize counts to 0.
  }

  // count the distribution of the bits in the radix digit at `radixDigitPos` to
  // `radixDigitPos`+RADIX_BITS-1 for values that match the desired pattern
  // ((val & desiredMask) == desired). counts[] will hold the results for the
  // current warp.
  if (dataSmemSize >
      0) { // if shared memory is filled, use dataSmem as the input data.
    countRadixLoop<scalar_t, bitwise_t, index_t, int, RadixSize, RadixBits, /*prefetch =*/ false>(
        counts,
        desired,
        desiredMask,
        radixDigitPos,
        dataSmemSize,
        [&](index_t i) -> scalar_t { return dataSmem[i]; });
  } else { // if shared memory is not filled, fall back to global memory.
    countRadixLoop<scalar_t, bitwise_t, index_t, int, RadixSize, RadixBits, /*prefetch =*/ true>(
        counts,
        desired,
        desiredMask,
        radixDigitPos,
        sliceSize,
        [&](index_t i) -> scalar_t {
          return doLdg(&data[i * withinSliceStride]);
        });
  }

  // aggregate counts across all warps and distribute results back to all threads.
  countRadixAggregateCounts<CountType, RadixSize, RadixBits>(
    counts,
    smem,
    buffer_index);
}

// This is the main loop of the findPattern function that finds the unique value
// that matches the pattern ((val & desired) == desiredMask) in the input data.
// DataAccessor is a function that returns the input data value at index i.
// It could potentially be a global memory accessor or a shared memory accessor.
template <
    typename scalar_t,
    typename bitwise_t,
    typename index_t,
    typename DataAccessor>
__device__ __forceinline__ scalar_t findPatternLoop(
    scalar_t* smem, // shared memory for inter-thread communication of the found
                    // value.
    bitwise_t
        desired, // combined with desiredMask to filter relevant elements. An
                 // element is relevant if ((val & desiredMask) == desired).
    bitwise_t
        desiredMask, // combined with desired to filter relevant elements. An
                     // element is relevant if ((val & desiredMask) == desired).
    index_t loopBound, // the upper bound of the loop.
    DataAccessor&&
        getData) { // a function that returns the input data value at index i.

  // TODO: this loop has two areas for improvement:
  //   1. no need to synchronize two times at each iteration. The assumption
  //   here is that the
  //      data is unique. So we can have the loop truncated to the part that
  //      smem is filled. We then do __syncthreads outside the loop. The current
  //      early termination is probably costing us way more performance than
  //      it's worth. If synchronization is moved outside the loop, we no longer
  //      need to pad loopbound to round_up(loopbound, blockDim.x).
  //   2. given this loop is potentially reading from global memory, we can
  //   prefetch the next value
  //      to improve performance. But it should not have a significant impact
  //      unless point 1 above is addressed.

  // we pad loopbound to round_up(loopbound, blockDim.x) to make sure all
  // threads in the block participate in the synchronization.
  for (index_t i = threadIdx.x;
       i < round_up(loopBound, static_cast<index_t>(blockDim.x));
       i += blockDim.x) {
    bool inRange = (i < loopBound);
    scalar_t v = inRange ? getData(i) : static_cast<scalar_t>(0);

    if (inRange &&
        ((TopKTypeConfig<scalar_t>::convert(v) & desiredMask) == desired)) {
      // There should not be conflicts if we are using findPattern,
      // since the result is unique
      smem[0] = static_cast<scalar_t>(1); // set the flag to 1.
      smem[1] = v; // store the value in smem. can't use val as the flag, since
                   // it could be 0.
    }

    __syncthreads(); // wait for all threads in the warp to finish setting the
                     // flag and storing the value.

    scalar_t found = smem[0]; // read the flag from smem.
    scalar_t val = smem[1]; // read the value from smem.

    __syncthreads(); // wait for all threads in the warp to finish reading the
                     // flag and value.

    // Checking to see if a thread found the value. If so, all threads return
    // this value.
    if (found != static_cast<scalar_t>(0)) {
      return val;
    }
  }

  CUDA_KERNEL_ASSERT(false); // should not get here.
  return static_cast<scalar_t>(0); // to make sure the compiler is happy.
}

// This function finds the unique value that matches the pattern
// ((val & desired) == desiredMask) in the input data.
// this is an smem-friendly version of the findPattern function.
// It works when data is in global memory or in shared memory.
template <typename scalar_t, typename bitwise_t, typename index_t>
__device__ scalar_t findPatternDataSmem(
    scalar_t* smem, // shared memory for inter-thread communication of the found
                    // value.
    const scalar_t* data, // input data.
    index_t sliceSize, // size of the input slice.
    index_t withinSliceStride, // stride of the input slice.
    bitwise_t
        desired, // combined with desiredMask to filter relevant elements. An
                 // element is relevant if ((val & desiredMask) == desired).
    bitwise_t
        desiredMask, // combined with desired to filter relevant elements. An
                     // element is relevant if ((val & desiredMask) == desired).
    const scalar_t* dataSmem, // input data stored in shared memory.
    index_t dataSmemSize) { // input data size stored in shared memory.

  // initialize smem to 0.
  // smem[0] is a flag to indicate if a value has been found.
  // smem[1] is the found value.
  if (threadIdx.x < 2) {
    smem[threadIdx.x] = static_cast<scalar_t>(0);
  }

  __syncthreads(); // all threads in the block wait for smem to be initialized.

  if (dataSmemSize >
      0) { // if shared memory is filled, use dataSmem as the input data.
    return findPatternLoop<scalar_t, bitwise_t, index_t>(
        smem, desired, desiredMask, dataSmemSize, [&](index_t i) -> scalar_t {
          return dataSmem[i];
        });
  } else { // if shared memory is not filled, fall back to global memory.
    return findPatternLoop<scalar_t, bitwise_t, index_t>(
        smem, desired, desiredMask, sliceSize, [&](index_t i) -> scalar_t {
          return doLdg(&data[i * withinSliceStride]);
        });
  }

  return static_cast<scalar_t>(
      0); // should not get here. This is to make sure the compiler is happy.
}

// This function fills the shared memory dataSmem with the input data.
// It is called at each iteration of the main loop of the radixSelect function.
//
// Four possible scenarios:
//    1. dataSmem is already filled (dataSmemSize > 0). This means at a previous
//    iteration
//       we have filled the shared memory with the input data. We return.
//    2. dataSmem is not filled (dataSmemSize == 0) and the input data is small
//    enough to
//       fit into shared memory (sliceSize <= dataSmemCap). If this case
//       happens, it should happen at the first iteration. In this case, we put
//       all the data into shared memory.
//    3. dataSmem is not filled (dataSmemSize == 0) and the input data, although
//    not fitting
//       into shared memory originally (otherwise we would have ended up in case
//       2), now fits into shared memory (dataSizeRemaining <= dataSmemCap). In
//       this case, filter the data using the desired pattern ((val &
//       desiredMask) == desired) and put the filtered data into shared memory.
//    4. None of the above. Data does not fit into shared memory. We return. The
//    situation
//       may change in the next iteration.
template <typename scalar_t, typename bitwise_t, typename index_t>
__device__ __forceinline__ void fillDataSmem(
    scalar_t* dataSmem, // shared memory to store the input data.
    index_t
        dataSmemCap, // max number of elements that can be stored in dataSmem.
    index_t
        dataSizeRemaining, // number of relevant elements remaining. We put data
                           // on dataSmem once dataSizeRemaining <= dataSmemCap.
    index_t& dataSmemSize, // actual number of elements in dataSmem.
    index_t sliceSize, // size of the input slice.
    index_t withinSliceStride, // stride of the input slice.
    const scalar_t* data, // input data.
    bitwise_t
        desired, // combined with desiredMask to filter relevant elements. An
                 // element is relevant if ((val & desiredMask) == desired).
    bitwise_t
        desiredMask, // combined with desired to filter relevant elements. An
                     // element is relevant if ((val & desiredMask) == desired).
    int& DataSmemWriteIndex // index used to write data to dataSmem. Incremented
                            // atomically. Shared by all threads in the block.
) {
  if (dataSmemSize > 0)
    return; // already filled

  if (sliceSize <= dataSmemCap) { // if the input data is small enough, put all
                                  // of it into shared memory.

    // reading from global memory. Prefetching to improve performance.
    scalar_t v = static_cast<scalar_t>(0);
    if (threadIdx.x < sliceSize)
      v = doLdg(&data[threadIdx.x * withinSliceStride]);
    for (index_t i = threadIdx.x; i < sliceSize; i += blockDim.x) {
      scalar_t v_next = (i + blockDim.x) < sliceSize
          ? doLdg(&data[(i + blockDim.x) * withinSliceStride])
          : static_cast<scalar_t>(0);
      dataSmem[i] = v;
      v = v_next; // closing the prefetching loop.
    }

    __syncthreads(); // wait for all threads in the block to finish writing to
                     // dataSmem.

    if (threadIdx.x == 0) {
      dataSmemSize = sliceSize; // thread 0 updates dataSmemSize to the size of
                                // the input slice.
    }

    __syncthreads(); // wait for all threads in the block to see the updated
                     // dataSmemSize.

  } else if (dataSizeRemaining <= dataSmemCap) { // if data did not fit
                                                 // originally, but now it does.
    // if this is the case, data needs to be filtered so only the relevant data
    // is stored in dataSmem. Each warp performs an internal counting of the
    // number of elements that match the desired pattern. Then reserves slots in
    // dataSmem for the matching elements by atomically incrementing
    // DataSmemWriteIndex. Finally, each thread within the warp writes its value
    // to the appropriate slot in dataSmem. This is done to minimize the amount
    // of time each warp spends waiting for others.

    int lane_id = at::cuda::getLaneId(); // = threadIdx.x % WARP_SIZE

    // prefetching from global memory.
    scalar_t v = threadIdx.x < sliceSize
        ? doLdg(&data[threadIdx.x * withinSliceStride])
        : static_cast<scalar_t>(0);

    for (index_t i = threadIdx.x; i < sliceSize;
         i += blockDim.x) {
      scalar_t v_next = (i + blockDim.x) < sliceSize
          ? doLdg(&data[(i + blockDim.x) * withinSliceStride])
          : static_cast<scalar_t>(0);

      bool match =
          (TopKTypeConfig<scalar_t>::convert(v) & desiredMask) == desired;

      // Warp-level ballot
      uint64_t ballot = WARP_BALLOT(
          match); // what threads in this warp match the desired pattern?
      int warp_count = __popcll(
          ballot); // how many threads in this warp match the desired pattern?

      int warp_base = 0; // base index to write data to dataSmem shared by all
                         // threads in the warp.
      if (lane_id == 0 &&
          warp_count >
              0) { // warp_count > 0 means there are matching elements in this
                   // warp. Only thread 0 in the warp needs to do this.
        warp_base = atomicAdd(
            &DataSmemWriteIndex,
            warp_count); // reserve warp_count slots in dataSmem for this warp,
                         // and get the base index.
      }
      warp_base = __shfl(
          warp_base, 0); // broadcast the warp_base to all threads in the warp.

      if (match) { // if the current thread has a matching value, store the
                   // value in dataSmem.
        uint64_t my_mask =
            (1ULL << lane_id) - 1; // a bitmask: [0, 0, 0, ..., 0, 1, 1, 1, ...,
                                   // 1] with (64-lane_id) 0s and lane_id 1s.
        int my_offset = __popcll(
            ballot & my_mask); // count the number of threads that have matches
                               // to the right of the current thread in bitmask.
        dataSmem[warp_base + my_offset] = v; // store the value in dataSmem.
      }

      v = v_next; // closing the prefetching loop.
    }

    __syncthreads(); // wait for all threads in the block to finish writing to
                     // dataSmem.

    if (threadIdx.x == 0) {
      dataSmemSize = DataSmemWriteIndex; // thread 0 updates dataSmemSize to the
                                         // number of elements in dataSmem.
    }

    __syncthreads(); // all threads in the block wait for dataSmemSize to be
                     // updated.
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

  // this kernel reads all the data at most (sizeof(scalar_t)*2/RADIX_BITS + 1)
  // times. if data fits into shared memory, we can avoid reading data from
  // global memory. if not, we may still be able to put the filtered data, after
  // a few iterations, into shared memory. after every pass, relevant data is
  // likely reduced by a factor of RADIX_SIZE. dataSmem is used to store the
  // relevant data.
  constexpr index_t DATA_SMEM_BYTES = 3 *
      1024; // 3KB is a good compromise between memory usage and performance.
  constexpr index_t dataSmemCap =
      DATA_SMEM_BYTES /
      sizeof(
          scalar_t); // max number of elements that can be stored in dataSmem.
  __shared__ scalar_t dataSmem[dataSmemCap];
  __shared__ index_t dataSmemSize; // actual number of elements in dataSmem.
  __shared__ index_t
      dataSizeRemaining; // number of relevant elements remaining. We put data
                         // on dataSmem once dataSizeRemaining <= dataSmemCap.
  __shared__ int DataSmemWriteIndex; // index used to write data to dataSmem.

  if (threadIdx.x == 0) {
    dataSmemSize = 0;
    DataSmemWriteIndex = 0;
    dataSizeRemaining = sliceSize;
  }

  __syncthreads(); // so the initialization is visible to all threads in the
                   // blocks.

  // buffer index for smem. We use two segments of smem for inter-warp communication of counts.
  // Given the counting operation in countRadixUsingMaskDataSmem performs __syncthreads() internally,
  // we need to alternate between the at most two segments of smem to avoid race conditions.
  // No more than two iterations of the loop will be "in flight" at any given time because
  // of the __syncthreads() in countRadixUsingMaskDataSmem.
  // buffer_index is either 0 or 1. It is toggled after each countRadixUsingMaskDataSmem invocation.
  int buffer_index = 0;

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

    // fill dataSmem with the input data if not already filled.
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

    // count the distribution of the bits in the radix digit at `digitPos` to
    // `digitPos`+RADIX_BITS-1
    countRadixUsingMaskDataSmem<
        scalar_t,
        bitwise_t,
        index_t,
        int,
        RADIX_SIZE,
        RADIX_BITS>(
        counts,
        smem,
        buffer_index,
        desired,
        desiredMask,
        digitPos,
        sliceSize,
        withinSliceStride,
        data,
        dataSmem,
        dataSmemSize);

    buffer_index ^= 1; // toggle buffer index.

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
        // find the unique value that matches the desired pattern
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
        desired = at::cuda::Bitfield<bitwise_t>::setBitfield(
            desired, i, digitPos, RADIX_BITS);
        desiredMask = at::cuda::Bitfield<bitwise_t>::setBitfield(
            desiredMask, RADIX_MASK, digitPos, RADIX_BITS);

#ifdef USE_ROCM
        if (dataSmemSize == 0) { // we only care about updating
                                 // dataSizeRemaining when dataSmem is empty.
          if (threadIdx.x == 0) {
            // this bucket has count >= kToFind elements. This means topK is in
            // this bucket and the number of elements with value & desiredMask
            // == desired (which is the relevant data) equals count. so we
            // update dataSizeRemaining to count.
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
