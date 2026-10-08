#include <ATen/ceil_div.h>
#include <c10/macros/Macros.h>
#include <ATen/cuda/AsmUtils.cuh>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/DeviceUtils.cuh>
#include <type_traits>

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
    return (static_cast<float>(v) == static_cast<float>(v)) ? (x ^ mask) : 0xffff;
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
    return (static_cast<float>(v) == static_cast<float>(v)) ? (x ^ mask) : 0xffff;
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
  // Note: uint64_t on Linux is unsigned long, but CUDA atomicAdd expects
  // unsigned long long. We use reinterpret_cast for compatibility.
  if (at::cuda::getLaneId() == 0) {
#pragma unroll
    for (uint32_t i = 0; i < RadixSize; ++i) {
      if constexpr (std::is_same_v<CountType, uint64_t>) {
        atomicAdd(reinterpret_cast<unsigned long long*>(smem) + i,
                  static_cast<unsigned long long>(counts[i]));
      } else {
        atomicAdd(&smem[i], counts[i]);
      }
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

// Moves a block-uniform value the compiler cannot prove uniform (an LDS load, or a value merged
// through lane-dependent control flow) to a scalar register, so the decisions made on it compile
// to scalar branches instead of exec-mask manipulation. A no-op under SPIR-V.
template <typename T>
__device__ __forceinline__ T blockUniform(T x) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__SPIRV__)
  if constexpr (sizeof(T) == 4) {
    return static_cast<T>(__builtin_amdgcn_readfirstlane(static_cast<int>(x)));
  } else if constexpr (sizeof(T) == 8) {
    const uint64_t u = static_cast<uint64_t>(x);
    const uint32_t lo = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(u));
    const uint32_t hi = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(u >> 32));
    return static_cast<T>((static_cast<uint64_t>(hi) << 32) | lo);
  } else {
    return x;
  }
#else
  return x;
#endif
}


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
//
// On GFX9 the per-warp counts are stored bin-major (smem[buffer_offset + bin * MAX_WARPS + warp_id]) so
// that after ONE barrier every wave reads the whole num_warps x RadixSize table with one LDS load per
// lane and reduces it with cross-lane operations. With 32-bit counts a bin is exactly one 16-lane DPP
// row (wave64, MAX_WARPS == 16): four row_shr adds leave the bin total in lane 15 and v_readlane moves it
// to a scalar register, so the decision chain in radixSelect stays wave-uniform; 64-bit counts use a
// shuffle butterfly instead. Other targets (wave32, SPIR-V) use the two-barrier form, where warp 0 sums
// the table: with 32 warps per block, every wave reducing the table costs more than the barrier saves.
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
  // Warp size is at least C10_WARP_SIZE_LOWER_BOUND, so this bounds the number of warps.
  // This sizes shared memory buffers to accommodate all possible warps.
  constexpr int MAX_WARPS = 1024/C10_WARP_SIZE_LOWER_BOUND;
  const int buffer_offset = buffer_index * MAX_WARPS * RadixSize; // offset of the buffer in smem.
  const int WARP_BITS = __builtin_ctz(C10_WARP_SIZE);

  const int num_warps = blockDim.x >> WARP_BITS;  // Actual number of warps in this block
  const int warp_id = threadIdx.x >> WARP_BITS; // = threadIdx.x / C10_WARP_SIZE
  const int lane_id = at::cuda::getLaneId(); // = threadIdx.x % C10_WARP_SIZE

#if !(defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__))
  // Not GFX9 (wave32, SPIR-V): a wave has no cheap cross-lane sum over 32 warps (each shuffle is a
  // ds_bpermute), so one wave reduces for the block and a second barrier publishes the totals. Every
  // wave reducing the table itself (below) saves that barrier but costs num_warps times the work,
  // which loses once the device is throughput bound (many blocks).
  // Stage 1: Each warp's lane 0 stores its counts in smem, warp-major.
  if (lane_id == 0) {
#pragma unroll
    for (int i = 0; i < RadixSize; ++i) {
      smem[buffer_offset + warp_id * RadixSize + i] = counts[i];
    }
  }

  __syncthreads(); // wait for all warps to finish storing their counts to smem.

  // Stage 2: Warp0 performs reduction for all bins, in place.
  if (warp_id == 0 && lane_id < RadixSize) {
    CountType sum = 0;
#pragma unroll
    for (int w = 0; w < num_warps; ++w) {
      sum += smem[buffer_offset + w * RadixSize + lane_id];
    }
    smem[buffer_offset + lane_id] = sum;
  }

  __syncthreads(); // Wait for warp 0 to finish reduction.

  // Stage 3: Each thread reads the final counts from smem.
#pragma unroll
  for (int i = 0; i < RadixSize; ++i) {
    counts[i] = smem[buffer_offset + i];
  }
#else
  // Stage 1: Each warp's lane 0 stores its counts in smem, bin-major.
  // Layout after Stage 1: [bin0: warp0..warp(MAX_WARPS-1)], [bin1: ...], ..., [bin(RadixSize-1): ...]
  // this layout starts from index buffer_offset.
  if (lane_id == 0) {
#pragma unroll
    for (int i = 0; i < RadixSize; ++i) {
      smem[buffer_offset + i * MAX_WARPS + warp_id] = counts[i];
    }
  }

  __syncthreads(); // wait for all warps to finish storing their counts to smem.

  // Stage 2: every wave reduces the table itself; the results are wave-uniform.
  if constexpr (sizeof(CountType) == 4 && MAX_WARPS * RadixSize == 64) {
    // Lane l holds the word of bin (l / 16), warp (l % 16); warps beyond num_warps contribute 0.
    uint32_t c = ((lane_id & (MAX_WARPS - 1)) < num_warps)
        ? static_cast<uint32_t>(smem[buffer_offset + lane_id]) : 0u;
    // row_shr:d (dpp_ctrl 0x110 + d) within the 16-lane row; lanes with (lane % 16) < d read 0 (bound_ctrl),
    // i.e. add nothing. The dpp_ctrl argument must be a literal, hence the four explicit steps.
    c += static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(c), 0x111, 0xf, 0xf, true));
    c += static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(c), 0x112, 0xf, 0xf, true));
    c += static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(c), 0x114, 0xf, 0xf, true));
    c += static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(c), 0x118, 0xf, 0xf, true));
#pragma unroll
    for (int i = 0; i < RadixSize; ++i) {
      counts[i] = static_cast<CountType>(static_cast<uint32_t>(
          __builtin_amdgcn_readlane(static_cast<int>(c), i * MAX_WARPS + MAX_WARPS - 1)));
    }
  } else {
    // Lane w of every warp loads warp w's count of each bin; a butterfly over the MAX_WARPS lanes
    // (lanes >= num_warps hold 0) and a broadcast from lane 0 give the block total.
#pragma unroll
    for (int i = 0; i < RadixSize; ++i) {
      CountType c = (lane_id < num_warps) ? smem[buffer_offset + i * MAX_WARPS + lane_id] : CountType(0);
#pragma unroll
      for (int d = 1; d < MAX_WARPS; d <<= 1) {
        c += __shfl_xor(c, d);
      }
      counts[i] = __shfl(c, 0);
    }
  }
#endif
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
    countRadixLoop<scalar_t, bitwise_t, index_t, CountType, RadixSize, RadixBits, /*prefetch =*/ false>(
        counts,
        desired,
        desiredMask,
        radixDigitPos,
        dataSmemSize,
        [&](index_t i) -> scalar_t { return dataSmem[i]; });
  } else { // if shared memory is not filled, fall back to global memory.
    countRadixLoop<scalar_t, bitwise_t, index_t, CountType, RadixSize, RadixBits, /*prefetch =*/ true>(
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
//
// One barrier per block-stride iteration: the matching thread publishes value then flag, the barrier
// orders those stores before every thread's reads, and the found decision is block-uniform. The flag
// and value words are double buffered by iteration parity: a wave can only reach iteration j + 2 (and
// rewrite the words of iteration j) after the barrier of iteration j + 1, which every wave arrives at
// only after finishing its reads of iteration j. Both flags are zeroed once at the start of radixSelect
// and written at most once (the match is unique and all threads return right after it is seen).
template <
    typename scalar_t,
    typename bitwise_t,
    typename index_t,
    typename DataAccessor>
__device__ __forceinline__ scalar_t findPatternLoop(
    int* foundFlag, // two shared flag words, zero on entry.
    scalar_t* foundValue, // two shared value words.
    bitwise_t
        desired, // combined with desiredMask to filter relevant elements. An
                 // element is relevant if ((val & desiredMask) == desired).
    bitwise_t
        desiredMask, // combined with desired to filter relevant elements. An
                     // element is relevant if ((val & desiredMask) == desired).
    index_t loopBound, // the upper bound of the loop.
    DataAccessor&&
        getData) { // a function that returns the input data value at index i.

  // Every thread runs the same (block-uniform) number of block-stride iterations so that all
  // threads in the block participate in the synchronization.
  const index_t stride = blockDim.x;
  const index_t numIter = (loopBound + stride - 1) / stride;
  int parity = 0;
  for (index_t it = 0; it < numIter; ++it) {
    const index_t i = it * stride + threadIdx.x;
    bool inRange = (i < loopBound);
    scalar_t v = inRange ? getData(i) : static_cast<scalar_t>(0);

    if (inRange &&
        ((TopKTypeConfig<scalar_t>::convert(v) & desiredMask) == desired)) {
      // There should not be conflicts if we are using findPattern,
      // since the result is unique
      foundValue[parity] = v; // store the value; can't use it as the flag, since it could be 0.
      foundFlag[parity] = 1; // set the flag.
    }

    __syncthreads(); // publish the flag and value to the whole block.

    if (blockUniform(foundFlag[parity]) != 0) {
      return foundValue[parity];
    }
    parity ^= 1;
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
    int* foundFlag, // two shared flag words, zero on entry.
    scalar_t* foundValue, // two shared value words.
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

  if (dataSmemSize >
      0) { // if shared memory is filled, use dataSmem as the input data.
    return findPatternLoop<scalar_t, bitwise_t, index_t>(
        foundFlag, foundValue, desired, desiredMask, dataSmemSize, [&](index_t i) -> scalar_t {
          return dataSmem[i];
        });
  } else { // if shared memory is not filled, fall back to global memory.
    return findPatternLoop<scalar_t, bitwise_t, index_t>(
        foundFlag, foundValue, desired, desiredMask, sliceSize, [&](index_t i) -> scalar_t {
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


// ROCm register-resident select (the helpers from here to radixSelectRegs).
//
// radixSelect picks one of two implementations for a slice; both return bit-identical results (the same
// decision order and tie rules), so the choice is purely about speed.
//
//   general path   1-, 2-, 4-, 8-byte types   the radix walk in radixSelect: every pass re-reads
//                                             the slice from global memory, or from dataSmem once the
//                                             survivors fit there.
//   register path  2- and 4-byte types, and 1-byte types on GFX9, with sliceSize <= REGS_MAX_E * blockDim.x:
//                  each thread loads and converts its E = 1, 4, 10 or REGS_MAX_E rows once and keeps them
//                  in registers. 8-byte types would double the register cost. 1-byte keys off GFX9 need only
//                  four 2-bit passes, and on gfx1100 the general path was faster for single short rows.
//
// radixSelectRegs runs these steps in order, each a labeled section of the function; each returns once
// it has the answer:
//   1. Load the rows and convert them to keys.
//   2. k (or n + 1 - k) == 1: one block-wide max reduction.
//   3. k (or n + 1 - k) <= REGS_SMALL_K: a threshold from wave or row maxima, then the keys at or above it
//      are ranked in dataSmem. Gives up when too many keys tie.
//   4. The radix walk over the held keys, with a one-time compaction of the survivors into dataSmem.
//      Per-pass counting: 16 bins with DPP and LDS atomics on GFX9 with a 32-bit index_t (regs16Count),
//      otherwise 2-bit ballots or packed byte counters.
// The steps stay in one function on purpose: moving any of them into a helper changed register allocation
// and scheduling, and cost 5 to 10 percent on single-row topk on gfx950 and gfx1100.
// Steps 2 and 3 answer with a key. The all-ones key (NaN for floating types, the maximum for integer
// types) is left to the walk, which publishes the original scalar and so keeps a NaN payload. (For 1- and
// 2-byte integers with `largest` on GFX9 the key carries extra high bits and is answered directly, which
// is exact since integers have no payload.)
//
// Tests: test_select_ties_at_dtype_extremes (test_sort_and_select.py) targets the step boundaries (k in
// 1, 2, 16, 17 and their mirrors), each register-path size (n = 1023 to 20480), ties at the dtype minimum,
// maximum and NaN, and tie-heavy rows that overflow step 3 into the walk.

// Wave ballot of a bool. HIP's __ballot takes an int, which makes the compiler materialize the
// predicate as 0/1 and compare it again before every ballot; the builtins take the mask directly.
__device__ __forceinline__ uint64_t ballotBool(bool p) {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_ballot_w64)) {
    return __builtin_amdgcn_ballot_w64(p);
  }
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_ballot_w32)) {
    return __builtin_amdgcn_ballot_w32(p);
  }
  return WARP_BALLOT(p);
}

// Sum of `x` over the wave, returned wave-uniform.
__device__ __forceinline__ uint32_t waveSum(uint32_t x) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
  // wave64: row_shr 1/2/4/8 leave each 16-lane row's sum in its lane 15, row_bcast15 (0x142, rows 1 and 3)
  // and row_bcast31 (0x143, rows 2 and 3) fold the rows into lane 63. dpp_ctrl must be a literal.
  x += static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x111, 0xf, 0xf, true));
  x += static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x112, 0xf, 0xf, true));
  x += static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x114, 0xf, 0xf, true));
  x += static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x118, 0xf, 0xf, true));
  x += static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x142, 0xa, 0xf, false));
  x += static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x143, 0xc, 0xf, false));
  return static_cast<uint32_t>(__builtin_amdgcn_readlane(static_cast<int>(x), 63));
#else
  for (int o = C10_WARP_SIZE / 2; o > 0; o >>= 1) {
    x += __shfl_xor(x, o);
  }
  return x;
#endif
}

// Maximum of `x` over the wave, returned wave-uniform (same DPP pattern as waveSum; 0 is the identity).
__device__ __forceinline__ uint32_t waveMax(uint32_t x) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
  x = max(x, static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x111, 0xf, 0xf, true)));
  x = max(x, static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x112, 0xf, 0xf, true)));
  x = max(x, static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x114, 0xf, 0xf, true)));
  x = max(x, static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x118, 0xf, 0xf, true)));
  x = max(x, static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x142, 0xa, 0xf, false)));
  x = max(x, static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x143, 0xc, 0xf, false)));
  return static_cast<uint32_t>(__builtin_amdgcn_readlane(static_cast<int>(x), 63));
#else
  for (int o = C10_WARP_SIZE / 2; o > 0; o >>= 1) {
    x = max(x, __shfl_xor(x, o));
  }
  return x;
#endif
}

// Minimum of `x` over the wave, returned wave-uniform (0xffffffff is the identity).
__device__ __forceinline__ uint32_t waveMin(uint32_t x) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
  x = min(x, static_cast<uint32_t>(__builtin_amdgcn_update_dpp(-1, static_cast<int>(x), 0x111, 0xf, 0xf, false)));
  x = min(x, static_cast<uint32_t>(__builtin_amdgcn_update_dpp(-1, static_cast<int>(x), 0x112, 0xf, 0xf, false)));
  x = min(x, static_cast<uint32_t>(__builtin_amdgcn_update_dpp(-1, static_cast<int>(x), 0x114, 0xf, 0xf, false)));
  x = min(x, static_cast<uint32_t>(__builtin_amdgcn_update_dpp(-1, static_cast<int>(x), 0x118, 0xf, 0xf, false)));
  x = min(x, static_cast<uint32_t>(__builtin_amdgcn_update_dpp(-1, static_cast<int>(x), 0x142, 0xa, 0xf, false)));
  x = min(x, static_cast<uint32_t>(__builtin_amdgcn_update_dpp(-1, static_cast<int>(x), 0x143, 0xc, 0xf, false)));
  return static_cast<uint32_t>(__builtin_amdgcn_readlane(static_cast<int>(x), 63));
#else
  for (int o = C10_WARP_SIZE / 2; o > 0; o >>= 1) {
    x = min(x, __shfl_xor(x, o));
  }
  return x;
#endif
}

// Maximum of `x` over each 16-lane row, valid in lane 15 of the row (row_shr 1, 2, 4, 8; lanes shifted
// in from outside the row read 0, the identity). Elsewhere every lane of the row holds it.
__device__ __forceinline__ uint32_t rowMax16(uint32_t x) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
  x = max(x, static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x111, 0xf, 0xf, true)));
  x = max(x, static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x112, 0xf, 0xf, true)));
  x = max(x, static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x114, 0xf, 0xf, true)));
  return max(x, static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x118, 0xf, 0xf, true)));
#else
  for (int o = 8; o > 0; o >>= 1) {
    x = max(x, __shfl_xor(x, o, 16));
  }
  return x;
#endif
}

// kEff-th largest of S values per lane (zeros count as values), wave-uniform, without a dependent
// extraction chain: every lane counts the values strictly greater than each of its own (one readlane
// broadcast per value, all independent), the values with fewer than kEff greater ones are the top kEff
// with multiplicity, and the smallest of those is the answer. 3 * 64 * S * S independent vector ALU
// instructions; with S == 1 that is about what 8 extraction rounds cost, with no dependency on kEff.
template <int S>
__device__ __forceinline__ uint32_t rankSelectWave(const uint32_t (&v)[S], uint32_t kEff) {
  uint32_t above[S] = {};
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
#pragma unroll
  for (int j = 0; j < 64; ++j) {
#pragma unroll
    for (int q = 0; q < S; ++q) {
      const uint32_t o = static_cast<uint32_t>(__builtin_amdgcn_readlane(static_cast<int>(v[q]), j));
#pragma unroll
      for (int r = 0; r < S; ++r) {
        above[r] += (o > v[r]) ? 1u : 0u;
      }
    }
  }
#else
  for (int j = 0; j < C10_WARP_SIZE; ++j) {
#pragma unroll
    for (int q = 0; q < S; ++q) {
      const uint32_t o = __shfl(v[q], j);
#pragma unroll
      for (int r = 0; r < S; ++r) {
        above[r] += (o > v[r]) ? 1u : 0u;
      }
    }
  }
#endif
  uint32_t best = 0xffffffffu;
#pragma unroll
  for (int r = 0; r < S; ++r) {
    best = min(best, above[r] < kEff ? v[r] : 0xffffffffu);
  }
  return waveMin(best);
}

// kEff-th largest of the 16 values held by the lanes of a 16-lane row (zeros count as values), returned
// wave-uniform; the rows must hold the same 16 values. Each lane counts the row's values greater than its
// own through 15 row rotations (no readlane, no dependency chain), then the smallest value with fewer
// than kEff greater ones is the answer. About 40 vector ALU instructions.
template <int... K>
__device__ __forceinline__ uint32_t rankSelectRow16(uint32_t x, uint32_t kEff, std::integer_sequence<int, K...>) {
  uint32_t above = 0;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
  ((above += (static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x120 + K + 1, 0xf, 0xf, false)) > x) ? 1u : 0u), ...);
#else
  ((above += (static_cast<uint32_t>(__shfl_xor(x, K + 1, 16)) > x) ? 1u : 0u), ...);
#endif
  return waveMin(above < kEff ? x : 0xffffffffu);
}
__device__ __forceinline__ uint32_t rankSelectRow16(uint32_t x, uint32_t kEff) {
  return rankSelectRow16(x, kEff, std::make_integer_sequence<int, 15>{});
}

// Number of set bits of `ballot` in lanes below the calling lane.
__device__ __forceinline__ uint32_t lanesBelow(uint64_t ballot) {
  const uint32_t r = __builtin_amdgcn_mbcnt_lo(static_cast<uint32_t>(ballot), 0u);
  return __builtin_amdgcn_mbcnt_hi(static_cast<uint32_t>(ballot >> 32), r);
}

// Stores a key (at most 8 * sizeof(scalar_t) significant bits) in a scalar_t shared-memory slot and back.
template <int N> struct UIntOfSize;
template <> struct UIntOfSize<1> { using type = uint8_t; };
template <> struct UIntOfSize<2> { using type = uint16_t; };
template <> struct UIntOfSize<4> { using type = uint32_t; };
template <typename scalar_t>
__device__ __forceinline__ scalar_t keyToSlot(uint32_t key) {
  using U = typename UIntOfSize<sizeof(scalar_t)>::type;
  return __builtin_bit_cast(scalar_t, static_cast<U>(key));
}
template <typename scalar_t>
__device__ __forceinline__ uint32_t keyFromSlot(scalar_t v) {
  using U = typename UIntOfSize<sizeof(scalar_t)>::type;
  return static_cast<uint32_t>(__builtin_bit_cast(U, v));
}

// Keeps `x` in a vector register at this point of the program (an empty asm the optimizer cannot look
// through), which stops the compiler from sinking its computation into a later conditional region.
template <typename T>
__device__ __forceinline__ void pinVgpr(T& x) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__SPIRV__)
  asm("" : "+v"(x));
#endif
}

// Largest kEff (k or n + 1 - k) that step 3 (threshold, then a rank select) handles before the radix walk.
constexpr int REGS_SMALL_K = 16;

// Drops the head of a descending run held in c[0..N) in the lanes where `hit` holds. Written as a fold
// over constant indices (no loop) so the slots stay in registers instead of an indexed array.
template <int N, int... I>
__device__ __forceinline__ void popHead(uint32_t (&c)[N], bool hit, std::integer_sequence<int, I...>) {
  ((c[I] = hit ? c[I + 1] : c[I]), ...);
  c[N - 1] = hit ? 0u : c[N - 1];
}

// Sorts c[0..N) descending with constant indices only (insertion sort as a fold; N is small).
template <int N, int... I>
__device__ __forceinline__ void sortDesc(uint32_t (&c)[N], std::integer_sequence<int, I...>) {
  auto insert = [&](auto J) {
    constexpr int j = decltype(J)::value;
    if constexpr (j > 0) {
      uint32_t t;
      [&]<int... P>(std::integer_sequence<int, P...>) {
        ((t = max(c[j - 1 - P], c[j - P]), c[j - P] = min(c[j - 1 - P], c[j - P]), c[j - 1 - P] = t), ...);
      }(std::make_integer_sequence<int, j>{});
    }
  };
  (insert(std::integral_constant<int, I>{}), ...);
}

// Register rows per thread from which the per-pass count uses packed per-lane byte counters (one
// vector add per held key, one cross-lane reduction per pass) instead of four ballots per row. Every
// type except half and bfloat16 uses it: with their packed keys it slowed single-row passes.
constexpr int REGS_PACKED_MIN_E = 4;

// How the register path keeps a held element. The generic case keeps the original scalar next to its
// key (the key alone cannot give back a NaN payload). Half and bfloat16 pack their raw bits into the upper
// half of the 32-bit key instead: the match test, the digit extraction and the masks only touch the low
// 16 bits, so the packing costs nothing and the published value stays bit-exact, NaN payload included.
template <typename scalar_t, typename bitwise_t>
struct RegKey {
  static constexpr bool kPacked = false;
  static constexpr bitwise_t kFlip = ~static_cast<bitwise_t>(0); // complements the whole key
  static __device__ __forceinline__ bitwise_t pack(scalar_t v) {
    return TopKTypeConfig<scalar_t>::convert(v);
  }
  static __device__ __forceinline__ scalar_t unpack(bitwise_t) {
    return static_cast<scalar_t>(0);
  }
};
template <>
struct RegKey<at::Half, uint32_t> {
  static constexpr bool kPacked = true;
  static constexpr uint32_t kFlip = 0xffffu; // complements the 16-bit key, not the raw bits
  static __device__ __forceinline__ uint32_t pack(at::Half v) {
    return TopKTypeConfig<at::Half>::convert(v) | (static_cast<uint32_t>(v.x) << 16);
  }
  static __device__ __forceinline__ at::Half unpack(uint32_t key) {
    return at::Half(static_cast<unsigned short>(key >> 16), at::Half::from_bits());
  }
};
template <>
struct RegKey<at::BFloat16, uint32_t> {
  static constexpr bool kPacked = true;
  static constexpr uint32_t kFlip = 0xffffu;
  static __device__ __forceinline__ uint32_t pack(at::BFloat16 v) {
    return TopKTypeConfig<at::BFloat16>::convert(v) | (static_cast<uint32_t>(v.x) << 16);
  }
  static __device__ __forceinline__ at::BFloat16 unpack(uint32_t key) {
    return at::BFloat16(static_cast<unsigned short>(key >> 16), at::BFloat16::from_bits());
  }
};

// Digit width of the register path. On wave64 GFX9 a pass resolves 4 key bits (16 bins): the per-wave
// count runs on packed one-hot counters (no ballots), the cross-wave sum on LDS atomics, and the
// decision on a 16-lane scan, so a pass costs about what a 2-bit pass costs and there are half as many.
// Elsewhere (wave32, SPIR-V, or a 64-bit index_t) the register path keeps the 2-bit digit of the general path.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
constexpr int REGS_RADIX_BITS = 4;
#else
constexpr int REGS_RADIX_BITS = RADIX_BITS;
#endif

// Largest number of rows per thread the register path is instantiated with. Each row costs about three
// VGPRs; past 96 VGPRs a wave32 GFX11 kernel drops from 16 to 12 waves per SIMD, so only one 32-wave
// block fits a WGP instead of two. Off GFX9 the last step is therefore smaller; longer slices take
// the general path. With 14, the largest gfx1100 kernels here (32-bit gatherKthValue / gatherMedian)
// compile to about 92 VGPRs; recheck that margin when changing the register path.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
constexpr int REGS_MAX_E = 20;
#else
constexpr int REGS_MAX_E = 14;
#endif

#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
// 16-bin count table in the caller's smem: 8 words per buffer, word p holds the block totals of bins 2p
// (low half) and 2p + 1 (high half); a total is at most REGS_MAX_E * 1024, so the halves never carry into each
// other. Three buffers rotate: pass t adds into buffer t % 3, zeroes buffer (t + 1) % 3 before its
// barrier (last read during pass t - 2, before barrier t - 1), and reads buffer t % 3 after its barrier.
static_assert(REGS_MAX_E * 1024 < 65536 && 8 * REGS_MAX_E < 256, "16-bit bin totals and 8-bit per-lane counters must not carry");
constexpr int REGS16_WORDS = 8;
constexpr int REGS16_BUFS = 3;

template <int Ctrl, int RowMask, bool BoundCtrl>
__device__ __forceinline__ uint32_t dppAdd(uint32_t x) {
  return x + static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), Ctrl, RowMask, 0xf, BoundCtrl));
}

// Sum of the 8 lanes of each half row, valid in lanes 8i + 7 (row_shr 1, 2, 4; lanes shifted in from
// outside the row read 0).
__device__ __forceinline__ uint32_t dppSumHalfRows(uint32_t x) {
  x = dppAdd<0x111, 0xf, true>(x);
  x = dppAdd<0x112, 0xf, true>(x);
  return dppAdd<0x114, 0xf, true>(x);
}

// Number of lanes that add their partial sum to the table (1, 2, 4 or 8): fewer lanes need more DPP
// steps (row_shr 8 folds the half rows into lanes 16i + 15, row_bcast15 the row sums of lanes 15/47
// into 31/63, row_bcast31 those into 63), more lanes make each atomic a same-address conflict of
// that depth.
constexpr int REGS16_ADD_LANES = 4;

// Sum of the wave given valid lanes 8i + 7: those lanes (REGS16_ADD_LANES == 8), lanes 16i + 15 (4),
// lanes 31 and 63 (2) or lane 63 (1) hold partial sums that together make the wave total.
__device__ __forceinline__ uint32_t dppSumHalfRowsToWave(uint32_t x) {
  if constexpr (REGS16_ADD_LANES <= 4) {
    x = dppAdd<0x118, 0xf, true>(x);
  }
  if constexpr (REGS16_ADD_LANES <= 2) {
    x = dppAdd<0x142, 0xa, false>(x);
  }
  if constexpr (REGS16_ADD_LANES == 1) {
    x = dppAdd<0x143, 0xc, false>(x);
  }
  return x;
}

// Adds this wave's 16 bin counts into the table. b[0..3] are per-lane byte counters of bins
// (0,2,4,6), (1,3,5,7), (8,10,12,14), (9,11,13,15), already summed over each half row (<= 8 * REGS_MAX_E per
// byte). They are widened to 16-bit halves with v_perm and summed into REGS16_ADD_LANES partial sums
// (lanes 16i + 15), and each of those lanes adds its 8 words.
__device__ __forceinline__ void regs16AddCounts(uint32_t* table, bool addLane, const uint32_t (&b)[4]) {
  uint32_t w[REGS16_WORDS];
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    // byte0 = byte p of the even word (bin 2p), byte2 = byte p of the odd word (bin 2p + 1), others 0.
    const uint32_t sel = 0x0c000c00u | (static_cast<uint32_t>(4 + p) << 16) | static_cast<uint32_t>(p);
    w[p] = __builtin_amdgcn_perm(b[1], b[0], sel);
    w[4 + p] = __builtin_amdgcn_perm(b[3], b[2], sel);
  }
#pragma unroll
  for (int p = 0; p < REGS16_WORDS; ++p) {
    w[p] = dppSumHalfRowsToWave(w[p]);
    // Pin the sum here: sunk into the adding lanes' branch, the last DPP add splits into a DPP move
    // plus an add.
    asm("" : "+v"(w[p]));
  }
  if (addLane) {
    // A zero the compiler cannot see through keeps the address divergent: a uniform-address atomic is
    // rewritten by the AMDGPU atomic optimizer into a per-lane readlane loop plus one atomic, which is
    // what this single-lane store already is.
    uint32_t opaqueZero = 0;
    asm("" : "+v"(opaqueZero));
#pragma unroll
    for (int p = 0; p < REGS16_WORDS; ++p) {
      atomicAdd(&table[p + opaqueZero], w[p]);
    }
  }
}

// Counts the digit at digitPos of this thread's matching keys (rows e < nValid with
// (key & desiredMask) == desired) into the table. Nibble d of a 64-bit one-hot accumulator counts the
// rows with digit d (at most 15 per accumulator); the nibbles are widened to bytes, summed over each half
// row, and handed to regs16AddCounts. With one key per thread a nibble also holds the half-row sum, so
// the two nibble words are reduced before widening.
template <int E>
__device__ __forceinline__ void regs16Count(
    uint32_t* table,
    bool addLane,
    const uint32_t (&keys)[E],
    uint32_t nValid,
    uint32_t desired,
    uint32_t desiredMask,
    int digitPos) {
  constexpr int kRowsPerAcc = 15;
  constexpr int kAccs = (E + kRowsPerAcc - 1) / kRowsPerAcc;
  uint64_t acc[kAccs] = {};
#pragma unroll
  for (int e = 0; e < E; ++e) {
    const bool match = (static_cast<uint32_t>(e) < nValid) && ((keys[e] & desiredMask) == desired);
    const uint32_t shift = static_cast<uint32_t>(at::cuda::Bitfield<uint32_t>::getBitfield(keys[e], digitPos, 4)) << 2;
    acc[e / kRowsPerAcc] += static_cast<uint64_t>(match ? 1u : 0u) << shift;
  }
  uint32_t b[4] = {0u, 0u, 0u, 0u};
  if constexpr (E == 1) {
    const uint32_t lo = dppSumHalfRows(static_cast<uint32_t>(acc[0]));
    const uint32_t hi = dppSumHalfRows(static_cast<uint32_t>(acc[0] >> 32));
    b[0] = lo & 0x0f0f0f0fu;
    b[1] = (lo >> 4) & 0x0f0f0f0fu;
    b[2] = hi & 0x0f0f0f0fu;
    b[3] = (hi >> 4) & 0x0f0f0f0fu;
  } else {
#pragma unroll
    for (int a = 0; a < kAccs; ++a) {
      const uint32_t lo = static_cast<uint32_t>(acc[a]);
      const uint32_t hi = static_cast<uint32_t>(acc[a] >> 32);
      b[0] += lo & 0x0f0f0f0fu;
      b[1] += (lo >> 4) & 0x0f0f0f0fu;
      b[2] += hi & 0x0f0f0f0fu;
      b[3] += (hi >> 4) & 0x0f0f0f0fu;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      b[i] = dppSumHalfRows(b[i]);
    }
  }
  regs16AddCounts(table, addLane, b);
}

// Decision on the aggregated table: lane i reads the total of bin i (lanes 16-63 repeat the rows), a
// row_shr scan gives the inclusive prefix sums in bin order, and the answer is the first bin whose
// prefix reaches kToFind: step = number of prefix sums below kToFind, identical in every row. Returns
// that bin and its count and the total of the bins before it, all wave-uniform.
__device__ __forceinline__ int regs16Decide(const uint16_t* bins, uint32_t kToFind, uint32_t& binCount, uint32_t& before) {
  const uint32_t c = *bins;
  uint32_t p = c;
  p = dppAdd<0x111, 0xf, true>(p);
  p = dppAdd<0x112, 0xf, true>(p);
  p = dppAdd<0x114, 0xf, true>(p);
  p = dppAdd<0x118, 0xf, true>(p);
  const int step = __popcll(__builtin_amdgcn_ballot_w64(p < kToFind)) >> 2;
  binCount = static_cast<uint32_t>(__builtin_amdgcn_readlane(static_cast<int>(c), step));
  before = static_cast<uint32_t>(__builtin_amdgcn_readlane(static_cast<int>(p), step)) - binCount;
  return step;
}
#endif

// Loads rows [Off, Off + Rem) of a contiguous slice for radixSelectRegs in chunks of V consecutive elements
// per thread (the largest power of two with V * sizeof(Raw) <= 16 that still fits), one vector load each.
// Thread t holds elements i0 .. i0 + V - 1 of a chunk, i0 = Off * blockDim.x + t * V, so the element
// order of the rows differs from the strided layout, which is free: every result is a value and the
// compaction order is arbitrary anyway. A thread whose chunk runs past the slice reads from the clamped
// base sliceSize - V (in bounds, as E > 1 implies sliceSize > blockDim.x) and keeps its rows in reverse
// element order, so its valid rows (the elements >= i0, none once i0 >= sliceSize) are a prefix of the
// chunk; chunks come in increasing element order, so a thread's valid rows are a prefix of the e range.
// A misaligned vector load (odd slice base) is legal on AMD GPUs (unaligned access mode); the compiler
// emits it from the memcpy.
template <int Off, int Rem, int MaxV, typename scalar_t, typename index_t, typename Raw, int E>
__device__ __forceinline__ void loadChunks(const scalar_t* data, index_t sliceSize, uint32_t bd, uint32_t tid, Raw (&raw)[E], uint32_t& nValid) {
  if constexpr (Rem > 0) {
    constexpr int V = Rem >= MaxV ? MaxV : (Rem >= 4 ? 4 : (Rem >= 2 ? 2 : 1));
    const index_t i0 = static_cast<index_t>(Off) * bd + static_cast<index_t>(tid) * V;
    const int held = min(max(static_cast<int>(sliceSize) - static_cast<int>(i0), 0), V);
    nValid += static_cast<uint32_t>(held);
    const index_t base = i0 < sliceSize - V ? i0 : sliceSize - V;
    if constexpr (V == 1) {
      raw[Off] = __builtin_bit_cast(Raw, data[base]);
    } else {
      using Vec = Raw __attribute__((ext_vector_type(V)));
      Vec l;
      __builtin_memcpy(&l, data + base, sizeof(Vec));
#pragma unroll
      for (int j = 0; j < V; ++j) {
        raw[Off + j] = l[V - 1 - j];
      }
    }
    loadChunks<Off + V, Rem - V, MaxV>(data, sliceSize, bd, tid, raw, nValid);
  }
}


// Register path entry (see the note above) for slices with sliceSize <= E * blockDim.x.
// Each thread holds E elements (e * blockDim.x + t, or contiguous chunks, see loadChunks), each loaded
// and converted ONCE, and every
// pass only counts the held keys, so there is no per-pass re-read or re-convert and no shared
// decision state (counts are block-uniform after the aggregation). Rows (values of e) past the slice
// are padding that never matches (the loops stay statically unrolled, so the keys stay in registers).
// For E > 1, once the surviving bin fits one element per thread (and in dataSmem) the survivors are
// compacted into dataSmem once and the remaining passes run on a single key per thread; the slot
// order is arrival dependent, the counts and the unique value are not. When a unique answer is found
// its owner publishes the ORIGINAL scalar through dataSmem[0] with one barrier (any compaction read of
// dataSmem is separated from that write by a later pass's barriers). Decision order and tie rules are
// those of the general path below, so the result is bit-identical.
//
// With the 16-bin digit (REGS_RADIX_BITS == 4) the keys are complemented when `largest`, so every pass
// walks the bins in increasing order and `desired` is uncomplemented on exit; the complement is an
// order-reversing bijection, so counts, the unique exit and the result are those of the direct walk.
template <typename scalar_t, typename bitwise_t, typename index_t, int E>
__device__ __forceinline__ void radixSelectRegs(
    const scalar_t* data,
    index_t k,
    bool largest,
    index_t sliceSize,
    index_t withinSliceStride,
    index_t* smem,
    scalar_t* dataSmem,
    index_t dataSmemCap,
    int& dataSmemWriteIndex,
    scalar_t* topK) {
  using Key = RegKey<scalar_t, bitwise_t>;
  // The 16-bin pass packs 16-bit counts and 4-bit digits of 32-bit keys.
  constexpr int RB = (REGS_RADIX_BITS == 4 && sizeof(bitwise_t) == 4 && sizeof(index_t) == 4) ? 4 : RADIX_BITS;
  constexpr int RS = 1 << RB;
  const uint32_t bd = __builtin_amdgcn_readfirstlane(blockDim.x);
  const uint32_t tid = threadIdx.x;
  // Survivors fit the single-key continuation once at most this many remain.
  const index_t compactCap = dataSmemCap < static_cast<index_t>(bd) ? dataSmemCap : static_cast<index_t>(bd);
  const bitwise_t flip = (RB == 4 && largest) ? Key::kFlip : static_cast<bitwise_t>(0);

  // Selection by reduction (below) compares keys in a flipped domain, t = (key & keyMask) ^ flipKey, where
  // the wanted extreme is the maximum and 0 the identity. The k-th largest is the (sliceSize + 1 - k)-th
  // smallest, so the nearer end of the order is used. All of this is block-uniform (k, sliceSize and
  // largest are kernel arguments). Padding rows (e >= nValid) hold the key flipKey, the minimum of that
  // domain (0, or for 1- and 2-byte integers with `largest` on GFX9 the high bits every t shares), so the
  // reductions need no validity mask; the walk masks them by e < nValid as before.
  constexpr uint32_t keyMask = sizeof(scalar_t) >= 4 ? 0xffffffffu : static_cast<uint32_t>((1u << (8 * sizeof(scalar_t))) - 1u);
  const index_t kMirror = sliceSize + 1 - k;
  const bool mirrored = kMirror < k;
  const index_t kEff = mirrored ? kMirror : k;
  const uint32_t flipSel = (largest != mirrored) ? 0u : keyMask; // t-domain -> raw key
  const uint32_t flipKey = flipSel ^ static_cast<uint32_t>(flip); // stored key -> t-domain
  const bitwise_t padKey = static_cast<bitwise_t>(flipKey);

  // Step 1: load. Every row is loaded unconditionally so the loads issue back to back (a guarded load per row puts each
  // in its own exec-mask region, where the compiler re-fetched the stride argument from the kernarg
  // segment before every load); padding rows read a clamped in-bounds index. A contiguous slice is read
  // in chunks of V consecutive elements per thread with one vector load each (loadChunks).
  const index_t stride = blockUniform(withinSliceStride);
  const index_t last = blockUniform(sliceSize - 1);
  using Raw = typename UIntOfSize<sizeof(scalar_t)>::type;
  Raw raw[E];
  uint32_t nValid = 0; // rows this thread holds (a prefix of the e range)
  if (E > 1 && stride == 1) {
    loadChunks<0, E, 16 / static_cast<int>(sizeof(Raw))>(data, sliceSize, bd, tid, raw, nValid);
  } else {
#pragma unroll
    for (int e = 0; e < E; ++e) {
      const index_t i = static_cast<index_t>(e) * bd + tid;
      const bool valid = i < sliceSize;
      raw[e] = __builtin_bit_cast(Raw, doLdg(&data[(valid ? i : last) * stride]));
      nValid += valid ? 1u : 0u;
    }
  }
  scalar_t vals[Key::kPacked ? 1 : E];
  bitwise_t keys[E];
#pragma unroll
  for (int e = 0; e < E; ++e) {
    const scalar_t v = __builtin_bit_cast(scalar_t, raw[e]);
    if constexpr (!Key::kPacked) {
      vals[e] = v;
    }
    bitwise_t key = Key::pack(v) ^ flip;
    // Pinned so the pack stays straight-line code and the select is one v_cndmask: left to itself the
    // compiler sinks the (longer, 16-bit) pack into a per-row exec-mask region behind the validity test.
    pinVgpr(key);
    keys[e] = (static_cast<uint32_t>(e) < nValid) ? key : padKey;
  }
  if (E > 1 && tid == 0) {
    dataSmemWriteIndex = 0; // ordered before any compaction by the first pass's barriers
  }

  // Steps 2 and 3: selection by reduction (block-uniform early-outs; k, sliceSize and largest are kernel
  // arguments).
  // The k-th largest is the (sliceSize + 1 - k)-th smallest, so the nearer end of the order is used.
  // Keys are compared in the flipped domain described above. The answer K is a key, and deconvert(K) is
  // bit-identical to what the radix walk publishes for every non-NaN K (convert is a bijection on non-NaN
  // values); the all-ones key falls through to the walk (see the note above).
  if constexpr (sizeof(bitwise_t) == 4) {
    constexpr int MAX_WARPS = 1024 / C10_WARP_SIZE_LOWER_BOUND;
    const int WARP_BITS = __builtin_ctz(C10_WARP_SIZE);
    const uint32_t num_warps = bd >> WARP_BITS;
    const uint32_t warp_id = tid >> WARP_BITS;
    const uint32_t lane_id = at::cuda::getLaneId();
    if (kEff == 1) {
      // Step 2. One reduction round: wave maxima through the second count buffer (unused until the second pass).
      uint32_t best = 0;
#pragma unroll
      for (int e = 0; e < E; ++e) {
        best = max(best, (static_cast<uint32_t>(keys[e]) & keyMask) ^ flipKey);
      }
      index_t* waveBest = smem + MAX_WARPS * RADIX_SIZE;
      best = waveMax(best);
      if (lane_id == 0) {
        waveBest[warp_id] = best;
      }
      __syncthreads();
      const uint32_t K = waveMax((lane_id < num_warps) ? static_cast<uint32_t>(waveBest[lane_id]) : 0u) ^ flipSel;
      if (K != keyMask) {
        *topK = TopKTypeConfig<scalar_t>::deconvert(K);
        return;
      }
    } else if (kEff <= static_cast<index_t>(REGS_SMALL_K)) {
      // Step 3. Threshold select: T, the kEff-th largest of the wave maxima (of the 16-lane row maxima when kEff
      // exceeds the wave count), is a lower bound with at least kEff keys >= T. Those keys are compacted
      // into dataSmem and the kEff-th largest among them is the answer. T and up to 64 candidates are
      // selected by rank (no per-k extraction rounds), sized for the usual case of distinct data, where the
      // candidates number about kEff: up to 16 are ranked by every wave on its own (no broadcast), up to 64
      // by wave 0, and up to CAP go through wave 0's extraction chain. Many equal keys overflow that, and the
      // walk decides.
      constexpr int MAX_WARPS = 1024 / C10_WARP_SIZE_LOWER_BOUND;
      constexpr int S3 = (REGS_SMALL_K * E + C10_WARP_SIZE_LOWER_BOUND - 1) / C10_WARP_SIZE_LOWER_BOUND;
      constexpr int SX = S3 > 2 ? S3 : 2; // extraction slots per lane
      constexpr uint32_t CAP = SX * C10_WARP_SIZE_LOWER_BOUND; // candidate slots
      constexpr int SR = 64 / C10_WARP_SIZE_LOWER_BOUND; // rank slots per lane for 64 values
      // Scratch after the first count buffer: MAX_WARPS wave maxima, 64 row maxima (1024 / 16), 1 broadcast.
      static_assert(MAX_WARPS * RADIX_SIZE + MAX_WARPS + 64 + 1 <= 256, "scratch must fit the caller's 256-word smem");
      index_t* waveMaxima = smem + MAX_WARPS * RADIX_SIZE;
      index_t* rowMaxima = waveMaxima + MAX_WARPS;
      index_t* bcast = rowMaxima + 64;
      uint32_t t[E];
      uint32_t best = 0;
#pragma unroll
      for (int e = 0; e < E; ++e) {
        t[e] = (static_cast<uint32_t>(keys[e]) & keyMask) ^ flipKey;
        best = max(best, t[e]);
      }
      const uint32_t rowsPerWave = C10_WARP_SIZE / 16;
      const uint32_t rm = rowMax16(best);
      if ((lane_id & 15) == 15) {
        rowMaxima[warp_id * rowsPerWave + (lane_id >> 4)] = rm;
      }
      const uint32_t wm = waveMax(best);
      if (lane_id == 0) {
        waveMaxima[warp_id] = wm;
      }
      if (tid == 0) {
        dataSmemWriteIndex = 0;
      }
      __syncthreads();
      const uint32_t l16 = lane_id & 15;
      uint32_t T;
      if (static_cast<uint32_t>(kEff) <= num_warps) {
        // Every wave ranks the wave maxima on its own.
        if constexpr (MAX_WARPS <= 16) {
          T = rankSelectRow16((l16 < num_warps) ? static_cast<uint32_t>(waveMaxima[l16]) : 0u, static_cast<uint32_t>(kEff));
        } else {
          uint32_t r[1] = {(lane_id < num_warps) ? static_cast<uint32_t>(waveMaxima[lane_id]) : 0u};
          T = rankSelectWave(r, static_cast<uint32_t>(kEff));
        }
      } else {
        // More wanted than waves: the row maxima (at most 64) give a usable bound; wave 0 ranks them.
        const uint32_t nRows = num_warps * rowsPerWave;
        if (warp_id == 0) {
          uint32_t r[SR];
#pragma unroll
          for (int q = 0; q < SR; ++q) {
            const uint32_t at = lane_id + static_cast<uint32_t>(q) * C10_WARP_SIZE;
            r[q] = (at < nRows) ? static_cast<uint32_t>(rowMaxima[at]) : 0u;
          }
          const uint32_t v = rankSelectWave(r, static_cast<uint32_t>(kEff));
          if (lane_id == 0) {
            *bcast = v;
          }
        }
        __syncthreads();
        T = blockUniform(static_cast<uint32_t>(*bcast));
      }
      if constexpr (E == 1) {
        // One slot reservation per wave.
        const bool cand = t[0] >= T;
        const uint64_t b = ballotBool(cand);
        int base = 0;
        if (lane_id == 0 && b != 0) {
          base = atomicAdd(&dataSmemWriteIndex, __popcll(b));
        }
        const uint32_t at = static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(base)) + lanesBelow(b);
        if (cand && at < CAP) {
          dataSmem[at] = keyToSlot<scalar_t>(t[0]);
        }
      } else {
        // Few lanes hold a key >= T, so each such lane reserves its own run of slots and writes them in row
        // order.
        uint32_t cnt = 0;
#pragma unroll
        for (int e = 0; e < E; ++e) {
          cnt += (t[e] >= T) ? 1u : 0u;
        }
        if (cnt > 0) {
          uint32_t at = static_cast<uint32_t>(atomicAdd(&dataSmemWriteIndex, static_cast<int>(cnt)));
#pragma unroll
          for (int e = 0; e < E; ++e) {
            if (t[e] >= T) {
              if (at < CAP) {
                dataSmem[at] = keyToSlot<scalar_t>(t[e]);
              }
              ++at;
            }
          }
        }
      }
      __syncthreads();
      const uint32_t total = blockUniform(static_cast<uint32_t>(dataSmemWriteIndex));
      uint32_t m;
      if (total <= 16) {
        m = rankSelectRow16((l16 < total) ? keyFromSlot<scalar_t>(dataSmem[l16]) : 0u, static_cast<uint32_t>(kEff));
      } else if (total <= 64) {
        if (warp_id == 0) {
          uint32_t c[SR];
#pragma unroll
          for (int q = 0; q < SR; ++q) {
            const uint32_t at = lane_id + static_cast<uint32_t>(q) * C10_WARP_SIZE;
            c[q] = (at < total) ? keyFromSlot<scalar_t>(dataSmem[at]) : 0u;
          }
          m = rankSelectWave(c, static_cast<uint32_t>(kEff));
          if (lane_id == 0) {
            *bcast = m;
          }
        }
        __syncthreads();
        m = blockUniform(static_cast<uint32_t>(*bcast));
      } else if (total <= CAP) {
        if (warp_id == 0) {
          uint32_t c[SX];
#pragma unroll
          for (int q = 0; q < SX; ++q) {
            const uint32_t at = lane_id * SX + q;
            c[q] = (at < total) ? keyFromSlot<scalar_t>(dataSmem[at]) : 0u;
          }
          sortDesc(c, std::make_integer_sequence<int, SX>{});
          uint32_t i = 0;
          auto extract = [&]() {
            m = waveMax(c[0]);
            const bool hit = (c[0] == m);
            i += __popcll(ballotBool(hit));
            popHead(c, hit, std::make_integer_sequence<int, SX - 1>{});
          };
          for (;;) {
            extract();
            if (i >= static_cast<uint32_t>(kEff)) {
              break;
            }
            extract();
            if (i >= static_cast<uint32_t>(kEff)) {
              break;
            }
          }
          if (lane_id == 0) {
            *bcast = m;
          }
        }
        __syncthreads();
        m = blockUniform(static_cast<uint32_t>(*bcast));
      } else {
        __syncthreads(); // every wave has read `total`
        if (tid == 0) {
          dataSmemWriteIndex = 0; // the walk's first barrier orders the reset before any compaction
        }
        m = keyMask ^ flipSel; // too many equal keys for the candidate buffer: the walk decides
      }
      const uint32_t K = m ^ flipSel;
      if (K != keyMask) {
        *topK = TopKTypeConfig<scalar_t>::deconvert(K);
        return;
      }
      // K == keyMask is ambiguous (NaN for floating types, the maximum for integers), so the walk decides.
      // The candidates above left their count in dataSmemWriteIndex; the walk's compaction must start at 0.
      __syncthreads();
      if (tid == 0) {
        dataSmemWriteIndex = 0;
      }
    }
  }

  // Step 4: the radix walk.
  bitwise_t desired = 0;
  bitwise_t desiredMask = 0;
  index_t kToFind = k;
  int buffer_index = 0;
  bool compacted = false; // block-uniform
  index_t remaining = 0; // survivors after compaction, block-uniform
  bitwise_t key1 = 0; // this thread's survivor after compaction
  scalar_t val1 = static_cast<scalar_t>(0);
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
  const int lane_id = at::cuda::getLaneId();
  const bool addLane = (lane_id & (C10_WARP_SIZE / REGS16_ADD_LANES - 1)) == C10_WARP_SIZE / REGS16_ADD_LANES - 1;
  // The 16-bin tables are 32-bit words; RB == 4 only for 4-byte index_t (int or uint32_t).
  uint32_t* smem32 = reinterpret_cast<uint32_t*>(smem);
  const uint16_t* bins16 = reinterpret_cast<const uint16_t*>(smem32) + (lane_id & 15);
  if constexpr (RB == 4) {
    if (tid < REGS16_BUFS * REGS16_WORDS) {
      smem32[tid] = 0;
    }
    __syncthreads();
  }
#endif

  for (int digitPos = sizeof(scalar_t) * 8 - RB; digitPos >= 0; digitPos -= RB) {
    int bin;
    index_t binCount;
    index_t before;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
    if constexpr (RB == 4) {
      uint32_t* table = smem32 + buffer_index * REGS16_WORDS;
      if (E == 1 || compacted) {
        const uint32_t key[1] = {compacted ? key1 : keys[0]};
        const uint32_t valid = compacted ? ((static_cast<index_t>(tid) < remaining) ? 1u : 0u) : nValid;
        regs16Count<1>(table, addLane, key, valid, desired, desiredMask, digitPos);
      } else {
        regs16Count<E>(table, addLane, keys, nValid, desired, desiredMask, digitPos);
      }
      const int nextBuffer = buffer_index == REGS16_BUFS - 1 ? 0 : buffer_index + 1;
      if (tid < REGS16_WORDS) {
        smem32[nextBuffer * REGS16_WORDS + tid] = 0;
      }
      __syncthreads();
      uint32_t c, p;
      bin = regs16Decide(bins16 + buffer_index * 2 * REGS16_WORDS, kToFind, c, p);
      binCount = c;
      before = p;
      buffer_index = nextBuffer;
    } else
#endif
    {
      index_t counts[RS];
#pragma unroll
      for (int j = 0; j < RS; ++j) {
        counts[j] = 0;
      }
      if (E > 1 && compacted) {
        const bool match = (static_cast<index_t>(tid) < remaining) && ((key1 & desiredMask) == desired);
        const uint32_t digit4 = match ? static_cast<uint32_t>(at::cuda::Bitfield<bitwise_t>::getBitfield(key1, digitPos, RB)) : RS;
#pragma unroll
        for (uint32_t j = 0; j < RS; ++j) {
          counts[j] += __popcll(ballotBool(digit4 == j));
        }
      } else if constexpr (E >= REGS_PACKED_MIN_E && !Key::kPacked) {
        // Byte b of `packed` counts this thread's matching keys with digit b (E <= 255 rows); the bins
        // are then widened to 16-bit fields (64 lanes x E < 65536) and summed across the wave once.
        uint32_t packed = 0;
#pragma unroll
        for (int e = 0; e < E; ++e) {
          const bool match = (static_cast<uint32_t>(e) < nValid) && ((keys[e] & desiredMask) == desired);
          const uint32_t digit = static_cast<uint32_t>(at::cuda::Bitfield<bitwise_t>::getBitfield(keys[e], digitPos, RB));
          packed += match ? (1u << (digit << 3)) : 0u;
        }
        const uint32_t lo = waveSum((packed & 0xffu) | ((packed & 0xff00u) << 8));
        const uint32_t hi = waveSum(((packed >> 16) & 0xffu) | ((packed >> 8) & 0xff0000u));
        counts[0] = lo & 0xffffu;
        counts[1] = lo >> 16;
        counts[2] = hi & 0xffffu;
        counts[3] = hi >> 16;
      } else {
#pragma unroll
        for (int e = 0; e < E; ++e) {
          const bool match = (static_cast<uint32_t>(e) < nValid) && ((keys[e] & desiredMask) == desired);
          // Non-matching elements vote for a fifth, uncounted bin so each ballot is one compare.
          const uint32_t digit4 = match ? static_cast<uint32_t>(at::cuda::Bitfield<bitwise_t>::getBitfield(keys[e], digitPos, RB)) : RS;
#pragma unroll
          for (uint32_t j = 0; j < RS; ++j) {
            counts[j] += __popcll(ballotBool(digit4 == j));
          }
        }
      }

      countRadixAggregateCounts<index_t, RS, RB>(counts, smem, buffer_index);
      buffer_index ^= 1;

      // Same rule as the general path, as straight-line scalar code: walking the bins largest-first or
      // smallest-first, the answer lives in the first bin whose cumulative count reaches kToFind, the bins
      // before it are skipped (kToFind shrinks by their total), and it is unique when that bin holds one
      // element and one is left to find. This decision runs on every wave, so it is kept branch-free.
      const index_t c0 = largest ? counts[3] : counts[0];
      const index_t c1 = largest ? counts[2] : counts[1];
      const index_t c2 = largest ? counts[1] : counts[2];
      const index_t c3 = largest ? counts[0] : counts[3];
      const index_t p0 = c0;
      const index_t p1 = p0 + c1;
      const index_t p2 = p1 + c2;
      const int step = (p0 < kToFind ? 1 : 0) + (p1 < kToFind ? 1 : 0) + (p2 < kToFind ? 1 : 0);
      before = step == 0 ? 0 : (step == 1 ? p0 : (step == 2 ? p1 : p2));
      binCount = step == 0 ? c0 : (step == 1 ? c1 : (step == 2 ? c2 : c3));
      bin = largest ? RS - 1 - step : step;
    }

    kToFind -= before;
    const bool unique = (binCount == 1) && (kToFind == 1);
    desired = at::cuda::Bitfield<bitwise_t>::setBitfield(desired, bin, digitPos, RB);
    desiredMask = at::cuda::Bitfield<bitwise_t>::setBitfield(desiredMask, RS - 1, digitPos, RB);

    if (unique) {
      // Exactly one held element matches; its owner publishes the original scalar.
      if (E > 1 && compacted) {
        if ((static_cast<index_t>(tid) < remaining) && ((key1 & desiredMask) == desired)) {
          dataSmem[0] = val1;
        }
      } else {
#pragma unroll
        for (int e = 0; e < E; ++e) {
          const bool match = (static_cast<uint32_t>(e) < nValid) && ((keys[e] & desiredMask) == desired);
          if (match) {
            if constexpr (Key::kPacked) {
              dataSmem[0] = Key::unpack(keys[e]);
            } else {
              dataSmem[0] = vals[e];
            }
          }
        }
      }
      __syncthreads();
      *topK = dataSmem[0];
      return;
    }

    if (E > 1 && !compacted && binCount <= compactCap) {
      // Compact the survivors into dataSmem: one slot reservation per warp (the per-row ballots are
      // cheap to redo, an LDS atomic round trip per row is not), then hold one survivor each.
      const int lane_id = at::cuda::getLaneId();
      const uint64_t lanesBelow = (1ULL << lane_id) - 1;
      uint32_t warpTotal = 0;
#pragma unroll
      for (int e = 0; e < E; ++e) {
        const bool match = (static_cast<uint32_t>(e) < nValid) && ((keys[e] & desiredMask) == desired);
        warpTotal += __popcll(ballotBool(match));
      }
      int warp_base = 0;
      if (lane_id == 0 && warpTotal > 0) {
        warp_base = atomicAdd(&dataSmemWriteIndex, static_cast<int>(warpTotal));
      }
      uint32_t slot = static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(warp_base));
#pragma unroll
      for (int e = 0; e < E; ++e) {
        const bool match = (static_cast<uint32_t>(e) < nValid) && ((keys[e] & desiredMask) == desired);
        const uint64_t ballot = ballotBool(match);
        if (match) {
          if constexpr (Key::kPacked) {
            dataSmem[slot + __popcll(ballot & lanesBelow)] = Key::unpack(keys[e]);
          } else {
            dataSmem[slot + __popcll(ballot & lanesBelow)] = vals[e];
          }
        }
        slot += __popcll(ballot);
      }
      __syncthreads();
      remaining = binCount;
      if (static_cast<index_t>(tid) < remaining) {
        val1 = dataSmem[tid];
        key1 = Key::pack(val1) ^ flip;
      }
      compacted = true;
    }
  }

  // There is no unique result, but there is a non-unique result matching `desired` exactly.
  *topK = TopKTypeConfig<scalar_t>::deconvert(desired ^ flip);
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
    index_t* smem,
    scalar_t* topK) {
  // Per-thread buckets into which we accumulate digit counts in our
  // radix
  //
  // counts must be index_t to safely handle sliceSize > INT_MAX.
  index_t counts[RADIX_SIZE];

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
  __shared__ int DataSmemWriteIndex; // index used to write data to dataSmem.
  // findPattern's flag and value words, double buffered by iteration parity (see findPatternLoop).
  __shared__ int findFlag[2];
  __shared__ scalar_t findValue[2];
  // number of relevant elements remaining. We put data on dataSmem once dataSizeRemaining <= dataSmemCap.
  // The counts it is derived from are block-uniform, so every thread keeps its own copy in a register.
  index_t dataSizeRemaining = sliceSize;
  // Register copy of dataSmemSize: fillDataSmem publishes it under a barrier and it only ever changes
  // from 0 to its final value, so once it is non-zero no pass needs to touch it (or fillDataSmem) again.
  index_t dataSmemSizeNow = 0;

  // Path choice: see the register-path note above.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
  constexpr bool kRegsPath = sizeof(scalar_t) <= 4;
#else
  constexpr bool kRegsPath = sizeof(scalar_t) == 2 || sizeof(scalar_t) == 4;
#endif
  if constexpr (kRegsPath) {
    // Slices that fit in the block's registers take the register-resident path (block-uniform choice,
    // smallest E that fits so the VGPR cost scales with the slice); it uses dataSmem for its one-time
    // compaction and as the publish slot of the unique answer, the rest of this function is skipped.
    const index_t bd = static_cast<index_t>(__builtin_amdgcn_readfirstlane(blockDim.x));
    if (sliceSize <= bd) {
      radixSelectRegs<scalar_t, bitwise_t, index_t, 1>(data, k, largest, sliceSize, withinSliceStride, smem, dataSmem, dataSmemCap, DataSmemWriteIndex, topK);
      return;
    }
    if (sliceSize <= 4 * bd) {
      radixSelectRegs<scalar_t, bitwise_t, index_t, 4>(data, k, largest, sliceSize, withinSliceStride, smem, dataSmem, dataSmemCap, DataSmemWriteIndex, topK);
      return;
    }
    if (sliceSize <= 10 * bd) {
      radixSelectRegs<scalar_t, bitwise_t, index_t, 10>(data, k, largest, sliceSize, withinSliceStride, smem, dataSmem, dataSmemCap, DataSmemWriteIndex, topK);
      return;
    }
    if (sliceSize <= REGS_MAX_E * bd) {
      radixSelectRegs<scalar_t, bitwise_t, index_t, REGS_MAX_E>(data, k, largest, sliceSize, withinSliceStride, smem, dataSmem, dataSmemCap, DataSmemWriteIndex, topK);
      return;
    }
  }

  if (threadIdx.x == 0) {
    dataSmemSize = 0;
    DataSmemWriteIndex = 0;
    findFlag[0] = 0;
    findFlag[1] = 0;
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
  index_t kToFind = k;

  // We start at the most significant digit in our radix, scanning
  // through to the least significant digit
  for (int digitPos = sizeof(scalar_t) * 8 - RADIX_BITS; digitPos >= 0;
       digitPos -= RADIX_BITS) {
    // Count radix distribution for the current position and reduce
    // across all threads

#ifdef USE_ROCM

    // fill dataSmem with the input data if not already filled (block-uniform decision).
    if (dataSmemSizeNow == 0) {
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
      dataSmemSizeNow = blockUniform(dataSmemSize);
    }

    // count the distribution of the bits in the radix digit at `digitPos` to
    // `digitPos`+RADIX_BITS-1
    countRadixUsingMaskDataSmem<
        scalar_t,
        bitwise_t,
        index_t,
        index_t,
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
        dataSmemSizeNow);

    buffer_index ^= 1; // toggle buffer index.

#else
    countRadixUsingMask<
        scalar_t,
        bitwise_t,
        index_t,
        index_t,
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
    auto found_unique = [&](int i, index_t count) -> bool {
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
            findFlag,
            findValue,
            data,
            sliceSize,
            withinSliceStride,
            desired,
            desiredMask,
            dataSmem,
            dataSmemSizeNow);
#endif
        return true;
      }
      return false;
    };
    auto found_non_unique = [&](int i, index_t count) -> bool {
      if (count >= kToFind) {
        desired = at::cuda::Bitfield<bitwise_t>::setBitfield(
            desired, i, digitPos, RADIX_BITS);
        desiredMask = at::cuda::Bitfield<bitwise_t>::setBitfield(
            desiredMask, RADIX_MASK, digitPos, RADIX_BITS);

#ifdef USE_ROCM
        if (dataSmemSizeNow == 0) { // we only care about updating
                                    // dataSizeRemaining when dataSmem is empty.
          // this bucket has count >= kToFind elements. This means topK is in
          // this bucket and the number of elements with value & desiredMask
          // == desired (which is the relevant data) equals count. so we
          // update dataSizeRemaining to count. count is block-uniform, so
          // this needs neither a shared store nor a barrier.
          dataSizeRemaining = count;
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
        index_t count = counts[i];
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
        index_t count = counts[i];
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
