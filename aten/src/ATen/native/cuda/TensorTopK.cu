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
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/util/CallOnce.h>

#include <array>

#if defined(USE_ROCM)
#include <rocprim/block/block_scan.hpp>
#endif

using namespace at::native;

namespace at::native {

#ifdef USE_ROCM
static bool is_gfx1250_device(c10::DeviceIndex device_index) {
  TORCH_INTERNAL_ASSERT(device_index >= 0 && device_index < C10_COMPILE_TIME_MAX_GPUS);
  static std::array<c10::once_flag, C10_COMPILE_TIME_MAX_GPUS> once_flags;
  static std::array<bool, C10_COMPILE_TIME_MAX_GPUS> results{};
  c10::call_once(once_flags[device_index], [device_index] {
    results[device_index] = at::detail::getCUDAHooks().isGPUArch({"gfx1250"}, device_index);
  });
  return results[device_index];
}
#endif

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
  // smem is used for:
  // 1. radixSelect: radix bin counts (RADIX_SIZE=4 elements)
  // 2. exclusiveBinaryPrefixScan: warp prefix sums (≤32 elements)
  // 3. findPattern: flag and value (2 elements, cast to scalar_t*)
  //
  // Type must be IndexType to safely handle sliceSize > INT_MAX.
  // In radix selection, counts can exceed INT_MAX when billions of
  // elements fall into a single radix bin.
  __shared__ IndexType smem[32];
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

    IndexType index;
    IndexType carry;
    at::cuda::exclusiveBinaryPrefixScan<IndexType, true>(
        smem, hasTopK, &index, &carry, AddOp<IndexType>());

    if (hasTopK) {
      IndexType writeIndex = writeIndexStart + index;
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

    IndexType index;
    IndexType carry;
    at::cuda::exclusiveBinaryPrefixScan<IndexType, true>(
        smem, hasTopK, &index, &carry, AddOp<IndexType>());

    if (hasTopK && index < topKRemaining) {
      IndexType writeIndex = writeIndexStart + index;
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
exclusiveBinaryPrefixScan to calculate the index to write the result to, which costs two block level
synchronization points per iteration. Here each warp counts its matches with a ballot and publishes the count to
shared memory; after a single barrier every thread derives its warp's exclusive prefix and the iteration total.
Results therefore land in ascending input-index order, exactly like the prefix-scan version, so the output is
deterministic run to run (an atomicAdd-based reservation would make the order depend on warp scheduling).
The counts are double buffered, so the next iteration's barrier also orders this iteration's reads before the
buffer is overwritten, and one barrier per iteration suffices. The running write index is kept in a register by
every thread; it is identical across the block, which keeps the phase 2 early exit block-uniform.

The barrier is amortized over kGatherBatch block-strided sub-iterations per round: sub-iteration u of a round
covers input indices base + u * blockDim.x + threadIdx.x, so u-major order is ascending input-index order. The
per-warp counts of two sub-iterations are packed into one 32-bit word (16 bits each; a sub-iteration counts at
most blockDim.x <= 1024 matches per block, so a field never carries into its neighbour) and scanned together.
Slices that fit one block stride take a single round per phase.

Phase 1 also ballots "== k-th value" per sub-iteration and scans those counts as extra packed fields, so every
thread learns the ascending-input-index rank of each equal element it holds and keeps up to kEqCap of them in
registers (value, index, rank). Unless some thread held more than kEqCap (a per-warp sticky flag published in the
last round, OR-reduced with one ballot), phase 2 is then two predicated writes with no re-read of the slice, no
loop and no barrier; otherwise the re-reading phase 2 runs as before.
*/

// Sub-iterations per barrier round in the gather phases.
constexpr int kGatherBatch = 4;
// num_warps <= 1024 / C10_WARP_SIZE_LOWER_BOUND <= 32, so a warp-count scan fits in one warp.
constexpr int kMaxWarps = 1024 / C10_WARP_SIZE_LOWER_BOUND;
// Phase 1 counts two predicates per sub-iteration (beyond the k-th value, equal to it), phase 2 one; two
// 16-bit per-warp counts share a 32-bit word.
constexpr int kMaxFields = 2 * kGatherBatch;
template <int F>
constexpr int countWords() { return (F + 1) / 2; }
constexpr int kCountWords = countWords<kMaxFields>();
// Equal elements each thread keeps ranked in registers for the loop-free phase 2.
constexpr int kEqCap = 2;
static_assert(kEqCap >= 1, "a single-round slice must never overflow the held equal elements");

// Inclusive scan of the per-warp counts held by lanes 0..num_warps-1; returns the block total and this warp's
// exclusive prefix as wave-uniform values. The combine after the barrier is a serial dependency chain, so it
// is written to minimize dependent cross-lane steps: on GFX9 (wave64, so <= 16 warps per 1024-thread block)
// the scan is four v_add_u32_dpp row_shr steps inside one 16-lane row and the two broadcasts are v_readlane
// (their lane index is wave-uniform). wave32 keeps the shuffle scan, SPIR-V also the shuffle broadcasts.
__device__ __forceinline__ void scanWarpCounts(uint32_t c, int lane_id, int warp_id, int num_warps,
                                               uint32_t& total, uint32_t& prefix) {
  uint32_t incl = c;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
  // Block-uniform branch; the compiler drops it where it can prove num_warps <= 16.
  if (num_warps <= 16) {
    // row_shr:d (dpp_ctrl 0x110 + d) within the 16-lane row; lanes with (lane % 16) < d read 0 (bound_ctrl),
    // i.e. add nothing. The dpp_ctrl argument must be a literal, hence the four explicit steps.
    incl += static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(incl), 0x111, 0xf, 0xf, true));
    incl += static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(incl), 0x112, 0xf, 0xf, true));
    incl += static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(incl), 0x114, 0xf, 0xf, true));
    incl += static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(incl), 0x118, 0xf, 0xf, true));
  } else
#endif
  {
    #pragma unroll
    for (int d = 1; d < kMaxWarps; d <<= 1) {
      if (d >= num_warps) break;  // block-uniform
      uint32_t n = __shfl_up(incl, d);
      incl += (lane_id >= d) ? n : 0;
    }
  }
  const uint32_t excl = incl - c;
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__SPIRV__)
  total = static_cast<uint32_t>(__builtin_amdgcn_readlane(static_cast<int>(incl), num_warps - 1));
  prefix = static_cast<uint32_t>(__builtin_amdgcn_readlane(static_cast<int>(excl), warp_id));
#else
  total = __shfl(incl, num_warps - 1);
  prefix = __shfl(excl, warp_id);
#endif
}

// Moves a scan output the compiler cannot prove wave-uniform to a scalar register (a no-op where
// scanWarpCounts already returns SGPRs).
__device__ __forceinline__ uint32_t asUniform(uint32_t x) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__SPIRV__)
  return x;
#else
  return static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(x));
#endif
}

// For each of the round's F ballots, returns the block total of set bits and the exclusive prefix over the
// warps before this one, both wave-uniform; the per-warp counts travel through LDS two per 32-bit word.
// warpCounts points at this round's [countWords<F>()][kMaxWarps] buffer. All threads must call it.
template <int F>
__device__ __forceinline__ void orderedWarpOffset(const uint64_t (&ballot)[kMaxFields],
                                                  uint32_t* warpCounts,
                                                  uint32_t (&total)[kMaxFields],
                                                  uint32_t (&prefix)[kMaxFields]) {
  constexpr int kWords = countWords<F>();
  const int lane_id = at::cuda::getLaneId();
  const int warp_id = threadIdx.x / C10_WARP_SIZE;
  const int num_warps = blockDim.x / C10_WARP_SIZE;
  if (lane_id == 0) {
    #pragma unroll
    for (int w = 0; w < kWords; ++w) {
      uint32_t packed = __popcll(ballot[2 * w]);
      if (2 * w + 1 < F) {
        packed |= static_cast<uint32_t>(__popcll(ballot[2 * w + 1])) << 16;
      }
      warpCounts[w * kMaxWarps + warp_id] = packed;
    }
  }
  __syncthreads();
  // Lane w of every warp loads warp w's packed counts; an inclusive scan across lanes then gives every
  // warp's prefix for both fields at once. The outputs are wave-uniform, so the unpack is scalar.
  #pragma unroll
  for (int w = 0; w < kWords; ++w) {
    uint32_t c = (lane_id < num_warps) ? warpCounts[w * kMaxWarps + lane_id] : 0;
    uint32_t t, p;
    scanWarpCounts(c, lane_id, warp_id, num_warps, t, p);
    t = asUniform(t);
    p = asUniform(p);
    total[2 * w] = t & 0xFFFF;
    prefix[2 * w] = p & 0xFFFF;
    if (2 * w + 1 < F) {
      total[2 * w + 1] = t >> 16;
      prefix[2 * w + 1] = p >> 16;
    }
  }
}

// helper function to write the result to the output.
template <typename T, typename IndexType>
__device__ __forceinline__ void writeResult(T* topKSliceStart,
                                            int64_t* indicesSliceStart,
                                            IndexType topKWithinSliceStride,
                                            IndexType indicesWithinSliceStride,
                                            IndexType writeIndex,
                                            T v,
                                            IndexType i){
  // Callers bound writeIndex: phase-1 rounds check it once per round, phase 2 and the held equals clip it.
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

  // smem and counts must use IndexType to safely handle sliceSize > INT_MAX.
  // In radix selection, counts tracks elements matching a radix pattern,
  // which can exceed INT_MAX when billions of elements fall into one bin.

  // Maximum shared memory size for radix select (used in countRadixAggregateCounts): NUM_BUFFERS * MAX_WARPS * RADIX_SIZE.
  // HIP workgroups have at most 1024 threads. Warp size is at least 32 (can be 64 on some
  // architectures), so we use 32 for safety: 2 buffers * (1024/32) warps * 4 radix bins = 256.
  __shared__ IndexType smem[256];
  // Packed per-warp match counts for the ordered compaction, double buffered; sized for wave32.
  __shared__ uint32_t warpCounts[2][kCountWords * kMaxWarps];
  // Per-warp "some lane holds more than kEqCap equal elements" flag, published in the last phase-1 round.
  __shared__ int warpOverflow[kMaxWarps];

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
  // the resulting values. orderedWarpOffset computes it.

  // orderedWarpOffset has a barrier, so every thread must run the same number of rounds. Slices that fit in
  // one block-stride take a single lean round per phase (U = 1, no prefetch). Longer slices run kGatherBatch
  // sub-iterations per round; the last round is peeled since it is the only one with out-of-range elements
  // and the only one without a prefetch, so the loop body issues its (clamped, in-bounds) prefetch
  // unconditionally and the compiler can keep it in flight instead of draining vector memory before the compare.
  using RadixType = typename TopKTypeConfig<T>::RadixType;
  const IndexType stride = blockDim.x;
  // Offsets are in units of inputWithinSliceStride; the per-thread part is fixed, the per-round part uniform.
  const IndexType tidOff = threadIdx.x * inputWithinSliceStride;
  const IndexType rowStep = stride * inputWithinSliceStride;
  const IndexType lastOff = (inputSliceSize - 1) * inputWithinSliceStride;
  // key > kthKey is "strictly beyond the k-th value" in the sort direction for both directions.
  const RadixType flip = largest ? RadixType(0) : ~RadixType(0);
  const RadixType kthKey = topKConverted ^ flip;
  const int lane_id = at::cuda::getLaneId();
  const int warp_id = threadIdx.x / C10_WARP_SIZE;
  const int num_warps = blockDim.x / C10_WARP_SIZE;
  IndexType writeIndex = 0;
  int buf = 0;
  T v[kGatherBatch];
  // Equal elements seen so far: block-uniform count, and this thread's first kEqCap of them with their ranks.
  IndexType eqRunning = 0;
  int eqHeld = 0;
  T eqVal[kEqCap];
  IndexType eqIdx[kEqCap];
  IndexType eqRank[kEqCap];
  #pragma unroll
  for (int e = 0; e < kEqCap; ++e) {
    eqVal[e] = static_cast<T>(0);
    eqIdx[e] = 0;
    eqRank[e] = 0;
  }

  // One barrier round over U sub-iterations starting at `base` (rowOff = base * inputWithinSliceStride);
  // unless kLast it also prefetches the next round's U values. Phase 1 ballots "beyond" in fields 0..U-1 and
  // "equal" in fields U..2U-1; phase 2 ballots "equal" in fields 0..U-1. kFlagOverflow (last phase-1 round of a
  // multi-round slice) publishes the per-warp held-equal overflow flag; a single U = 1 round cannot overflow.
  auto gatherRound = [&]<int U, bool kPhase2, bool kLast>(IndexType base, IndexType rowOff) {
    // The overflow flag is published in the last phase-1 round of a multi-round slice.
    constexpr bool kFlagOverflow = !kPhase2 && kLast && U > 1;
    constexpr int F = kPhase2 ? U : 2 * U;
    T v_next[kGatherBatch];
    if constexpr (!kLast) {
      #pragma unroll
      for (int u = 0; u < U; ++u) {
        v_next[u] = doLdg(&inputSliceStart[::min(rowOff + (U + u) * rowStep + tidOff, lastOff)]);
      }
    }

    bool has[kMaxFields];
    uint64_t ballot[kMaxFields];
    #pragma unroll
    for (int u = 0; u < U; ++u) {
      const RadixType key = TopKTypeConfig<T>::convert(v[u]) ^ flip;
      const bool match = kPhase2 ? (key == kthKey) : (key > kthKey);
      const bool equal = key == kthKey;
      if constexpr (kLast) {
        const bool inRange = base + u * stride + threadIdx.x < inputSliceSize;
        const uint64_t inMask = WARP_BALLOT(inRange);
        has[u] = match && inRange;
        ballot[u] = WARP_BALLOT(match) & inMask;
        if constexpr (!kPhase2) {
          has[U + u] = equal && inRange;
          ballot[U + u] = WARP_BALLOT(equal) & inMask;
        }
      } else {
        has[u] = match;
        ballot[u] = WARP_BALLOT(match);
        if constexpr (!kPhase2) {
          has[U + u] = equal;
          ballot[U + u] = WARP_BALLOT(equal);
        }
      }
    }
    if constexpr (kFlagOverflow) {
      // The overflow condition is sticky, so the flag computed in the last round covers every round. It is
      // written before this round's barrier and read after phase 1.
      int eqAfter = eqHeld;
      #pragma unroll
      for (int u = 0; u < U; ++u) {
        eqAfter += has[U + u] ? 1 : 0;
      }
      const bool overflow = WARP_BALLOT(eqAfter > kEqCap) != 0;
      if (lane_id == 0) {
        warpOverflow[warp_id] = overflow;
      }
    }

    uint32_t total[kMaxFields];
    uint32_t prefix[kMaxFields];
    orderedWarpOffset<F>(ballot, warpCounts[buf], total, prefix);

    if constexpr (!kPhase2) {
      // The per-write bound check of writeResult, hoisted to one block-uniform check per round ahead of the
      // round's writes (phase 2 clips its writes explicitly).
      IndexType roundTotal = 0;
      #pragma unroll
      for (int u = 0; u < U; ++u) {
        roundTotal += total[u];
      }
      CUDA_KERNEL_ASSERT(writeIndex + roundTotal <= outputSliceSize);
    }

    // total/prefix are wave-uniform: the running indices stay in scalar registers.
    IndexType running = writeIndex;
    IndexType eqRun = eqRunning;
    #pragma unroll
    for (int u = 0; u < U; ++u) {
      const IndexType idx = base + u * stride + threadIdx.x;
      const IndexType slot = running + prefix[u] + laneRank(ballot[u]);
      running += total[u];
      bool doWrite = has[u];
      if constexpr (kPhase2) {
        doWrite = doWrite && slot < outputSliceSize;
      }
      if (doWrite) {
        writeResult(topKSliceStart,
          indicesSliceStart,
          topKWithinSliceStride,
          indicesWithinSliceStride,
          /*writeIndex=*/slot,
          /*value=*/v[u],
          /*index=*/idx);
      }
      if constexpr (!kPhase2) {
        if (has[U + u]) {
          const IndexType rank = eqRun + prefix[U + u] + laneRank(ballot[U + u]);
          #pragma unroll
          for (int e = 0; e < kEqCap; ++e) {
            if (eqHeld == e) {
              eqVal[e] = v[u];
              eqIdx[e] = idx;
              eqRank[e] = rank;
            }
          }
          ++eqHeld;
        }
        eqRun += total[U + u];
      }
      if constexpr (!kLast) {
        v[u] = v_next[u];
      }
    }
    writeIndex = running;
    eqRunning = eqRun;
    buf ^= 1;
  };

  // After phase 1: fills the output from the register-held equal elements and returns true, unless some
  // thread held more than kEqCap of them (block-uniform decision), in which case the caller re-reads.
  auto finishWithHeldEquals = [&](bool mayOverflow) -> bool {
    if (writeIndex >= outputSliceSize) {
      return true;
    }
    if (mayOverflow) {
      const bool warpFlag = lane_id < num_warps && warpOverflow[lane_id] != 0;
      if (WARP_BALLOT(warpFlag) != 0) {
        return false;
      }
    }
    const IndexType need = outputSliceSize - writeIndex;
    #pragma unroll
    for (int e = 0; e < kEqCap; ++e) {
      if (eqHeld > e && eqRank[e] < need) {
        writeResult(topKSliceStart,
          indicesSliceStart,
          topKWithinSliceStride,
          indicesWithinSliceStride,
          /*writeIndex=*/writeIndex + eqRank[e],
          /*value=*/eqVal[e],
          /*index=*/eqIdx[e]);
      }
    }
    return true;
  };

  if (inputSliceSize <= stride) {
    // Short slice: one round per phase.
    v[0] = doLdg(&inputSliceStart[::min(tidOff, lastOff)]);
    gatherRound.template operator()<1, false, true>(0, 0);
    // A single round never overflows the held equals (kEqCap >= 1), so this always finishes the output.
    finishWithHeldEquals(/*mayOverflow=*/false);
    return;
  }

  const IndexType roundStride = stride * kGatherBatch;
  const IndexType roundStep = roundStride * inputWithinSliceStride;
  const IndexType numIterations = round_up(inputSliceSize, roundStride);
  const IndexType lastBase = numIterations - roundStride;

  auto loadFirstRound = [&]() {
    #pragma unroll
    for (int u = 0; u < kGatherBatch; ++u) {
      v[u] = doLdg(&inputSliceStart[::min(u * rowStep + tidOff, lastOff)]);
    }
  };

  // phase 1: write actual > `pattern` (or < `pattern`, depending on the sort direction) values to the output.
  loadFirstRound();
  IndexType rowOff = 0;
  for (IndexType base = 0; base < lastBase; base += roundStride, rowOff += roundStep) {
    gatherRound.template operator()<kGatherBatch, false, false>(base, rowOff);
  }
  gatherRound.template operator()<kGatherBatch, false, true>(lastBase, 0);

  // We need to fill in the rest with actual == top-K values.
  // The number that we need is outputSliceSize - writeIndex.
  // There might be more than that number available in input,
  // in which case we have to choose the first seen set.

  if (finishWithHeldEquals(/*mayOverflow=*/true)) {
    return;
  }

  // phase 2 (many ties): write actual == `pattern` values to the output, until the output is full.
  // writeIndex is the same in every thread, so the whole block exits together.
  loadFirstRound();
  rowOff = 0;
  for (IndexType base = 0; base < lastBase; base += roundStride, rowOff += roundStep) {
    if (writeIndex >= outputSliceSize) {
      break;
    }
    gatherRound.template operator()<kGatherBatch, true, false>(base, rowOff);
  }
  if (writeIndex < outputSliceSize) {
    gatherRound.template operator()<kGatherBatch, true, true>(lastBase, 0);
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
// Largest slice that should_use_warp_topk sends to the warp sort on every ROCm
// GPU. Larger slices, up to MAX_WARP_TOPK_SLICE, take it only on gfx1250 and
// only in place of mbtopk.
constexpr int DEFAULT_MAX_WARP_TOPK_SLICE = 256;

// Orders values by their radix key (TopKTypeConfig::convert), the order sbtopk
// and mbtopk select in: unlike GTOp/LTOp it puts +0 above -0. The
// sort_size > DEFAULT_MAX_WARP_TOPK_SLICE instantiations, which replace mbtopk
// on gfx1250, sort with it so that they select the same elements.
template <typename scalar_t, bool is_descending>
struct RadixKeyOrder {
  __device__ bool operator()(const scalar_t& lhs, const scalar_t& rhs) const {
    const auto l = TopKTypeConfig<scalar_t>::convert(lhs);
    const auto r = TopKTypeConfig<scalar_t>::convert(rhs);
    return is_descending ? l > r : l < r;
  }
};

// Writes the top-k of one slice in the order that sbtopk/mbtopk's gather
// produces for an unsorted topk, which slices above
// DEFAULT_MAX_WARP_TOPK_SLICE take on other GPUs: first the values that sort
// strictly before the k-th value, by increasing index, then the lowest indices
// of values equal to it. These are exactly the elements the stable sort
// selected. Called by the single warp of the block that owns the slice.
template <typename scalar_t, typename IndexType, bool is_descending, int WARP_SIZE, typename Scan,
          typename InputIt, typename ValuesIt, typename IndicesIt>
__device__ __forceinline__ void storeTopKInGatherOrder(
    typename Scan::TempStorage& scan_storage,
    scalar_t kth,
    InputIt input,
    IndexType slice_size,
    IndexType k,
    ValuesIt values,
    IndicesIt indices) {
  const RadixKeyOrder<scalar_t, is_descending> sorts_before;
  IndexType written = 0;
  // Pass 0 writes the values that sort before the k-th, pass 1 the ties with it.
  for (int pass = 0; pass < 2; ++pass) {
    const bool ties = pass == 1;
    for (IndexType base = 0; base < slice_size && written < k; base += WARP_SIZE) {
      const IndexType idx = base + threadIdx.x;
      scalar_t v{};
      bool take = false;
      if (idx < slice_size) {
        v = input[idx];
        take = ties ? !sorts_before(v, kth) && !sorts_before(kth, v) : sorts_before(v, kth);
      }
      int offset, total;
      Scan(scan_storage).ExclusiveSum(take ? 1 : 0, offset, total);
      // WARP_SYNC() is a no-op on ROCm; the block is a single warp, so a block
      // barrier orders the reuse of the scan's temp storage.
      __syncthreads();
      if (take && written + offset < k) {
        values[written + offset] = v;
        indices[written + offset] = idx;
      }
      written += total;
    }
  }
}

// Comparator for sorting with TopK semantics
// Note: The WarpMergeSort instantiations up to DEFAULT_MAX_WARP_TOPK_SLICE sort with
// the comparison operators GTOp/LTOp; the larger ones sort with RadixKeyOrder.

// Kernel using WarpMergeSort for small topK operations
// See Note [warp merge sort WARP_SIZE template param]
template <int KeyDims, int ValueDims, int sort_size, int max_block_dim_y,
          typename scalar_t, typename IndexType, bool is_descending, int WARP_SIZE>
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

#if !(defined(__gfx1250__) || defined(__gfx1250_strict__) || defined(__SPIRV__))
  if constexpr (sort_size > DEFAULT_MAX_WARP_TOPK_SLICE) {
    // should_use_warp_topk sends slices above DEFAULT_MAX_WARP_TOPK_SLICE here on gfx1250 only.
    CUDA_KERNEL_ASSERT(false && "warpMergeSortTopK above DEFAULT_MAX_WARP_TOPK_SLICE is gfx1250-only");
    return;
  }
#endif

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

  CUDA_KERNEL_ASSERT(blockDim.x == WARP_SIZE);
  CUDA_KERNEL_ASSERT(blockDim.y <= max_block_dim_y);
  constexpr int items_per_thread = sort_size / WARP_SIZE;
  static_assert(items_per_thread * WARP_SIZE == sort_size,
                "sort_size must be a multiple of WARP_SIZE template param");

  using LoadKeys = cub::WarpLoad<scalar_t, items_per_thread, cub::WARP_LOAD_TRANSPOSE>;
  using Sort = cub::WarpMergeSort<scalar_t, items_per_thread, WARP_SIZE, int64_t>;
  using StoreKeys = cub::WarpStore<scalar_t, items_per_thread, cub::WARP_STORE_TRANSPOSE>;
  using StoreIndices = cub::WarpStore<int64_t, items_per_thread, cub::WARP_STORE_TRANSPOSE>;
  using Scan = cub::WarpScan<int, WARP_SIZE>;

  __shared__ union {
    typename LoadKeys::TempStorage load_keys;
    typename Sort::TempStorage sort;
    typename StoreKeys::TempStorage store_keys;
    typename StoreIndices::TempStorage store_indices;
    typename Scan::TempStorage scan;
  } tmp_storage[max_block_dim_y];
  __shared__ scalar_t kth_value[max_block_dim_y];

  auto& warp_storage = tmp_storage[threadIdx.y];

  // Thread-local arrays for values and indices
  scalar_t local_values[items_per_thread];
  int64_t local_indices[items_per_thread];

  // Invalid sentinel for padding.
  //
  // For integer types std::numeric_limits<T>::infinity() returns 0 (the
  // default-constructed value of T), which is a legal value in the input and
  // therefore not a valid out-of-band sentinel. With a zero sentinel the
  // padding slots interleave with real input values during the sort and their
  // placeholder indices (inputSliceSize..sort_size-1) can leak into the top-k
  // output, producing out-of-bounds indices.
  //
  // Use has_infinity to pick a true "sorts-to-the-end" sentinel for each type:
  // infinity() for floats, lowest()/max() for integer types.
  //
  // The radix-key sort of the larger instantiations needs the padding to sort
  // last by radix key: NaN, not infinity, has the largest key.
  constexpr bool kRadixKeySort = sort_size > DEFAULT_MAX_WARP_TOPK_SLICE;
  const scalar_t invalid_value = is_descending
      ? (std::numeric_limits<scalar_t>::has_infinity
             ? -std::numeric_limits<scalar_t>::infinity()
             : std::numeric_limits<scalar_t>::lowest())
      : (kRadixKeySort && std::numeric_limits<scalar_t>::has_quiet_NaN
             ? std::numeric_limits<scalar_t>::quiet_NaN()
             : (std::numeric_limits<scalar_t>::has_infinity
                    ? std::numeric_limits<scalar_t>::infinity()
                    : std::numeric_limits<scalar_t>::max()));

  // Initialize indices for this slice in blocked arrangement
  // WARP_LOAD_TRANSPOSE uses blocked layout: thread t gets items [t*items_per_thread, t*items_per_thread+items_per_thread-1]
  #pragma unroll
  for (int i = 0; i < items_per_thread; ++i) {
    local_indices[i] = threadIdx.x * items_per_thread + i;
  }

  // Load values from input
  LoadKeys(warp_storage.load_keys).Load(input_iter, local_values, inputSliceSize, invalid_value);
  WARP_SYNC();

  // Sort values with their indices: by radix key above DEFAULT_MAX_WARP_TOPK_SLICE,
  // otherwise GTOp (largest=true, larger values first) or LTOp (smaller first)
  if constexpr (kRadixKeySort) {
    Sort(warp_storage.sort).StableSort(
        local_values, local_indices, RadixKeyOrder<scalar_t, is_descending>(), inputSliceSize, invalid_value);
  } else if constexpr (is_descending) {
    Sort(warp_storage.sort).StableSort(local_values, local_indices, GTOp<scalar_t, true>(), inputSliceSize, invalid_value);
  } else {
    Sort(warp_storage.sort).StableSort(local_values, local_indices, LTOp<scalar_t, true>(), inputSliceSize, invalid_value);
  }
  WARP_SYNC();

  if constexpr (sort_size <= DEFAULT_MAX_WARP_TOPK_SLICE) {
    // Store top-k results
    StoreKeys(warp_storage.store_keys).Store(topK_iter, local_values, k);
    WARP_SYNC();
    StoreIndices(warp_storage.store_indices).Store(indices_iter, local_indices, k);
  } else {
    // storeTopKInGatherOrder's barriers run a slice-dependent number of times,
    // which is safe only with one warp per block.
    static_assert(max_block_dim_y == 1, "the gather-order store needs one warp per block");
    // Broadcast the k-th value of the stable sort (blocked layout: thread t
    // holds sorted positions [t * items_per_thread, (t + 1) * items_per_thread)).
    #pragma unroll
    for (int i = 0; i < items_per_thread; ++i) {
      if (local_indices[i] < inputSliceSize && static_cast<IndexType>(threadIdx.x * items_per_thread + i) == k - 1) {
        kth_value[threadIdx.y] = local_values[i];
      }
    }
    // WARP_SYNC() is a no-op on ROCm; the block is a single warp, so a block
    // barrier orders the LDS broadcast.
    __syncthreads();
    storeTopKInGatherOrder<scalar_t, IndexType, is_descending, WARP_SIZE, Scan>(
        warp_storage.scan, kth_value[threadIdx.y], input_iter, inputSliceSize, k, topK_iter, indices_iter);
  }
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
  // See Note [warp merge sort WARP_SIZE template param]
#ifdef USE_ROCM
  #define LAUNCH_KERNEL(SORT_SIZE, IS_DESCENDING) \
  if (at::cuda::warp_size() == 32) { \
    warpMergeSortTopK<Dim, Dim, SORT_SIZE, max_block_dim_y, scalar_t, IndexType, IS_DESCENDING, 32> \
      <<<grid, block, 0, stream>>>( \
          input, inputSliceSize, k, numInputSlices, inputWithinSliceStride, \
          topK, topKWithinSliceStride, indices, indicesWithinSliceStride); \
  } \
  else { \
    warpMergeSortTopK<Dim, Dim, SORT_SIZE, max_block_dim_y, scalar_t, IndexType, IS_DESCENDING, 64> \
      <<<grid, block, 0, stream>>>( \
          input, inputSliceSize, k, numInputSlices, inputWithinSliceStride, \
          topK, topKWithinSliceStride, indices, indicesWithinSliceStride); \
  } \
  C10_CUDA_KERNEL_LAUNCH_CHECK()
#else
  #define LAUNCH_KERNEL(SORT_SIZE, IS_DESCENDING) \
    warpMergeSortTopK<Dim, Dim, SORT_SIZE, max_block_dim_y, scalar_t, IndexType, IS_DESCENDING, 32> \
      <<<grid, block, 0, stream>>>( \
          input, inputSliceSize, k, numInputSlices, inputWithinSliceStride, \
          topK, topKWithinSliceStride, indices, indicesWithinSliceStride); \
    C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif

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

// Top-k of many short slices with a small k on gfx1250: one wave per slice,
// each lane keeps the best K of the elements it reads in registers, and the
// wave merges the lanes' lists with shuffles. It reads every element once and
// needs no LDS, no atomics and no block barriers, which bound sbtopk, mbtopk
// and the warp sort on these shapes.
//
// It selects exactly what the path it replaces selects. Elements are ranked
// by key, then by lower index. Slices up to warptopk::DEFAULT_MAX_WARP_TOPK_SLICE
// replace the warp sort: keys follow GTOp/LTOp (the radix key of the value
// with -0 read as +0, so NaNs compare equal and above everything), and the
// output is sorted. Longer slices replace sbtopk/mbtopk: keys are the radix
// keys, and the output is in their gather order (the elements ranked strictly
// above the k-th, by index, then the ones equal to it, by index).
namespace waveselect {

// gfx1250 wave size; the path runs on gfx1250 only. A constant rather than
// C10_WARP_SIZE, which is not constexpr under __SPIRV__: the host routes only
// gfx1250 (wave32) devices here, SPIR-V code objects included.
constexpr int kLanes = 32;
// Slices per 256-thread block, one per wave.
constexpr int kSlicesPerBlock = 8;
// Largest k: the merge gives each of the first k lanes one result, and the
// per-lane lists are instantiated for K = 2, 4, 8 and 16.
constexpr int kMaxK = 16;

// Compares with == like GTOp/LTOp, so it also maps the values that the
// build's float compares flush to zero.
template <typename scalar_t>
__device__ __forceinline__ scalar_t zeroAsPositive(scalar_t v) {
  if constexpr (std::is_floating_point_v<scalar_t> || std::is_same_v<scalar_t, at::Half> ||
                std::is_same_v<scalar_t, at::BFloat16>) {
    return v == scalar_t(0) ? scalar_t(0) : v;
  } else {
    return v;
  }
}

template <typename Key>
__device__ __forceinline__ bool ranksAbove(Key key, uint32_t idx, Key other_key, uint32_t other_idx) {
  return key > other_key || (key == other_key && idx < other_idx);
}

template <typename T, typename IndexType, int K, bool kSortedOutput>
C10_LAUNCH_BOUNDS_1(kLanes * kSlicesPerBlock)
__global__ void waveSelectTopK(
    at::cuda::detail::TensorInfo<const T, IndexType> input,
    IndexType inputSliceSize,
    IndexType k,
    bool largest,
    IndexType numInputSlices,
    IndexType inputWithinSliceStride,
    at::cuda::detail::TensorInfo<T, IndexType> topK,
    IndexType topKWithinSliceStride,
    at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
    IndexType indicesWithinSliceStride) {
#if defined(__gfx1250__) || defined(__gfx1250_strict__) || defined(__SPIRV__)
  using Bitwise = typename TopKTypeConfig<T>::RadixType;
  // Shuffles move 32- or 64-bit values.
  using Key = std::conditional_t<sizeof(Bitwise) <= sizeof(uint32_t), uint32_t, uint64_t>;
  constexpr uint32_t kNoIndex = std::numeric_limits<uint32_t>::max();

  const IndexType slice = getLinearBlockId<IndexType>() * kSlicesPerBlock + threadIdx.x / kLanes;
  if (slice >= numInputSlices) {
    return;
  }
  const uint32_t lane = threadIdx.x % kLanes;
  const uint32_t slice_size = static_cast<uint32_t>(inputSliceSize);
  const T* in = &input.data[at::cuda::detail::IndexToOffset<const T, IndexType, -1>::get(slice, input)];

  // Larger key ranks higher in both directions.
  auto rank_key = [&](T v) -> Key {
    const Key key = TopKTypeConfig<T>::convert(kSortedOutput ? zeroAsPositive(v) : v);
    return largest ? key : static_cast<Key>(static_cast<Bitwise>(~key));
  };

  // Per-lane best K, best first. (0, kNoIndex) ranks below every element.
  Key keys[K];
  uint32_t idxs[K];
#pragma unroll
  for (int j = 0; j < K; ++j) {
    keys[j] = 0;
    idxs[j] = kNoIndex;
  }
  auto insert = [&](Key key, uint32_t idx) {
    // A lane reads increasing indices, so an equal key ranks below the list.
    if (ranksAbove(key, idx, keys[K - 1], idxs[K - 1])) {
      keys[K - 1] = key;
      idxs[K - 1] = idx;
#pragma unroll
      for (int j = K - 1; j > 0; --j) {
        if (ranksAbove(keys[j], idxs[j], keys[j - 1], idxs[j - 1])) {
          const Key tk = keys[j];
          keys[j] = keys[j - 1];
          keys[j - 1] = tk;
          const uint32_t ti = idxs[j];
          idxs[j] = idxs[j - 1];
          idxs[j - 1] = ti;
        }
      }
    }
  };

  constexpr uint32_t kBatch = 4;
  uint32_t i = lane;
  for (; i + (kBatch - 1) * kLanes < slice_size; i += kBatch * kLanes) {
    T v[kBatch];
#pragma unroll
    for (uint32_t u = 0; u < kBatch; ++u) {
      v[u] = doLdg(&in[static_cast<IndexType>(i + u * kLanes) * inputWithinSliceStride]);
    }
#pragma unroll
    for (uint32_t u = 0; u < kBatch; ++u) {
      insert(rank_key(v[u]), i + u * kLanes);
    }
  }
  for (; i < slice_size; i += kLanes) {
    insert(rank_key(doLdg(&in[static_cast<IndexType>(i) * inputWithinSliceStride])), i);
  }

  // Merge: round r moves the wave's best remaining element to lane r.
  const uint32_t num_out = static_cast<uint32_t>(k);
  Key out_key = 0;
  uint32_t out_idx = kNoIndex;
  for (uint32_t r = 0; r < num_out; ++r) {
    Key best_key = keys[0];
    uint32_t best_idx = idxs[0];
#pragma unroll
    for (int offset = kLanes / 2; offset > 0; offset /= 2) {
      const Key other_key = WARP_SHFL_XOR(best_key, offset);
      const uint32_t other_idx = WARP_SHFL_XOR(best_idx, offset);
      if (ranksAbove(other_key, other_idx, best_key, best_idx)) {
        best_key = other_key;
        best_idx = other_idx;
      }
    }
    if (lane == r) {
      out_key = best_key;
      out_idx = best_idx;
    }
    // Indices are unique, so exactly one lane owns the best element.
    if (idxs[0] == best_idx) {
#pragma unroll
      for (int j = 0; j < K - 1; ++j) {
        keys[j] = keys[j + 1];
        idxs[j] = idxs[j + 1];
      }
      keys[K - 1] = 0;
      idxs[K - 1] = kNoIndex;
    }
  }

  // Output position of lane r's element.
  uint32_t pos = lane;
  if constexpr (!kSortedOutput) {
    const Key kth_key = WARP_SHFL(out_key, num_out - 1);
    const bool tie = out_key == kth_key;
    uint32_t above = 0;
    uint32_t before = 0;
    for (uint32_t j = 0; j < num_out; ++j) {
      const Key other_key = WARP_SHFL(out_key, j);
      const uint32_t other_idx = WARP_SHFL(out_idx, j);
      const bool other_tie = other_key == kth_key;
      above += other_tie ? 0 : 1;
      before += (other_tie == tie && other_idx < out_idx) ? 1 : 0;
    }
    pos = (tie ? above : 0) + before;
  }
  if (lane < num_out) {
    T* out_values = &topK.data[at::cuda::detail::IndexToOffset<T, IndexType, -1>::get(slice, topK)];
    int64_t* out_indices =
        &indices.data[at::cuda::detail::IndexToOffset<int64_t, IndexType, -1>::get(slice, indices)];
    out_values[static_cast<IndexType>(pos) * topKWithinSliceStride] =
        doLdg(&in[static_cast<IndexType>(out_idx) * inputWithinSliceStride]);
    out_indices[static_cast<IndexType>(pos) * indicesWithinSliceStride] = out_idx;
  }
#else
  CUDA_KERNEL_ASSERT(false && "waveSelectTopK is gfx1250-only");
#endif
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
  TORCH_INTERNAL_ASSERT(k > 0 && k <= kMaxK, "wave select topk requires 0 < k <= ", kMaxK);
  dim3 grid;
  TORCH_INTERNAL_ASSERT(getGridFromTiles(at::ceil_div(numInputSlices, static_cast<IndexType>(kSlicesPerBlock)), grid),
                        "Too many slices for wave select topk");
  dim3 block(kLanes * kSlicesPerBlock);
  const auto stream = c10::cuda::getCurrentCUDAStream();
  // Slices the warp sort handles everywhere get its semantics and sorted output.
  const bool sorted_output = inputSliceSize <= warptopk::DEFAULT_MAX_WARP_TOPK_SLICE;
#define LAUNCH_WAVE_SELECT(K, SORTED)                                                   \
  waveSelectTopK<scalar_t, IndexType, K, SORTED><<<grid, block, 0, stream>>>(           \
      input, inputSliceSize, k, largest, numInputSlices, inputWithinSliceStride,        \
      topK, topKWithinSliceStride, indices, indicesWithinSliceStride)
#define LAUNCH_WAVE_SELECT_K(SORTED) \
  if (k <= 2) {                      \
    LAUNCH_WAVE_SELECT(2, SORTED);   \
  } else if (k <= 4) {               \
    LAUNCH_WAVE_SELECT(4, SORTED);   \
  } else if (k <= 8) {               \
    LAUNCH_WAVE_SELECT(8, SORTED);   \
  } else {                           \
    LAUNCH_WAVE_SELECT(16, SORTED);  \
  }
  if (sorted_output) {
    LAUNCH_WAVE_SELECT_K(true);
  } else {
    LAUNCH_WAVE_SELECT_K(false);
  }
#undef LAUNCH_WAVE_SELECT_K
#undef LAUNCH_WAVE_SELECT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace waveselect
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

#if defined(USE_ROCM)
// computeBlockDigitCountsLanePrivate, used on gfx1250 only. The values come
// from a sweep on gfx1250 of 4, 8, 16 and 32 copies, 4, 8 and 16 loads in
// flight, and 4, 8 and 16 minimum items per thread.
//
// One histogram copy per lane of a wave32 wave, so no two lanes of a wave
// increment the same LDS counter (fastest of the four copy counts).
constexpr int kLanePrivateCopies = 32;
// Global loads in flight per thread; 4, 8 and 16 are within 2% of each other.
constexpr int kLanePrivateLoadBatch = 8;
// Fewest items counted per thread that take the lane-private counts. 4, 8 and
// 16 time the same on long slices, and 4 is also faster on many slices of
// 4 to 7 * BLOCK_THREADS elements.
constexpr int kLanePrivateMinItemsPerThread = 4;

// computeBlockDigitCounts with kLanePrivateCopies copies of the histogram, so
// that the lanes of a wave, whose keys share their high bits, do not serialize
// on the same LDS counters, and with kLanePrivateLoadBatch global loads in
// flight per thread.
template <typename T, typename IndexType, typename Bitwise, int Dim>
C10_LAUNCH_BOUNDS_1(BLOCK_THREADS)
__global__ void computeBlockDigitCountsLanePrivate(
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
#if defined(__gfx1250__) || defined(__gfx1250_strict__) || defined(__SPIRV__)
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
  // Interleaved histogram copies, [digit][copy]; thread i updates copy
  // i % kLanePrivateCopies.
  __shared__ uint32_t digit_counters[RADIX_DIGITS * kLanePrivateCopies];
  for (int i = tidx; i < RADIX_DIGITS * kLanePrivateCopies; i += BLOCK_THREADS) {
    digit_counters[i] = 0;
  }
  __syncthreads();

  items_per_thread = (blk_idx_in_slice + 1 < blocks_per_slice)
      ? items_per_thread
      : at::ceil_div((int64_t)(slice_size - blk_idx_in_slice * items_per_block), (int64_t)BLOCK_THREADS);

  const int copy = tidx % kLanePrivateCopies;
  for (int i0 = 0; i0 < items_per_thread; i0 += kLanePrivateLoadBatch) {
    T vals[kLanePrivateLoadBatch];
    bool valid[kLanePrivateLoadBatch];
#pragma unroll
    for (int u = 0; u < kLanePrivateLoadBatch; ++u) {
      IndexType idx = blk_idx_in_slice * items_per_block + (i0 + u) * BLOCK_THREADS + tidx;
      valid[u] = (i0 + u < items_per_thread) && idx < slice_size;
      if (valid[u]) {
        vals[u] = doLdg(&data[idx * withinSliceStride]);
      }
    }
#pragma unroll
    for (int u = 0; u < kLanePrivateLoadBatch; ++u) {
      if (valid[u]) {
        Bitwise val = TopKTypeConfig<T>::convert(vals[u]);
        if ((val & desiredMask) == (desired & desiredMask)) {
          Bitwise digit = at::cuda::Bitfield<Bitwise>::getBitfield(val, current_bit, RADIX_BITS);
          atomicAdd(&digit_counters[digit * kLanePrivateCopies + copy], 1);
        }
      }
    }
  }
  __syncthreads();

  if (tidx < RADIX_DIGITS) {
    uint32_t digit_count = 0;
    // Rotate the start copy per thread so the reads spread over LDS banks.
    for (int j = 0; j < kLanePrivateCopies; ++j) {
      digit_count += digit_counters[tidx * kLanePrivateCopies + (j + tidx) % kLanePrivateCopies];
    }
    counts[block_idx * RADIX_DIGITS + tidx] = digit_count;
  }
#else
  CUDA_KERNEL_ASSERT(false && "computeBlockDigitCountsLanePrivate is gfx1250-only");
#endif
}
#endif

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

  constexpr int SHMEM_SIZE = RADIX_DIGITS / C10_WARP_SIZE_LOWER_BOUND;  // max shmem size on ROCm
  const int num_warps = RADIX_DIGITS / C10_WARP_SIZE;
  __shared__ uint32_t warp_counts[SHMEM_SIZE];
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

  constexpr int SHMEM_SIZE = RADIX_DIGITS / C10_WARP_SIZE_LOWER_BOUND; // max shmem size on ROCm
  const int num_warps = RADIX_DIGITS / C10_WARP_SIZE;
  __shared__ uint32_t warp_counts[SHMEM_SIZE];
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

// computeBlockwiseWithinKCounts with one block per slice instead of one per
// counting block, used on gfx1250. The per-block kernel sums the digit counts
// of all blocks of the slice in every block, blocks_per_slice times the same
// work, and on gfx1250 its many small blocks are bound by their own latency
// (one BlockScan and its barriers per block). Here each block sums its slice's
// counts once, picks the digit, and its waves then compute the within-k count
// (and on the last pass the k-th value count) of the slice's blocks in turn.
// The results are the same integers.
template <typename Bitwise, typename T>
C10_LAUNCH_BOUNDS_1(RADIX_DIGITS)  // one thread per digit
__global__ void computeSliceWithinKCounts(
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
  uint32_t* kthCounts,
  uint32_t num_slices
) {
#if defined(__gfx1250__) || defined(__gfx1250_strict__) || defined(__SPIRV__)
  const int tidx = threadIdx.x;
  const uint32_t slice_idx = getLinearBlockId<uint32_t>();
  if (slice_idx >= num_slices) {
    return;
  }
  const uint32_t k_to_find = ks_to_find_in[slice_idx];
  const short* slice_counts = counts + static_cast<size_t>(slice_idx) * blocks_per_slice * RADIX_DIGITS;

  typedef ROCM_HIPCUB(at_cuda_detail::cub)::BlockScan<uint32_t, RADIX_DIGITS> BlockScan;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ uint32_t digit_totals[RADIX_DIGITS];
  __shared__ Bitwise desired;

  uint32_t sum = 0;
  for (uint32_t blk = 0; blk < blocks_per_slice; ++blk) {
    sum += slice_counts[blk * RADIX_DIGITS + tidx];
  }
  uint32_t digit_count_cumsum;
  BlockScan(scan_storage).InclusiveSum(sum, digit_count_cumsum);
  digit_totals[tidx] = digit_count_cumsum;
  __syncthreads();
  const uint32_t digit_count_cumsum_left = (tidx == 0) ? 0 : digit_totals[tidx - 1];
  if (digit_count_cumsum_left < k_to_find && k_to_find <= digit_count_cumsum) {
    Bitwise d = at::cuda::Bitfield<Bitwise>::setBitfield(desires_in[slice_idx], tidx, current_bit, RADIX_BITS);
    desired = d;
    desires_out[slice_idx] = d;
    if (current_bit > 0) {
      ks_to_find_out[slice_idx] = k_to_find - digit_count_cumsum_left;
    } else {
      kthValues[slice_idx] = TopKTypeConfig<T>::deconvert(d);
    }
  }
  __syncthreads();

  const int desired_digit = at::cuda::Bitfield<Bitwise>::getBitfield(desired, current_bit, RADIX_BITS);
  const int desired_digit0 = at::cuda::Bitfield<Bitwise>::getBitfield(desired, 0, RADIX_BITS);
  const int lane = tidx % C10_WARP_SIZE;
  const int warp = tidx / C10_WARP_SIZE;
  const int num_warps = RADIX_DIGITS / C10_WARP_SIZE;
  for (uint32_t blk = warp; blk < blocks_per_slice; blk += num_warps) {
    const short* block_counts = slice_counts + blk * RADIX_DIGITS;
    uint32_t count = 0;
    for (int digit = lane; digit < RADIX_DIGITS; digit += C10_WARP_SIZE) {
      if (largest ? digit > desired_digit : digit < desired_digit) {
        count += block_counts[digit];
      }
    }
    for (int offset = C10_WARP_SIZE / 2; offset > 0; offset /= 2) {
      count += WARP_SHFL_DOWN(count, offset);
    }
    if (lane == 0) {
      const uint32_t block_idx = slice_idx * blocks_per_slice + blk;
      withinKCounts[block_idx] += count;
      if (current_bit == 0) {
        kthCounts[block_idx] = block_counts[desired_digit0];
      }
    }
  }
#else
  CUDA_KERNEL_ASSERT(false && "computeSliceWithinKCounts is gfx1250-only");
#endif
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

#if defined(USE_ROCM)
// gatherTopK for gfx1250: places each item at the same offset (its rank in
// index order plus the per-block bases), but ranks kWaveBallotRows rows of
// BLOCK_THREADS items per barrier instead of one block-wide scan per row. Each
// wave counts its items of each row with ballots, one barrier exchanges the
// per-wave counts, and every wave scans them to find where its items go. The
// earlier blocks' counts are summed by all threads.
//
// Rows loaded and ranked per barrier, the same batch as the lane-private digit
// counts, which load one item per lane per row too. Swept on gfx1250 with 4, 8
// and 16 rows: 4 and 8 are within 2% of each other, 16 is slower.
constexpr int kWaveBallotRows = kLanePrivateLoadBatch;
// A wave's counts of one row packed in one word: within-k items in the low half,
// k-th value items in the high half. Sums over an exchange must fit a half.
constexpr int kCountBits = std::numeric_limits<uint32_t>::digits / 2;
constexpr uint32_t kCountMask = (1u << kCountBits) - 1;
static_assert(kWaveBallotRows * BLOCK_THREADS <= kCountMask, "the counts of an exchange must fit in a half word");

template <typename T, typename IndexType, int Dim>
C10_LAUNCH_BOUNDS_1(BLOCK_THREADS)
__global__ void gatherTopKWaveBallot(at::cuda::detail::TensorInfo<const T, IndexType> input,
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
#if defined(__gfx1250__) || defined(__gfx1250_strict__) || defined(__SPIRV__)
  constexpr int kMaxWaves = BLOCK_THREADS / C10_WARP_SIZE_LOWER_BOUND;
  // (row, wave) counts per exchange, and how many of them each lane scans.
  constexpr int kEntries = kWaveBallotRows * kMaxWaves;
  constexpr int kEntriesPerLane = (kEntries + C10_WARP_SIZE_LOWER_BOUND - 1) / C10_WARP_SIZE_LOWER_BOUND;
  const uint32_t block_idx = getLinearBlockId<uint32_t>();
  if (block_idx >= num_blocks) {
    return;
  }
  const uint32_t tidx = threadIdx.x;
  const uint32_t wave = tidx / C10_WARP_SIZE;
  const uint32_t lane = tidx % C10_WARP_SIZE;
  const uint32_t num_waves = BLOCK_THREADS / C10_WARP_SIZE;
  const uint32_t slice_idx = block_idx / blocks_per_slice;
  const uint32_t blk_idx_in_slice = block_idx % blocks_per_slice;
  const uint32_t items_per_block = items_per_thread * BLOCK_THREADS;

  // Within-k and k-th value counts of the earlier blocks of the slice, and the
  // slice's total within-k count, summed per wave here and over waves below.
  const uint32_t slice_offset = slice_idx * blocks_per_slice;
  uint32_t earlier_within = 0;
  uint32_t earlier_kth = 0;
  uint32_t slice_within = 0;
  for (uint32_t blk = tidx; blk < blocks_per_slice; blk += BLOCK_THREADS) {
    const uint32_t within = withinKCounts[slice_offset + blk];
    slice_within += within;
    if (blk < blk_idx_in_slice) {
      earlier_within += within;
      earlier_kth += kthCounts[slice_offset + blk];
    }
  }
  for (int offset = C10_WARP_SIZE / 2; offset > 0; offset /= 2) {
    earlier_within += WARP_SHFL_DOWN(earlier_within, offset);
    earlier_kth += WARP_SHFL_DOWN(earlier_kth, offset);
    slice_within += WARP_SHFL_DOWN(slice_within, offset);
  }
  __shared__ uint32_t earlier_within_counts[kMaxWaves];
  __shared__ uint32_t earlier_kth_counts[kMaxWaves];
  __shared__ uint32_t slice_within_counts[kMaxWaves];
  if (lane == 0) {
    earlier_within_counts[wave] = earlier_within;
    earlier_kth_counts[wave] = earlier_kth;
    slice_within_counts[wave] = slice_within;
  }

  items_per_thread = (blk_idx_in_slice + 1 < blocks_per_slice)
      ? items_per_thread
      : at::ceil_div((int64_t)(inputSliceSize - blk_idx_in_slice * items_per_block), (int64_t)BLOCK_THREADS);

  const T* inputSliceStart =
      &input.data[at::cuda::detail::IndexToOffset<const T, IndexType, Dim>::get(slice_idx, input)];
  T* topKSliceStart = &topK.data[at::cuda::detail::IndexToOffset<T, IndexType, Dim>::get(slice_idx, topK)];
  int64_t* indicesSliceStart =
      &indices.data[at::cuda::detail::IndexToOffset<int64_t, IndexType, Dim>::get(slice_idx, indices)];
  const auto kthValueConverted = at::native::TopKTypeConfig<T>::convert(kthValues[slice_idx]);
  const auto lanes_before = at::cuda::getLaneMaskLt();

  // Double-buffered so that one barrier per exchange suffices.
  __shared__ uint32_t row_counts[2][kEntries];
  uint32_t start_within = 0;
  uint32_t start_kth = 0;
  for (uint32_t row0 = 0, exchange = 0; row0 < items_per_thread; row0 += kWaveBallotRows, ++exchange) {
    uint32_t* counts = row_counts[exchange % 2];
    // Zero-initialized: the ballots below also convert the invalid slots.
    T vals[kWaveBallotRows]{};
    bool valid[kWaveBallotRows];
#pragma unroll
    for (int u = 0; u < kWaveBallotRows; ++u) {
      const IndexType idx = blk_idx_in_slice * items_per_block + (row0 + u) * BLOCK_THREADS + tidx;
      valid[u] = row0 + u < items_per_thread && idx < inputSliceSize;
      if (valid[u]) {
        vals[u] = doLdg(inputSliceStart + idx * inputWithinSliceStride);
      }
    }
    bool within[kWaveBallotRows];
    bool kth[kWaveBallotRows];
    decltype(WARP_BALLOT(0)) within_mask[kWaveBallotRows];
    decltype(WARP_BALLOT(0)) kth_mask[kWaveBallotRows];
#pragma unroll
    for (int u = 0; u < kWaveBallotRows; ++u) {
      const auto valConverted = at::native::TopKTypeConfig<T>::convert(vals[u]);
      within[u] = valid[u] && (largest ? valConverted > kthValueConverted : valConverted < kthValueConverted);
      kth[u] = valid[u] && valConverted == kthValueConverted;
      within_mask[u] = WARP_BALLOT(within[u]);
      kth_mask[u] = WARP_BALLOT(kth[u]);
      if (lane == 0) {
        counts[u * num_waves + wave] = __popcll(within_mask[u]) | (__popcll(kth_mask[u]) << kCountBits);
      }
    }
    __syncthreads();

    if (exchange == 0) {
      for (uint32_t w = 0; w < num_waves; ++w) {
        start_within += earlier_within_counts[w];
        start_kth += slice_within_counts[w] + earlier_kth_counts[w];
      }
    }

    // Exclusive scan of the (row, wave) counts in index order: entry u * num_waves + w
    // counts the items of wave w in row row0 + u.
    uint32_t entry_before[kEntriesPerLane];
    uint32_t lane_sum = 0;
#pragma unroll
    for (int p = 0; p < kEntriesPerLane; ++p) {
      const uint32_t e = lane * kEntriesPerLane + p;
      entry_before[p] = lane_sum;
      lane_sum += e < kWaveBallotRows * num_waves ? counts[e] : 0;
    }
    uint32_t lane_inclusive = lane_sum;
    for (int offset = 1; offset < C10_WARP_SIZE; offset *= 2) {
      const uint32_t v = WARP_SHFL_UP(lane_inclusive, offset);
      if (lane >= (uint32_t)offset) {
        lane_inclusive += v;
      }
    }
#pragma unroll
    for (int p = 0; p < kEntriesPerLane; ++p) {
      entry_before[p] += lane_inclusive - lane_sum;
    }
    const uint32_t exchange_total = WARP_SHFL(lane_inclusive, C10_WARP_SIZE - 1);

#pragma unroll
    for (int u = 0; u < kWaveBallotRows; ++u) {
      const uint32_t e = u * num_waves + wave;
      uint32_t before = entry_before[0];
#pragma unroll
      for (int p = 1; p < kEntriesPerLane; ++p) {
        if (e % kEntriesPerLane == (uint32_t)p) {
          before = entry_before[p];
        }
      }
      before = WARP_SHFL(before, e / kEntriesPerLane);
      const IndexType idx = blk_idx_in_slice * items_per_block + (row0 + u) * BLOCK_THREADS + tidx;
      if (within[u]) {
        const uint32_t offset = start_within + (before & kCountMask) + __popcll(within_mask[u] & lanes_before);
        topKSliceStart[offset * topKWithinSliceStride] = vals[u];
        indicesSliceStart[offset * indicesWithinSliceStride] = idx;
      }
      if (kth[u]) {
        const uint32_t offset = start_kth + (before >> kCountBits) + __popcll(kth_mask[u] & lanes_before);
        if (offset < outputSliceSize) {
          topKSliceStart[offset * topKWithinSliceStride] = vals[u];
          indicesSliceStart[offset * indicesWithinSliceStride] = idx;
        }
      }
    }
    start_within += exchange_total & kCountMask;
    start_kth += exchange_total >> kCountBits;
  }
#else
  CUDA_KERNEL_ASSERT(false && "gatherTopKWaveBallot is gfx1250-only");
#endif
}
#endif

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

#if defined(USE_ROCM)
// Fewest slices that take computeSliceWithinKCounts: with fewer, the per-slice
// kernel runs too few blocks to be faster.
constexpr uint32_t kSliceWithinKCountsMinSlices = 65;

static inline bool can_use_lane_private_digit_counts(c10::DeviceIndex device_index, int items_per_thread, int64_t slice_size) {
  if (!is_gfx1250_device(device_index)) {
    return false;
  }
  const int64_t counted_per_thread = std::min<int64_t>(items_per_thread, at::ceil_div(slice_size, (int64_t)BLOCK_THREADS));
  return counted_per_thread >= kLanePrivateMinItemsPerThread;
}
#endif

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
#if defined(USE_ROCM)
  // The gfx1250 kernels compile their bodies for gfx1250 and SPIR-V only
  // (__gfx1250__, __gfx1250_strict__ or __SPIRV__); elsewhere they assert.
  const bool lane_private_digit_counts =
      can_use_lane_private_digit_counts(stream.device_index(), items_per_thread, inputSliceSize);
  const bool wave_ballot_gather = is_gfx1250_device(stream.device_index());
  const bool slice_within_k_counts = wave_ballot_gather && numInputSlices >= kSliceWithinKCountsMinSlices;
#endif

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
#if defined(USE_ROCM)
  dim3 slice_grid;
  TORCH_INTERNAL_ASSERT(getGridFromTiles(numInputSlices, slice_grid), "Too many slices for topk");
#endif

  uint32_t * ks_to_find_in = ks_to_find;
  uint32_t * ks_to_find_out = ks_to_find + numInputSlices;
  Bitwise * desired_in = desired;
  Bitwise * desired_out = desired + numInputSlices;

  // iterate radix bits for multiple passes
  for (int current_bit = sizeof(T) * 8 - RADIX_BITS; current_bit >= 0; current_bit -= RADIX_BITS) {
#if defined(USE_ROCM)
    if (lane_private_digit_counts) {
      computeBlockDigitCountsLanePrivate<T, IndexType, Bitwise, Dim><<<grid, block, 0, stream>>>(
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
    } else
#endif
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
    if (slice_within_k_counts) {
      computeSliceWithinKCounts<Bitwise, T><<<slice_grid, RADIX_DIGITS, 0, stream>>>(
        desired_in, counts, ks_to_find_in, blocks_per_slice, current_bit, largest,
        withinKCounts, kthValues, ks_to_find_out, desired_out, kthCounts, numInputSlices);
    } else {
      computeBlockwiseWithinKCounts<Bitwise, T><<<grid, RADIX_DIGITS, 0, stream>>>(
        desired_in, counts, ks_to_find_in, blocks_per_slice, current_bit, largest,
        withinKCounts, kthValues, ks_to_find_out, desired_out, kthCounts, num_blocks);
    }
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
  using counting_iter_t = ATEN_CUB_COUNTING_ITERATOR(uint32_t);
  using slice_idx_iter_t = ATEN_CUB_TRANSFORM_ITERATOR(uint32_t, BlockIdxToKey, counting_iter_t);
  slice_idx_iter_t slice_idx_iter(counting_iter_t(0), BlockIdxToKey(blocks_per_slice));
  at::cuda::cub::inclusive_sum_by_key(slice_idx_iter, withinKCounts, withinKCounts, num_blocks);
  at::cuda::cub::inclusive_sum_by_key(slice_idx_iter, kthCounts, kthCounts, num_blocks);
#else
  // ROCm path: kthCounts already produced in computeBlockwiseWithinKCounts at last pass
  // No CUB scans; prefix computation fused into gatherTopK
#endif

  // copy topk values to output tensor
#if defined(USE_ROCM)
  if (wave_ballot_gather) {
    gatherTopKWaveBallot<T, IndexType, Dim><<<grid, block, 0, stream>>>(
      input, inputSliceSize, outputSliceSize, largest, numInputSlices, inputWithinSliceStride,
      topK, topKWithinSliceStride, indices, indicesWithinSliceStride, items_per_thread,
      blocks_per_slice, kthValues, withinKCounts, kthCounts, num_blocks);
  } else
#endif
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

bool should_use_warp_topk(c10::DeviceIndex device_index, int64_t num_slices, int64_t slice_size, int64_t k) {
#if !defined(USE_ROCM) || !HAS_WARP_MERGE_SORT()
  return false;
#else
  if (slice_size <= 0 || k <= 0 || slice_size > warptopk::MAX_WARP_TOPK_SLICE) {
    return false;
  }
  // On gfx1250 the multi-block radix select (mbtopk) is bound by same-address
  // LDS atomics, and the warp sort is faster than it for every slice up to
  // MAX_WARP_TOPK_SLICE, but not faster than the single-block sbtopk. So there
  // the warp sort replaces mbtopk only. The warpMergeSortTopK instantiations
  // above DEFAULT_MAX_WARP_TOPK_SLICE compile their bodies for gfx1250 and
  // SPIR-V only; elsewhere they assert.
  if (is_gfx1250_device(device_index) && should_use_multiblock(num_slices, slice_size)) {
    return true;
  }
  // Use WarpMergeSort for small slices
  // WarpMergeSort has O(n log n) complexity and is efficient for small sizes
  // Conservative threshold: slice_size <= 256 shows consistent improvements
  return slice_size <= warptopk::DEFAULT_MAX_WARP_TOPK_SLICE;
#endif
}

#if defined(USE_ROCM) && HAS_WARP_MERGE_SORT()
// One wave reads each slice, so the wave select needs many slices to fill the
// GPU, and its time per wave grows with the slice length.
constexpr int64_t kWaveSelectMinSlices = 512;
constexpr int64_t kWaveSelectMaxSlice = 4096;

// Must agree with the device gate of waveselect::waveSelectTopK (__gfx1250__,
// __gfx1250_strict__ or __SPIRV__); a mismatch asserts.
bool should_use_wave_select(c10::DeviceIndex device_index, int64_t num_slices, int64_t slice_size, int64_t k) {
  return k > 0 && k <= waveselect::kMaxK && slice_size >= k && slice_size <= kWaveSelectMaxSlice &&
      num_slices >= kWaveSelectMinSlices && is_gfx1250_device(device_index);
}
#endif

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
  if (should_use_wave_select(self.device().index(), numInputSlices, sliceSize, k)) { \
    RUN_K(INDEX_T, DIM, waveselect::launch);                              \
  } else if (should_use_warp_topk(self.device().index(), numInputSlices, sliceSize, k)) { \
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
