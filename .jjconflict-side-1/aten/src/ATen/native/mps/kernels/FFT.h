#pragma once
#include <c10/metal/common.h>

// Kernel variant bits; their sum ends the kernel name
C10_METAL_CONSTEXPR uint32_t kFFTInverse = 1u << 0;
C10_METAL_CONSTEXPR uint32_t kFFTFourStepTwiddles = 1u << 2;
C10_METAL_CONSTEXPR uint32_t kFFTExtendedStores = 1u << 3;
C10_METAL_CONSTEXPR uint32_t kFFTExtendedLoads = 1u << 4;
C10_METAL_CONSTEXPR uint32_t kFFTGlobalStage = 1u << 5;
// The bits of four-step, chirp and global-memory passes
C10_METAL_CONSTEXPR uint32_t kFFTMultiPass = kFFTFourStepTwiddles |
    kFFTExtendedStores | kFFTExtendedLoads | kFFTGlobalStage;

// Metal's threadgroup memory limit, and the complex points that fit in it
C10_METAL_CONSTEXPR uint32_t kFFTSmemBytes = 32768;
C10_METAL_CONSTEXPR uint32_t kFFTMaxPoints =
    kFFTSmemBytes / (2 * sizeof(float));
// Stages of one plan: radix 2 over 4096 points takes 12, the global radices
// below factor any length under 2^31 in fewer
C10_METAL_CONSTEXPR uint32_t kFFTMaxStages = 16;

// Threads per group: the host never dispatches more, and the kernels are
// compiled to fit it (register-heavy variants otherwise allow as few as 384 on
// M1 and M2, and a larger group silently does nothing)
C10_METAL_CONSTEXPR uint32_t kFFTMaxThreads = 512;

// 8 points per thread; 16 only pay for plain 1024-point kernels (one stage
// fewer at full occupancy)
inline constexpr uint32_t fft_points_per_thread(uint32_t n, uint32_t flags) {
  return n == 1024 && !(flags & kFFTMultiPass) ? 16 : 8;
}

// Spare smem slots: one every 1 << kFFTPadShift slots, none with kFFTNoPadShift
C10_METAL_CONSTEXPR uint32_t kFFTPadShift = 4;
C10_METAL_CONSTEXPR uint32_t kFFTNoPadShift = 31;

// A group holding one contiguous transform pads if the padded row fits smem
inline constexpr uint32_t fft_contig_pad_shift(uint32_t n) {
  return n + (n >> kFFTPadShift) <= kFFTMaxPoints ? kFFTPadShift
                                                  : kFFTNoPadShift;
}

// Two-level twiddle table: W^b for b < kFFTTwiddleBlock, then
// W^(kFFTTwiddleBlock a)
C10_METAL_CONSTEXPR uint32_t kFFTTwiddleBlock = 4096;

// Global-memory stage radices, largest first: lengths factor in this order
#define FFT_GLOBAL_RADICES(_) \
  _(28) _(25) _(21) _(16) _(15) _(12) _(9) _(8) _(7) _(5) _(4) _(3) _(2)

#define FFT_FIXED_SIZES(_) \
  _(64) _(128) _(200) _(240) _(256) _(512) _(1024) _(2048) _(4096)
#define FFT_CONTIG_SIZES(_) _(512) _(1024) _(2048) _(4096)

// Radix sets of the kernel families, one bit per radix (bit 1 is length 1);
// the host names them p16, p2, p235 and p2357
C10_METAL_CONSTEXPR uint32_t kFFTMaskP16 = (1u << 1) | (1u << 16);
C10_METAL_CONSTEXPR uint32_t kFFTMaskP2 =
    kFFTMaskP16 | (1u << 2) | (1u << 4) | (1u << 8);
C10_METAL_CONSTEXPR uint32_t kFFTMaskP235 = kFFTMaskP2 | (1u << 3) | (1u << 5);
C10_METAL_CONSTEXPR uint32_t kFFTMaskP2357 = kFFTMaskP235 | (1u << 7);

// tgb transforms of n points per group, jt threads each. n_sig: full signal
// (zero-padded beyond), n_out: outputs. Four-step: input e * ld_mul + i0,
// output k * out_k_mul + i0, i0_dim1 reads i0 from batch dim 1. Smem slot i
// of a row of `row` slots is at i + (i >> padshift); tiles interleave by es.
// index_t: the offset type, 64-bit where one transform spans 2^31 elements.
template <typename index_t>
struct FFTParams {
  uint32_t n, n_sig, n_out, tgb, jt, ndim, nstages;
  index_t in_stride, out_stride; // along the axis
  float scale;
  uint32_t ld_mul, out_k_mul, row, padshift, es;
  bool i0_dim1, via_smem;
  ::c10::metal::array<uint32_t, kFFTMaxStages> radix, ns,
      twstep; // ns: product so far
  ::c10::metal::array<float, kFFTMaxStages> inv_ns;
  ::c10::metal::array<uint32_t, ::c10::metal::max_ndim> sizes;
  ::c10::metal::array<index_t, ::c10::metal::max_ndim> in_strides, out_strides;
};

#ifndef __METAL__
#include <algorithm>
// Host planning constants; the kernel never reads them

#define FFT_LIST_ENTRY(X) X,
// Threadgroup stage radices, largest first
constexpr int kFFTRadices[] = {16, 8, 4, 2, 3, 5, 7};
constexpr int kFFTGlobalRadices[] = {FFT_GLOBAL_RADICES(FFT_LIST_ENTRY)};
constexpr int kFFTFixedSizes[] = {FFT_FIXED_SIZES(FFT_LIST_ENTRY)};
constexpr int kFFTContigSizes[] = {FFT_CONTIG_SIZES(FFT_LIST_ENTRY)};
#undef FFT_LIST_ENTRY
static_assert(std::ranges::all_of(kFFTRadices, [](int r) {
  return (kFFTMaskP2357 & (1u << r)) != 0;
}));

// Below this many elements, extra launches cost more than they save
constexpr int64_t kFFTLongPoints = 1 << 17;
// Past this length, tiles strided on both sides no longer coalesce
constexpr int64_t kFFTMaxCoalescedPoints = 1024;
// Columns of a nested four-step: short, so the recursion's rows stay long
constexpr int64_t kFFTMaxNestedColumns = 256;
// Strided axes tile rows into groups of about this many threads
constexpr int64_t kFFTStridedThreads = 256;
// At least this many rows per strided tile, so its loads coalesce
constexpr int64_t kFFTMinStridedRows = 4;
// A contiguous axis fills this many threads with one transform before adding
// rows
constexpr int64_t kFFTContiguousThreads = 64;
// A contiguous axis adds rows up to this many while more than 1 in
// kFFTIdleLaneDivisor SIMD lanes would sit idle
constexpr int64_t kFFTMaxContiguousRows = 8;
constexpr int64_t kFFTIdleLaneDivisor = 20;
// Rows per group of any axis: tiny transforms would otherwise pile
// kFFTStridedThreads rows into one group
constexpr int64_t kFFTMaxRows = 32;
// Global-memory passes stream in small groups; the unpack pass does one point
// per thread
constexpr int64_t kFFTGlobalThreads = 64;
constexpr int64_t kFFTUnpackThreads = 256;
// setThreadgroupMemoryLength takes multiples of this; passes without smem bind
// the minimum
constexpr int64_t kFFTMinSmemBytes = 16;
#endif
