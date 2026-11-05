#pragma once
#include <c10/metal/common.h>

// Kernel variant bits; their sum ends the kernel name
C10_METAL_CONSTEXPR uint32_t kFFTInverse = 1u << 0;
C10_METAL_CONSTEXPR uint32_t kFFTPackedLoads = 1u << 1;
C10_METAL_CONSTEXPR uint32_t kFFTFourStepTwiddles = 1u << 2;
C10_METAL_CONSTEXPR uint32_t kFFTExtendedStores = 1u << 3;
C10_METAL_CONSTEXPR uint32_t kFFTExtendedLoads = 1u << 4;
C10_METAL_CONSTEXPR uint32_t kFFTGlobalStage = 1u << 5;
C10_METAL_CONSTEXPR uint32_t kFFTPackedRealStores = 1u << 6;
// The bits of four-step, chirp and global-memory passes
C10_METAL_CONSTEXPR uint32_t kFFTMultiPass = kFFTFourStepTwiddles |
    kFFTExtendedStores | kFFTExtendedLoads | kFFTGlobalStage;

// Metal's threadgroup memory limit, and the complex points that fit in it
C10_METAL_CONSTEXPR uint32_t kFFTSmemBytes = 32768;
C10_METAL_CONSTEXPR uint32_t kFFTMaxPoints =
    kFFTSmemBytes / (2 * sizeof(float));
C10_METAL_CONSTEXPR uint32_t kFFTMaxStages = 16;

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

// Radix of the global-memory stage that only unpacks a real transform
C10_METAL_CONSTEXPR uint32_t kFFTUnpackRadix = 1;

// Global-memory stage radices, largest first: lengths factor in this order
#define FFT_GLOBAL_RADICES(_) \
  _(28) _(25) _(21) _(16) _(15) _(12) _(9) _(8) _(7) _(5) _(4) _(3) _(2)

#define FFT_FIXED_SIZES(_) \
  _(64) _(128) _(200) _(240) _(256) _(512) _(1024) _(2048) _(4096)
#define FFT_CONTIG_SIZES(_) _(512) _(1024) _(2048) _(4096)

// tgb transforms of n points per group, jt threads each. n_sig: full signal
// (zero-padded beyond), n_in: inputs (c2r: n_sig/2+1, mirrored), n_out:
// outputs. Four-step: input e * ld_mul + i0, output k * out_k_mul + i0,
// mirror_out adds conj X[k] at n_sig - k, i0_dim1 reads i0 from batch dim 1.
// Smem slot i of a row of `row` slots is at i + (i >> padshift); tiles
// interleave by es.
struct FFTParams {
  uint32_t n, n_sig, n_in, n_out, tgb, jt, ndim, nstages;
  int32_t in_stride, in_unit, out_stride; // along the axis: slice, full, out
  float scale;
  uint32_t ld_mul, out_k_mul, row, padshift, es;
  bool mirror_out, i0_dim1, via_smem;
  ::c10::metal::array<uint32_t, kFFTMaxStages> radix, ns,
      twstep; // ns: product so far
  ::c10::metal::array<float, kFFTMaxStages> inv_ns;
  ::c10::metal::array<uint32_t, ::c10::metal::max_ndim> sizes;
  ::c10::metal::array<int32_t, ::c10::metal::max_ndim> in_strides, out_strides;
};
