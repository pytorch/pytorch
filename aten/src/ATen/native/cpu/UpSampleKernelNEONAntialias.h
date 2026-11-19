// NEON-optimized uint8 bilinear resize with antialiasing for aarch64.
//
// This is the NEON counterpart of UpSampleKernelAVXAntialias.h.
// It only supports num_channels == 3 with channels-last input.

#pragma once
#if defined(__aarch64__)

#include <ATen/core/Tensor.h>
#include <arm_neon.h>
#include <c10/util/irange.h>
#include <cpuinfo.h>

#include <algorithm>
#include <vector>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace {

// Both passes compute, per output value, (sum(pixel * w) + (1 << (precision - 1))) >> precision
// clamped to [0, 255], with int16 weights w and int32 sums, like the AVX2 kernels.
//
// The horizontal pass processes 4 output pixels (of 2 rows) at a time. Each pixel's taps are taken
// in groups of 4: a 16-byte load covers 4 RGB input pixels, regrouped by a table lookup into one
// 32-bit lane per channel. Weights are zero-padded to a multiple of 4 taps, so loads can extend
// past a pixel's taps (contributing 0) but never past the end of the input.
//
// With dotprod, each weight is split exactly as w = 256 * hi + lo (hi signed, lo unsigned), so that
// sum(p * w) = 256 * sum(p * hi) + sum(p * lo) takes two dot products per group of 4 taps: UDOT,
// or USDOT for the signed high bytes of bicubic and lanczos weights (requires i8mm). Without it,
// the same sums use widening multiply-accumulates.

#if defined(__clang__)
#define NEON_RESAMPLE_TARGET_DOTPROD __attribute__((target("dotprod")))
#define NEON_RESAMPLE_TARGET_I8MM __attribute__((target("dotprod,i8mm")))
#else
// GCC 11 only accepts these extensions with an explicit base architecture.
#define NEON_RESAMPLE_TARGET_DOTPROD __attribute__((target("arch=armv8.2-a+dotprod")))
#define NEON_RESAMPLE_TARGET_I8MM __attribute__((target("arch=armv8.2-a+dotprod+i8mm")))
#endif

// Lane c of the result holds channel c of 4 consecutive RGB pixels: {p0[c], p1[c], p2[c], p3[c]}.
inline uint8x16_t neon_rgb_taps_by_channel(const uint8_t* p) {
  const uint8x16_t idx = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 255, 255, 255, 255};
  return vqtbl1q_u8(vld1q_u8(p), idx);
}

// sum over 4 taps of pixel * weight, one int32 lane per channel.
inline int32x4_t neon_mla_rgb_4taps(int32x4_t acc, const uint8_t* p, const int16_t* k) {
  const uint8x16_t idx = {0, 1, 2, 255, 3, 4, 5, 255, 6, 7, 8, 255, 9, 10, 11, 255};
  const uint8x16_t v = vqtbl1q_u8(vld1q_u8(p), idx);
  const int16x8_t lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v)));
  const int16x8_t hi = vreinterpretq_s16_u16(vmovl_high_u8(v));
  const int16x4_t w = vld1_s16(k);
  int32x4_t a = vmlal_lane_s16(acc, vget_low_s16(lo), w, 0);
  int32x4_t b = vmull_lane_s16(vget_low_s16(hi), w, 2);
  a = vmlal_lane_s16(a, vget_high_s16(lo), w, 1);
  b = vmlal_lane_s16(b, vget_high_s16(hi), w, 3);
  return vaddq_s32(a, b);
}

inline void neon_store_rgb_4px(uint8_t* out, const int32x4_t* acc, int32x4_t shift) {
  const uint8x16_t pack = {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 255, 255, 255, 255};
  const uint16x8_t u01 = vcombine_u16(vqmovun_s32(vrshlq_s32(acc[0], shift)), vqmovun_s32(vrshlq_s32(acc[1], shift)));
  const uint16x8_t u23 = vcombine_u16(vqmovun_s32(vrshlq_s32(acc[2], shift)), vqmovun_s32(vrshlq_s32(acc[3], shift)));
  const uint8x16_t px = vqtbl1q_u8(vcombine_u8(vqmovn_u16(u01), vqmovn_u16(u23)), pack);
  vst1_u8(out, vget_low_u8(px));
  vst1q_lane_u32(reinterpret_cast<uint32_t*>(out + 8), vreinterpretq_u32_u8(px), 2);
}

struct NeonHorizontalArgs {
  const uint8_t* lines[2];  // input rows
  uint8_t* outs[2];  // output rows
  int rows;  // 1 or 2
  int64_t xout;
  const int64_t* xmin;  // byte offsets into a row
  const uint8_t* end;  // end of the input buffer
  int groups;  // padded taps / 4
  int32x4_t shift;
};

// 4 output pixels per step using widening multiply-accumulates on the zero-padded weights `kp`.
// Returns the first output pixel not computed.
inline int64_t neon_horizontal_mla(const NeonHorizontalArgs& a, const int16_t* kp) {
  int64_t x = 0;
  for (; x + 4 <= a.xout && a.lines[a.rows - 1] + a.xmin[x + 3] + 12 * (a.groups - 1) + 16 <= a.end; x += 4) {
    for (int r = 0; r < a.rows; r++) {
      int32x4_t acc[4];
      for (int q = 0; q < 4; q++) {
        const uint8_t* p = a.lines[r] + a.xmin[x + q];
        const int16_t* k = kp + (x + q) * 4 * a.groups;
        acc[q] = vdupq_n_s32(0);
        for (int g = 0; g < a.groups; g++) {
          acc[q] = neon_mla_rgb_4taps(acc[q], p + 12 * g, k + 4 * g);
        }
      }
      neon_store_rgb_4px(a.outs[r] + 3 * x, acc, a.shift);
    }
  }
  return x;
}

// Same as neon_horizontal_mla with dot products, for non-negative weights. `w` holds, per output
// pixel and group of 4 taps, {hi bytes, lo bytes} of the weights, each replicated across the 4 lanes.
NEON_RESAMPLE_TARGET_DOTPROD inline int64_t neon_horizontal_udot(const NeonHorizontalArgs& a, const uint8x16_t* w) {
  int64_t x = 0;
  for (; x + 4 <= a.xout && a.lines[a.rows - 1] + a.xmin[x + 3] + 12 * (a.groups - 1) + 16 <= a.end; x += 4) {
    int32x4_t acc[2][4];
    for (int q = 0; q < 4; q++) {
      const int64_t off = a.xmin[x + q];
      const uint8x16_t* wq = w + (x + q) * 2 * a.groups;
      for (int r = 0; r < a.rows; r++) {
        uint32x4_t hi = vdupq_n_u32(0);
        uint32x4_t lo = vdupq_n_u32(0);
        for (int g = 0; g < a.groups; g++) {
          const uint8x16_t v = neon_rgb_taps_by_channel(a.lines[r] + off + 12 * g);
          hi = vdotq_u32(hi, v, wq[2 * g]);
          lo = vdotq_u32(lo, v, wq[2 * g + 1]);
        }
        acc[r][q] = vreinterpretq_s32_u32(vmlaq_n_u32(lo, hi, 256));
      }
    }
    for (int r = 0; r < a.rows; r++) {
      neon_store_rgb_4px(a.outs[r] + 3 * x, acc[r], a.shift);
    }
  }
  return x;
}

// Same as neon_horizontal_udot for weights of any sign: the high bytes are signed.
NEON_RESAMPLE_TARGET_I8MM inline int64_t neon_horizontal_usdot(const NeonHorizontalArgs& a, const uint8x16_t* w) {
  int64_t x = 0;
  for (; x + 4 <= a.xout && a.lines[a.rows - 1] + a.xmin[x + 3] + 12 * (a.groups - 1) + 16 <= a.end; x += 4) {
    int32x4_t acc[2][4];
    for (int q = 0; q < 4; q++) {
      const int64_t off = a.xmin[x + q];
      const uint8x16_t* wq = w + (x + q) * 2 * a.groups;
      for (int r = 0; r < a.rows; r++) {
        int32x4_t hi = vdupq_n_s32(0);
        uint32x4_t lo = vdupq_n_u32(0);
        for (int g = 0; g < a.groups; g++) {
          const uint8x16_t v = neon_rgb_taps_by_channel(a.lines[r] + off + 12 * g);
          hi = vusdotq_s32(hi, v, vreinterpretq_s8_u8(wq[2 * g]));
          lo = vdotq_u32(lo, v, wq[2 * g + 1]);
        }
        acc[r][q] = vmlaq_n_s32(vreinterpretq_s32_u32(lo), hi, 256);
      }
    }
    for (int r = 0; r < a.rows; r++) {
      neon_store_rgb_4px(a.outs[r] + 3 * x, acc[r], a.shift);
    }
  }
  return x;
}

// One output pixel at a time, with 8 taps per step. Without dot products this is faster than
// neon_horizontal_mla for more than 8 taps.
inline void neon_horizontal_by_pixel(const at::Tensor& unpacked_output,
                            const at::Tensor& unpacked_input,
                            int ksize,
                            const std::vector<at::Tensor>& horiz_indices_weights,
                            unsigned int horiz_weights_precision) {
  const auto* kk = (const int16_t*)(horiz_indices_weights[3].const_data_ptr<double>());

  auto xout = unpacked_output.size(2);
  auto yin = unpacked_output.size(1);
  auto xin = unpacked_input.size(2);
  const auto num_channels = unpacked_input.size(0);
  TORCH_INTERNAL_ASSERT(num_channels == 3);

  const int64_t* idx_ptr_xmin = horiz_indices_weights[0].const_data_ptr<int64_t>();
  const int64_t* idx_ptr_size = horiz_indices_weights[1].const_data_ptr<int64_t>();

  uint8_t* output_p = unpacked_output.data_ptr<uint8_t>();
  const uint8_t* input_p = unpacked_input.const_data_ptr<uint8_t>();

  auto xout_stride = xout * num_channels;
  auto xin_stride = xin * num_channels;

  for (int64_t yy = 0; yy < yin; yy++) {
    uint8_t* C10_RESTRICT lineOut = output_p + yy * xout_stride;
    const uint8_t* C10_RESTRICT lineIn = input_p + yy * xin_stride;
    const int32_t initial_val = 1 << (horiz_weights_precision - 1);

    // Horizontal convolution for a single row.
    //
    // For each output pixel out_x, computes the weighted sum of ids_size input
    // pixels using the weight vector k[]. Uses vld3_u8 to load 8 interleaved RGB
    // pixels and deinterleave them into separate R, G, B vectors. This avoids
    // the shuffle masks needed by the AVX2 path.
    //
    // The weight vector is split as: ids_size = num_blocks_8 * 8 + remainder.
    // We process 8 weights in a vectorized loop, then handle remaining weights
    // in a scalar cleanup loop. Accumulators are kept per-channel in int32 to
    // avoid overflow.
    for (int64_t out_x = 0; out_x < xout; out_x++) {
      // ids_min is a byte offset (pre-multiplied by num_channels)
      const int64_t ids_min = idx_ptr_xmin[out_x];
      const int64_t ids_size = idx_ptr_size[out_x];
      const int16_t* k = &kk[out_x * ksize];
      const uint8_t* lineIn_min = lineIn + ids_min;

      int32x4_t acc_r = vdupq_n_s32(0);
      int32x4_t acc_g = vdupq_n_s32(0);
      int32x4_t acc_b = vdupq_n_s32(0);

      int64_t i = 0;

      // Block 8: load 8 RGB pixels with vld3, deinterleave, widen to int16,
      // multiply-accumulate with 8 weights per channel.
      // Each vmlal_s16 accumulates 4 products; calling it on the low and high
      // halves puts all 8 products into 4 int32 lanes, reduced later by
      // vaddvq_s32.
      for (; i + 8 <= ids_size; i += 8) {
        uint8x8x3_t rgb = vld3_u8(lineIn_min + num_channels * i);
        int16x8_t weights = vld1q_s16(&k[i]);

        int16x8_t r16 = vreinterpretq_s16_u16(vmovl_u8(rgb.val[0]));
        int16x8_t g16 = vreinterpretq_s16_u16(vmovl_u8(rgb.val[1]));
        int16x8_t b16 = vreinterpretq_s16_u16(vmovl_u8(rgb.val[2]));

        acc_r = vmlal_s16(acc_r, vget_low_s16(r16), vget_low_s16(weights));
        acc_r = vmlal_s16(acc_r, vget_high_s16(r16), vget_high_s16(weights));
        acc_g = vmlal_s16(acc_g, vget_low_s16(g16), vget_low_s16(weights));
        acc_g = vmlal_s16(acc_g, vget_high_s16(g16), vget_high_s16(weights));
        acc_b = vmlal_s16(acc_b, vget_low_s16(b16), vget_low_s16(weights));
        acc_b = vmlal_s16(acc_b, vget_high_s16(b16), vget_high_s16(weights));
      }

      // Block 4: handle 4 pixels that didn't fit in a block of 8
      // We also use vld3_u8 here, which still loads 8 pixels, but we only use
      // the lower half - so the computation is correct.
      // On all rows except the last one, reading 8 pixels is safe (the tensors
      // are channels-last). But on the last row, we have to be careful not to
      // read past the buffer, hence the extra boundary check.
      const uint8_t* block_of_4_safe_load_end = input_p + yin * xin_stride - 24;
      for (; i + 4 <= ids_size && lineIn_min + num_channels * i <= block_of_4_safe_load_end; i += 4) {
        uint8x8x3_t rgb = vld3_u8(lineIn_min + num_channels * i);
        int16x4_t weights4 = vld1_s16(&k[i]);

        int16x4_t r16 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb.val[0])));
        int16x4_t g16 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb.val[1])));
        int16x4_t b16 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb.val[2])));

        acc_r = vmlal_s16(acc_r, r16, weights4);
        acc_g = vmlal_s16(acc_g, g16, weights4);
        acc_b = vmlal_s16(acc_b, b16, weights4);
      }

      // Horizontal reduction + rounding bias
      int32_t sum_r = vaddvq_s32(acc_r) + initial_val;
      int32_t sum_g = vaddvq_s32(acc_g) + initial_val;
      int32_t sum_b = vaddvq_s32(acc_b) + initial_val;

      // Scalar cleanup for remaining pixels
      for (; i < ids_size; i++) {
        int16_t w = k[i];
        const uint8_t* p = lineIn_min + num_channels * i;
        sum_r += w * p[0];
        sum_g += w * p[1];
        sum_b += w * p[2];
      }

      // Right-shift and clamp to [0, 255]
      uint8_t* out = lineOut + num_channels * out_x;
      out[0] = static_cast<uint8_t>(std::clamp(sum_r >> horiz_weights_precision, 0, 255));
      out[1] = static_cast<uint8_t>(std::clamp(sum_g >> horiz_weights_precision, 0, 255));
      out[2] = static_cast<uint8_t>(std::clamp(sum_b >> horiz_weights_precision, 0, 255));
    }
  }
}

// Interpolation horizontal pass on interleaved RGB rows, one output row per input row.
void NeonResampleHorizontal(const at::Tensor& unpacked_output,
                            const at::Tensor& unpacked_input,
                            int ksize,
                            const std::vector<at::Tensor>& horiz_indices_weights,
                            unsigned int horiz_weights_precision) {
  const auto* kk = (const int16_t*)(horiz_indices_weights[3].const_data_ptr<double>());
  const int64_t xout = unpacked_output.size(2);
  const int64_t yin = unpacked_output.size(1);
  const int64_t xin = unpacked_input.size(2);
  const auto num_channels = unpacked_input.size(0);
  TORCH_INTERNAL_ASSERT(num_channels == 3);
  const int64_t* idx_ptr_xmin = horiz_indices_weights[0].const_data_ptr<int64_t>();
  const int64_t* idx_ptr_size = horiz_indices_weights[1].const_data_ptr<int64_t>();
  uint8_t* output_p = unpacked_output.data_ptr<uint8_t>();
  const uint8_t* input_p = unpacked_input.const_data_ptr<uint8_t>();
  const int64_t xout_stride = xout * num_channels;
  const int64_t xin_stride = xin * num_channels;

  bool use_dot = cpuinfo_initialize() && cpuinfo_has_arm_neon_dot();
  if (!use_dot && ksize > 8) {
    return neon_horizontal_by_pixel(unpacked_output, unpacked_input, ksize, horiz_indices_weights, horiz_weights_precision);
  }
  const int groups = (ksize + 3) / 4;
  std::vector<int16_t> kp(xout * 4 * groups, 0);
  bool negative_weights = false;
  for (const auto i : c10::irange(xout)) {
    for (const auto j : c10::irange(ksize)) {
      kp[i * 4 * groups + j] = kk[i * ksize + j];
      negative_weights |= kk[i * ksize + j] < 0;
    }
  }
  if (use_dot && negative_weights && !cpuinfo_has_arm_i8mm()) {
    if (ksize > 8) {
      return neon_horizontal_by_pixel(unpacked_output, unpacked_input, ksize, horiz_indices_weights, horiz_weights_precision);
    }
    use_dot = false;
  }
  std::vector<uint8x16_t> wdot;
  if (use_dot) {
    wdot.resize(xout * 2 * groups);
    for (const auto i : c10::irange(xout * groups)) {
      uint8_t hi[16], lo[16];
      for (const auto t : c10::irange(4)) {
        const int16_t w = kp[i * 4 + t];
        for (const auto lane : c10::irange(4)) {
          hi[4 * lane + t] = static_cast<uint8_t>(static_cast<int8_t>(w >> 8));
          lo[4 * lane + t] = static_cast<uint8_t>(w & 255);
        }
      }
      wdot[2 * i] = vld1q_u8(hi);
      wdot[2 * i + 1] = vld1q_u8(lo);
    }
  }

  NeonHorizontalArgs args;
  args.xout = xout;
  args.xmin = idx_ptr_xmin;
  args.end = input_p + yin * xin_stride;
  args.groups = groups;
  args.shift = vdupq_n_s32(-static_cast<int32_t>(horiz_weights_precision));
  for (int64_t y0 = 0; y0 < yin; y0 += 2) {
    args.rows = static_cast<int>(std::min<int64_t>(2, yin - y0));
    for (const auto r : c10::irange(2)) {
      const int64_t y = y0 + std::min<int64_t>(r, args.rows - 1);
      args.lines[r] = input_p + y * xin_stride;
      args.outs[r] = output_p + y * xout_stride;
    }
    int64_t x_done = 0;
    if (!use_dot) {
      x_done = neon_horizontal_mla(args, kp.data());
    } else if (negative_weights) {
      x_done = neon_horizontal_usdot(args, wdot.data());
    } else {
      x_done = neon_horizontal_udot(args, wdot.data());
    }
    // Remaining output pixels of these rows, one at a time.
    for (const auto r : c10::irange(args.rows)) {
      for (int64_t x = x_done; x < xout; x++) {
        const uint8_t* p = args.lines[r] + idx_ptr_xmin[x];
        const int16_t* k = kp.data() + x * 4 * groups;
        int32x4_t acc = vdupq_n_s32(0);
        int g = 0;
        for (; g < groups && p + 12 * g + 16 <= args.end; g++) {
          acc = neon_mla_rgb_4taps(acc, p + 12 * g, k + 4 * g);
        }
        int32_t sums[4];
        vst1q_s32(sums, acc);
        for (int64_t t = 4 * g; t < idx_ptr_size[x]; t++) {
          for (const auto c : c10::irange(3)) {
            sums[c] += k[t] * p[3 * t + c];
          }
        }
        uint8_t* out = args.outs[r] + 3 * x;
        for (const auto c : c10::irange(3)) {
          out[c] = static_cast<uint8_t>(std::clamp((sums[c] + (1 << (horiz_weights_precision - 1))) >> horiz_weights_precision, 0, 255));
        }
      }
    }
  }
}

// One output row of the vertical pass: a weighted sum of `n` input rows `row_bytes` apart.
// N is the number of taps when known at compile time (0 otherwise).
template <int N>
inline void neon_vertical_row(const uint8_t* base, uint8_t* out, int64_t row_bytes, const int16_t* k, int n_rt,
                              int32x4_t shift, unsigned int precision) {
  const int n = N ? N : n_rt;
  int16x8_t w[N ? N : 1];
  if constexpr (N > 0) {
    for (int i = 0; i < N; i++) {
      w[i] = vdupq_n_s16(k[i]);
    }
  }
  int64_t j = 0;
  for (; j + 16 <= row_bytes; j += 16) {
    int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0;
    for (int i = 0; i < n; i++) {
      const uint8x16_t v = vld1q_u8(base + i * row_bytes + j);
      const int16x8_t lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v)));
      const int16x8_t hi = vreinterpretq_s16_u16(vmovl_high_u8(v));
      const int16x8_t wi = N ? w[N ? i : 0] : vdupq_n_s16(k[i]);
      a0 = vmlal_s16(a0, vget_low_s16(lo), vget_low_s16(wi));
      a1 = vmlal_high_s16(a1, lo, wi);
      a2 = vmlal_s16(a2, vget_low_s16(hi), vget_low_s16(wi));
      a3 = vmlal_high_s16(a3, hi, wi);
    }
    const uint16x8_t u0 = vcombine_u16(vqmovun_s32(vrshlq_s32(a0, shift)), vqmovun_s32(vrshlq_s32(a1, shift)));
    const uint16x8_t u1 = vcombine_u16(vqmovun_s32(vrshlq_s32(a2, shift)), vqmovun_s32(vrshlq_s32(a3, shift)));
    vst1q_u8(out + j, vcombine_u8(vqmovn_u16(u0), vqmovn_u16(u1)));
  }
  for (; j < row_bytes; j++) {
    int32_t sss = 1 << (precision - 1);
    for (int i = 0; i < n; i++) {
      sss += k[i] * static_cast<int32_t>(base[i * row_bytes + j]);
    }
    out[j] = static_cast<uint8_t>(std::clamp(sss >> precision, 0, 255));
  }
}

// Interpolation vertical pass: each output row is a weighted sum of ids_size input rows.
void NeonResampleVertical(const at::Tensor& unpacked_output,
                          const at::Tensor& unpacked_input,
                          int ksize,
                          const std::vector<at::Tensor>& vert_indices_weights,
                          unsigned int vert_weights_precision) {
  const auto* kk = (const int16_t*)(vert_indices_weights[3].const_data_ptr<double>());
  const int64_t* idx_ptr_xmin = vert_indices_weights[0].const_data_ptr<int64_t>();
  const int64_t* idx_ptr_size = vert_indices_weights[1].const_data_ptr<int64_t>();
  uint8_t* output_p = unpacked_output.data_ptr<uint8_t>();
  const uint8_t* input_p = unpacked_input.const_data_ptr<uint8_t>();
  const auto xout = unpacked_output.size(2);
  const auto yout = unpacked_output.size(1);
  const auto num_channels = unpacked_input.size(0);
  TORCH_INTERNAL_ASSERT(num_channels == unpacked_output.size(0));
  const int64_t data_size = xout * num_channels;
  const int32x4_t shift = vdupq_n_s32(-static_cast<int32_t>(vert_weights_precision));
  for (const auto yy : c10::irange(yout)) {
    const int16_t* k = &kk[yy * ksize];
    const uint8_t* base = input_p + idx_ptr_xmin[yy];
    uint8_t* out = output_p + yy * data_size;
    const auto n = idx_ptr_size[yy];
    switch (n) {
      case 1: neon_vertical_row<1>(base, out, data_size, k, 1, shift, vert_weights_precision); break;
      case 2: neon_vertical_row<2>(base, out, data_size, k, 2, shift, vert_weights_precision); break;
      case 3: neon_vertical_row<3>(base, out, data_size, k, 3, shift, vert_weights_precision); break;
      case 4: neon_vertical_row<4>(base, out, data_size, k, 4, shift, vert_weights_precision); break;
      default: neon_vertical_row<0>(base, out, data_size, k, static_cast<int>(n), shift, vert_weights_precision);
    }
  }
}

// Main entry point for NEON-accelerated uint8 bilinear resize.
//
// Only supports num_channels == 3 with channels-last memory format.
// Mirrors upsample_avx_bilinear_bicubic_uint8 but uses NEON intrinsics and
// works directly on interleaved RGB data (no unpack/pack step needed).
//
// Weights are computed as int16 values by compute_index_ranges_int16_weights
// with align_i32=false (no int32 alignment padding needed, unlike AVX2 which
// processes pixels as 4-byte RGBA units).
template <typename scale_type, class F>
void upsample_neon_bilinear_bicubic_uint8(const at::Tensor& input_,
                                          const at::Tensor& output,
                                          bool align_corners,
                                          const scale_type& scales,
                                          bool antialias) {
  auto batch_size = input_.size(0);
  auto num_channels = input_.size(1);
  auto xin = input_.size(3);
  auto yin = input_.size(2);
  auto xout = output.size(3);
  auto yout = output.size(2);

  if (xin == xout && yin == yout) {
    output.copy_(input_);
    return;
  }

  TORCH_INTERNAL_ASSERT(num_channels == 3);
  TORCH_INTERNAL_ASSERT(output.is_contiguous(at::MemoryFormat::ChannelsLast));

  auto input = input_.contiguous(at::MemoryFormat::ChannelsLast);

  auto need_horizontal = xout != xin;
  auto need_vertical = yout != yin;

  int ksize_horiz = 0, ksize_vert = 0;
  std::vector<at::Tensor> horiz_indices_weights, vert_indices_weights;
  unsigned int horiz_weights_precision = 0, vert_weights_precision = 0;

  if (need_horizontal) {
    int interp_dim = 3;
    std::tie(horiz_indices_weights, ksize_horiz, horiz_weights_precision) = F::compute_index_ranges_int16_weights(
        /*input_size=*/xin,
        /*output_size=*/xout,
        /*stride=*/num_channels,
        /*ndims=*/4,
        /*reshape_dim=*/interp_dim,
        /*align_corners=*/align_corners,
        /*opt_scale=*/scales[interp_dim - 2],
        /*antialias=*/antialias,
        /*align_i32=*/false);
  }

  if (need_vertical) {
    int interp_dim = 2;
    std::tie(vert_indices_weights, ksize_vert, vert_weights_precision) = F::compute_index_ranges_int16_weights(
        /*input_size=*/yin,
        /*output_size=*/yout,
        /*stride=*/num_channels * xout,
        /*ndims=*/4,
        /*reshape_dim=*/interp_dim,
        /*align_corners=*/align_corners,
        /*opt_scale=*/scales[interp_dim - 2],
        /*antialias=*/antialias,
        /*align_i32=*/false);
  }

  at::Tensor buffer_horiz;
  if (need_horizontal && need_vertical) {
    buffer_horiz = at::empty({num_channels, yin, xout}, input.options());
  }

  for (const auto i : c10::irange(batch_size)) {
    at::Tensor input_slice = input[i];

    if (need_horizontal) {
      at::Tensor horiz_output = need_vertical ? buffer_horiz : output[i];
      NeonResampleHorizontal(horiz_output, input_slice, ksize_horiz, horiz_indices_weights, horiz_weights_precision);
      if (need_vertical) {
        input_slice = horiz_output;
      }
    }
    if (need_vertical) {
      NeonResampleVertical(output[i], input_slice, ksize_vert, vert_indices_weights, vert_weights_precision);
    }
  }
}

} // anonymous namespace

#endif // __aarch64__
