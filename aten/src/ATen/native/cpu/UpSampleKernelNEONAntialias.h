// NEON-optimized uint8 bilinear resize with antialiasing for aarch64.
//
// This is the NEON counterpart of UpSampleKernelAVXAntialias.h.
// It only supports num_channels == 3 with channels-last input.

#pragma once
#if defined(__aarch64__)

#include <ATen/core/Tensor.h>
#include <arm_neon.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace {

// Emulate SSE _mm_madd_epi16: multiply 8 int16 pairs element-wise, then add
// adjacent 32-bit products pairwise, producing 4 int32 results.
// [a0*b0+a1*b1, a2*b2+a3*b3, a4*b4+a5*b5, a6*b6+a7*b7]
inline int32x4_t neon_madd_s16(int16x8_t a, int16x8_t b) {
  int32x4_t prod_low = vmull_s16(vget_low_s16(a), vget_low_s16(b));
  int32x4_t prod_high = vmull_s16(vget_high_s16(a), vget_high_s16(b));
  return vpaddq_s32(prod_low, prod_high);
}

// Interpolation horizontal pass: compute x-axis interpolation outputs.
//
// Input data is interleaved RGB:
//   input = [r0, g0, b0, r1, g1, b1, r2, g2, b2, ...]
// Weights are float values rescaled to int16:
//   weights[i] = [w[i, 0], w[i, 1], ..., w[i, K-1]]
// For each output pixel i, we compute:
//   oR[i] = r[xmin[i]] * w[i,0] + ... + r[xmin[i]+K-1] * w[i,K-1]
//   oG[i] = g[xmin[i]] * w[i,0] + ... + g[xmin[i]+K-1] * w[i,K-1]
//   oB[i] = b[xmin[i]] * w[i,0] + ... + b[xmin[i]+K-1] * w[i,K-1]
//
// Processes one row at a time.
void NeonResampleHorizontal(const at::Tensor& unpacked_output,
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

// Interpolation vertical pass: compute y-axis interpolation outputs.
//
// Input data is interleaved RGB:
//   input = [r0, g0, b0, r1, g1, b1, ...]
// For each output row yy, we compute a weighted sum of ids_size input rows:
//   oC[x] = C[ymin[yy] * xstride + x] * w[yy,0] + ... + C[(ymin[yy]+K-1) * xstride + x] * w[yy,K-1]
// for each channel C in {R, G, B} and each horizontal position x.
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

  auto xout = unpacked_output.size(2);
  auto yout = unpacked_output.size(1);
  const auto num_channels = unpacked_input.size(0);
  TORCH_INTERNAL_ASSERT(num_channels == unpacked_output.size(0));

  const int64_t data_size = xout * num_channels;

  for (const auto yy : c10::irange(yout)) {
    const auto* k = &kk[yy * ksize];
    auto ids_min = idx_ptr_xmin[yy];
    auto ids_size = idx_ptr_size[yy];

    uint8_t* C10_RESTRICT lineOut = output_p + yy * data_size;
    const int32_t initial_val = 1 << (vert_weights_precision - 1);
    const int32x4_t initial = vdupq_n_s32(initial_val);

    // Vertical convolution for one output row.
    // Computes a weighted sum of ids_size input rows for all x positions.
    //
    // The data is treated as a flat byte array of size xout * num_channels.
    // We process 16 bytes at a time in the NEON path, then fall back to scalar
    // for the remainder. The vertical pass doesn't need channel deinterleaving
    // because the same weight applies to all channels at a given (x, y) position.
    //
    // To process 2 weights at a time, we interleave pixels from two input rows
    // using vzipq_u8, then use neon_madd_s16 (which pairs adjacent int16 values,
    // multiplies, and adds) so that each pair [pixel_row_i, pixel_row_i+1] is
    // multiplied by [w_i, w_i+1] and summed in one operation.
    int64_t j = 0;

    // Process 16 bytes at a time using 4 accumulators (sss0..sss3),
    // each holding 4 int32 partial sums.
    for (; j + 16 <= data_size; j += 16) {
      int32x4_t sss0 = initial;
      int32x4_t sss1 = initial;
      int32x4_t sss2 = initial;
      int32x4_t sss3 = initial;

      const uint8_t* lineIn_min = input_p + j + ids_min;

      // Process 2 weights at a time by interleaving rows.
      // For weights w0, w1 and pixel bytes from row_i and row_{i+1}:
      //   mmk = [w0, w1, w0, w1, w0, w1, w0, w1]
      //   vzipq interleaves: [r0_0, r1_0, r0_1, r1_1, ...] (from 2 rows)
      //   neon_madd_s16 computes: r0_0*w0 + r1_0*w1, r0_1*w0 + r1_1*w1, ...
      int64_t i = 0;
      for (; i + 1 < ids_size; i += 2) {
        int16_t w0 = k[i];
        int16_t w1 = k[i + 1];
        int16x8_t mmk = {w0, w1, w0, w1, w0, w1, w0, w1};

        uint8x16_t src1 = vld1q_u8(lineIn_min + i * data_size);
        uint8x16_t src2 = vld1q_u8(lineIn_min + (i + 1) * data_size);

        uint8x16x2_t interleaved = vzipq_u8(src1, src2);

        int16x8_t pix0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(interleaved.val[0])));
        int16x8_t pix1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(interleaved.val[0])));
        int16x8_t pix2 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(interleaved.val[1])));
        int16x8_t pix3 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(interleaved.val[1])));

        sss0 = vaddq_s32(sss0, neon_madd_s16(pix0, mmk));
        sss1 = vaddq_s32(sss1, neon_madd_s16(pix1, mmk));
        sss2 = vaddq_s32(sss2, neon_madd_s16(pix2, mmk));
        sss3 = vaddq_s32(sss3, neon_madd_s16(pix3, mmk));
      }

      // Handle remaining single weight (when ids_size is odd)
      for (; i < ids_size; i++) {
        int16x8_t mmk = vdupq_n_s16(k[i]);

        uint8x16_t src = vld1q_u8(lineIn_min + i * data_size);

        int16x8_t pix_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(src)));
        int16x8_t pix_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(src)));

        sss0 = vaddq_s32(sss0, vmull_s16(vget_low_s16(pix_lo), vget_low_s16(mmk)));
        sss1 = vaddq_s32(sss1, vmull_s16(vget_high_s16(pix_lo), vget_high_s16(mmk)));
        sss2 = vaddq_s32(sss2, vmull_s16(vget_low_s16(pix_hi), vget_low_s16(mmk)));
        sss3 = vaddq_s32(sss3, vmull_s16(vget_high_s16(pix_hi), vget_high_s16(mmk)));
      }

      // Right-shift accumulators by coefs_precision to convert fixed-point -> int
      int32x4_t shift = vdupq_n_s32(-static_cast<int32_t>(vert_weights_precision));
      sss0 = vshlq_s32(sss0, shift);
      sss1 = vshlq_s32(sss1, shift);
      sss2 = vshlq_s32(sss2, shift);
      sss3 = vshlq_s32(sss3, shift);

      // Narrow int32 -> int16 (with saturation), then int16 -> uint8 (with
      // saturation), clamping to [0, 255]
      int16x8_t narrow_lo = vcombine_s16(vqmovn_s32(sss0), vqmovn_s32(sss1));
      int16x8_t narrow_hi = vcombine_s16(vqmovn_s32(sss2), vqmovn_s32(sss3));

      // Store 16 output bytes
      vst1_u8(lineOut + j, vqmovun_s16(narrow_lo));
      vst1_u8(lineOut + j + 8, vqmovun_s16(narrow_hi));
    }

    // Scalar fallback for remaining bytes
    for (; j < data_size; j++) {
      int32_t sss = initial_val;
      const uint8_t* lineIn_min = input_p + j + ids_min;

      for (int64_t i = 0; i < ids_size; i++) {
        sss += k[i] * static_cast<int32_t>(lineIn_min[i * data_size]);
      }

      sss >>= vert_weights_precision;
      lineOut[j] = static_cast<uint8_t>(std::clamp(sss, 0, 255));
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

  int ksize_horiz, ksize_vert;
  std::vector<at::Tensor> horiz_indices_weights, vert_indices_weights;
  unsigned int horiz_weights_precision, vert_weights_precision;

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
