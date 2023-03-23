/*
The Python Imaging Library (PIL) is

    Copyright © 1997-2011 by Secret Labs AB
    Copyright © 1995-2011 by Fredrik Lundh

Pillow is the friendly PIL fork. It is

    Copyright © 2010-2022 by Alex Clark and contributors

Like PIL, Pillow is licensed under the open source HPND License
*/

// This code is heavily inspired from PILLOW-SIMD's implementation:
// https://github.com/uploadcare/pillow-simd/blob/simd/master/src/libImaging/Resample.c

#pragma once
#ifdef CPU_CAPABILITY_AVX2
// TODO: This file only supports AVX2. We could split the AVX kernels into
// smaller logical blocks in order to port them into the Vec.h logic. This would
// allow to support other vectorization architectures and perhaps also support
// the non-vectorized fallback (we'd need to make sure it's not slower than the
// current fallback).

#include <ATen/core/Tensor.h>
#include <ATen/cpu/vec/intrinsics.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif


namespace {

static __m128i inline mm_cvtepu8_epi32(const uint32_t* C10_RESTRICT ptr) {
  return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)ptr));
}

at::Tensor unpack_rgb(const at::Tensor& packed_tensor) {
  // Convert a "packed" tensor (typically RGBRGBRGB if channels_last) into
  // RGBARGBARGBA format where A is hard-coded to 0. Each pixel is encoded
  // into as 32 bits. This generalizes to num_channels <= 4 and also works for
  // non-channels_last tensors.

  const uint8_t* packed = (const uint8_t*)packed_tensor.data_ptr<uint8_t>();
  auto num_pixels = packed_tensor.size(1) * packed_tensor.size(2);
  auto num_channels = packed_tensor.size(0);

  constexpr int rgba_size = 4;
  auto unpacked_tensor = at::empty({rgba_size, packed_tensor.size(1), packed_tensor.size(2)}, at::CPU(at::kByte));
  uint8_t* unpacked = (uint8_t*) unpacked_tensor.data_ptr<uint8_t>();

  auto stride_i = packed_tensor.stride(2);
  auto stride_j = packed_tensor.stride(0);

  for (const auto i : c10::irange(num_pixels)) {
    for (const auto j : c10::irange(rgba_size)) {
      unpacked[rgba_size * i + j] = (j < num_channels) ? packed[stride_i * i + stride_j * j] : 0;
    }
  }
  return unpacked_tensor;
}

void pack_rgb(
    const at::Tensor& unpacked_tensor, // IN
    const at::Tensor& packed_tensor // OUT
) {
  // Convert from unpacked channels last 4-channels tensor into original data layout.

  constexpr int rgba_size = 4;
  uint8_t* unpacked = (uint8_t*)unpacked_tensor.data_ptr<uint8_t>();
  uint8_t* packed = (uint8_t*)packed_tensor.data_ptr<uint8_t>();
  auto num_pixels = packed_tensor.size(1) * packed_tensor.size(2);
  auto num_channels = packed_tensor.size(0);

  auto packed_increment = packed_tensor.stride(2);
  auto packed_stride = packed_tensor.stride(0);

  for (const auto i C10_UNUSED : c10::irange(num_pixels)) {
    for (const auto j : c10::irange(num_channels)) {
      packed[j * packed_stride] = unpacked[j];
    }
    unpacked += rgba_size;
    packed += packed_increment;
  }
}

void ImagingResampleHorizontalConvolution8u4x(
    uint32_t* C10_RESTRICT lineOut0,
    uint32_t* C10_RESTRICT lineOut1,
    uint32_t* C10_RESTRICT lineOut2,
    uint32_t* C10_RESTRICT lineOut3,
    int64_t out_xsize,
    const uint32_t* C10_RESTRICT lineIn0,
    const uint32_t* C10_RESTRICT lineIn1,
    const uint32_t* C10_RESTRICT lineIn2,
    const uint32_t* C10_RESTRICT lineIn3,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision);

void ImagingResampleHorizontalConvolution8u(
    uint32_t* C10_RESTRICT lineOut,
    int64_t out_xsize,
    const uint32_t* C10_RESTRICT lineIn,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision);

void ImagingResampleVerticalConvolution8u(
    uint32_t* C10_RESTRICT lineOut,
    const uint32_t* C10_RESTRICT lineIn,
    int64_t xsize,
    int64_t ids_min,
    int64_t ids_size,
    const int16_t* k,
    unsigned int coefs_precision);

void ImagingResampleHorizontal(
    const at::Tensor & unpacked_output,
    const at::Tensor & unpacked_input,
    int ksize,
    const std::vector<at::Tensor>& horiz_indices_weights,
    unsigned int horiz_weights_precision) {

  // Interpolation horizontal pass: we compute x-axis (image width) interpolation outputs.

  // Input data is stored as
  //   input = [r[0], g[0], b[0], a[0], r[1], g[1], b[1], a[1], r[2], g[2], b[2], a[2], ...]
  // Weights are float values computed for each output pixel and rescaled to uint16:
  //   weights[i] = [w[i, 0], w[i, 1], ..., w[i, K-1]]
  // We want to compute the output as following:
  //   output = [oR[0], oG[0], oB[0], oA[0], oR[1], oG[1], oB[1], oA[1], ...]
  // where
  //   oR[yoffset + i] = r[yoffset + xmin[i]] * w[i, 0] + ... + r[yoffset + xmin[i] + K-1] * w[i, K-1]
  //   oG[yoffset + i] = g[yoffset + xmin[i]] * w[i, 0] + ... + g[yoffset + xmin[i] + K-1] * w[i, K-1]
  //   oB[yoffset + i] = b[yoffset + xmin[i]] * w[i, 0] + ... + b[yoffset + xmin[i] + K-1] * w[i, K-1]
  //

  // TODO: we may want to merge that into the fallback code (currently called
  // basic_loop_aa_horizontal<uint8_t>)
  // Although this may not be needed if / when we port all this code to use
  // Vec.h since this would potentially give us another fall-back implem

  const int16_t* kk = (int16_t*)(horiz_indices_weights[3].data_ptr<double>());

  auto xout = unpacked_output.size(2);
  auto yout = unpacked_output.size(1);
  auto xin = unpacked_input.size(2);

  const int64_t* idx_ptr_xmin = horiz_indices_weights[0].data_ptr<int64_t>();
  const int64_t* idx_ptr_size = horiz_indices_weights[1].data_ptr<int64_t>();

  uint32_t* unpacked_output_p = (uint32_t*) unpacked_output.data_ptr<uint8_t>();
  const uint32_t* unpacked_input_p = (uint32_t*) unpacked_input.data_ptr<uint8_t>();

  int64_t yy = 0;
  for (; yy < yout - 3; yy += 4) {
    ImagingResampleHorizontalConvolution8u4x(
        unpacked_output_p + yy * xout,
        unpacked_output_p + (yy + 1) * xout,
        unpacked_output_p + (yy + 2) * xout,
        unpacked_output_p + (yy + 3) * xout,
        xout,
        unpacked_input_p + yy * xin,
        unpacked_input_p + (yy + 1) * xin,
        unpacked_input_p + (yy + 2) * xin,
        unpacked_input_p + (yy + 3) * xin,
        idx_ptr_xmin,
        idx_ptr_size,
        kk,
        ksize,
        horiz_weights_precision);
  }
  for (; yy < yout; yy++) {
    ImagingResampleHorizontalConvolution8u(
        unpacked_output_p + yy * xout,
        xout,
        unpacked_input_p + yy * xin,
        idx_ptr_xmin,
        idx_ptr_size,
        kk,
        ksize,
        horiz_weights_precision);
  }
}

void ImagingResampleVertical(
    const at::Tensor & unpacked_output,
    const at::Tensor & unpacked_input,
    int ksize,
    const std::vector<at::Tensor>& vert_indices_weights,
    unsigned int vert_weights_precision) {

  // Interpolation vertical pass: we compute y-axis interpolation outputs.
  // Input data is stored as
  //   input = [r[0], g[0], b[0], a[0], r[1], g[1], b[1], a[1], r[2], g[2], b[2], a[2], ...]
  // Weights are float values computed for each output pixel and rescaled to uint16:
  //   weights[i] = [w[i, 0], w[i, 1], ..., w[i, K-1]]
  // We want to compute the output as following:
  //   output = [oR[0], oG[0], oB[0], oA[0], oR[1], oG[1], oB[1], oA[1], ...]
  // where
  //   oR[xoffset + i] = r[xoffset + ymin[i]] * w[i, 0] + ... + r[xoffset + ymin[i] + (K-1) * xsize] * w[i, K-1]
  //   oG[xoffset + i] = g[xoffset + ymin[i]] * w[i, 0] + ... + g[xoffset + ymin[i] + (K-1) * xsize] * w[i, K-1]
  //   oB[xoffset + i] = b[xoffset + ymin[i]] * w[i, 0] + ... + b[xoffset + ymin[i] + (K-1) * xsize] * w[i, K-1]

  // TODO: we may want to merge that into the fallback code (currently called
  // basic_loop_aa_vertical<uint8_t>)
  // Although this may not be needed if / when we port all this code to use
  // Vec.h since this would potentially give us another fall-back implem
  const int16_t* kk = (int16_t*)(vert_indices_weights[3].data_ptr<double>());

  const int64_t* idx_ptr_xmin = vert_indices_weights[0].data_ptr<int64_t>();
  const int64_t* idx_ptr_size = vert_indices_weights[1].data_ptr<int64_t>();

  uint32_t* unpacked_output_p = (uint32_t*) unpacked_output.data_ptr<uint8_t>();
  const uint32_t* unpacked_input_p = (uint32_t*) unpacked_input.data_ptr<uint8_t>();

  auto xout = unpacked_output.size(2);
  auto yout = unpacked_output.size(1);

  for (const auto yy : c10::irange(yout)) {
    const auto* k = &kk[yy * ksize];
    auto ids_min = idx_ptr_xmin[yy];
    auto ids_size = idx_ptr_size[yy];
    ImagingResampleVerticalConvolution8u(
        unpacked_output_p + yy * xout,
        unpacked_input_p,
        xout,
        ids_min,
        ids_size,
        k,
        vert_weights_precision);
  }
}

// This is the only public entry point in this file.  It supports bilinear
// mode for uint8 dtype when C <= 4, with or without antialias. The
// implem is based on PIL-SIMD.
// Its equivalent implementation (fallback) for when AVX isn't supported or when
// C > 4 is separable_upsample_generic_Nd_kernel_impl()  There are a bunch of
// future improvement that can be done: look for the TODOs in this file.
// For details on how the weights are computed and how the multiplications are
// run on int (instead of float weights), see
// [ Weights computation for uint8_t and multiplication trick ]
// For details on how the AVX kernels are implemented, see
// https://gist.github.com/NicolasHug/47c97d731f05eaad5694c173849b86f5
// See also [ Support for antialias=False as a subcase of antilias=True ] to
// learn more about how the antialias=False case is computed. The same holds
// here: all these kernels are general enough to handle an arbitrary number of
// weights, but when aa=False they could be optimized further.
template <typename scale_type, class F>
void upsample_avx_bilinear_uint8(
    const at::Tensor& input,
    const at::Tensor& output,
    bool align_corners,
    const scale_type& scales,
    bool antialias) {
  auto batch_size = input.size(0);
  auto num_channels = input.size(1);
  auto xin = input.size(3);
  auto yin = input.size(2);
  auto xout = output.size(3);
  auto yout = output.size(2);

  if (xin == xout && yin == yout) {
    output.copy_(input);
    return;
  }

  auto need_horizontal = xout != xin;
  auto need_vertical = yout != yin;

  int ksize_horiz, ksize_vert;
  std::vector<at::Tensor> horiz_indices_weights, vert_indices_weights;
  unsigned int horiz_weights_precision, vert_weights_precision;

  if (need_horizontal) {
    int interp_dim = 3;
    std::tie(horiz_indices_weights, ksize_horiz, horiz_weights_precision) =
        F::compute_indices_int16_weights_aa(
            /*input_size=*/xin,
            /*output_size=*/xout,
            /*stride=*/1,
            /*ndims=*/4,
            /*reshape_dim=*/interp_dim,
            /*align_corners=*/align_corners,
            /*opt_scale=*/scales[interp_dim - 2],
            /*antialias=*/antialias,
            /*align_i32=*/true);
  }

  if (need_vertical) {
    int interp_dim = 2;
    std::tie(vert_indices_weights, ksize_vert, vert_weights_precision) =
        F::compute_indices_int16_weights_aa(
            /*input_size=*/yin,
            /*output_size=*/yout,
            /*stride=*/xout,
            /*ndims=*/4,
            /*reshape_dim=*/interp_dim,
            /*align_corners=*/align_corners,
            /*opt_scale=*/scales[interp_dim - 2],
            /*antialias=*/antialias,
            /*align_i32=*/true);
  }

  bool needs_unpacking = num_channels == 4 && input.is_contiguous(at::MemoryFormat::ChannelsLast);

  at::Tensor buffer_horiz, buffer_vert;
  // Minor optimization: we can avoid allocating an extra buffer if we're performing
  // horizontal-only or vertical-only interpolation, and if the tensor doesn't
  // need unpacking
  if (need_horizontal && !(needs_unpacking && !need_vertical)) {
    buffer_horiz = at::empty({4, yin, xout}, input.options());
  }
  if (need_vertical && !needs_unpacking) {
    buffer_vert = at::empty({4, yout, xout}, input.options());
  }

  // TODO: The unpack / pack operations create a copy of the original input and
  // output tensor. There should be a way to avoid these copies by instead
  // modifying the low-level kernels. Or maybe at least avoid copying the entire
  // tensors and just copy part of them (line by line).
  for (const auto i : c10::irange(batch_size)) {

    at::Tensor unpacked_input = (needs_unpacking) ? input[i] : unpack_rgb(input[i]);
    at::Tensor unpacked_output;

    if (need_horizontal) {

      at::Tensor unpacked_output_temp = (needs_unpacking && !need_vertical) ? output[i] : buffer_horiz;

      ImagingResampleHorizontal(
          unpacked_output_temp,
          unpacked_input,
          ksize_horiz,
          horiz_indices_weights,
          horiz_weights_precision);
      unpacked_output = unpacked_input = unpacked_output_temp;
    }
    if (need_vertical) {
      unpacked_output = (needs_unpacking) ? output[i] : buffer_vert;

      ImagingResampleVertical(
          unpacked_output,
          unpacked_input,
          ksize_vert,
          vert_indices_weights,
          vert_weights_precision);
    }

    TORCH_INTERNAL_ASSERT(unpacked_output.defined());

    if (!needs_unpacking) {
      pack_rgb(unpacked_output, output[i]);
    }
  }
}

void ImagingResampleHorizontalConvolution8u4x(
    uint32_t* C10_RESTRICT lineOut0,
    uint32_t* C10_RESTRICT lineOut1,
    uint32_t* C10_RESTRICT lineOut2,
    uint32_t* C10_RESTRICT lineOut3,
    int64_t out_xsize,
    const uint32_t* C10_RESTRICT lineIn0,
    const uint32_t* C10_RESTRICT lineIn1,
    const uint32_t* C10_RESTRICT lineIn2,
    const uint32_t* C10_RESTRICT lineIn3,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision) {
  // Interpolation horizontal pass processing together 4 vertical lines.
  // - Input data format is RGBA with R,G,B,A being uint8, we can encode 4 values as a single uint32 value.
  // - We split the size of weight vector for a given output index as a sum:
  //   ids_size = num_blocks_4 * 4 + num_blocks_2 * 2 + num_blocks_1.
  // - We load and process 4 weights values in a loop ("block 4") then we process 2 weights values
  // in another loop ("block 2") and finally we process 1 weights value in the final loop ("block 1").

  // Define shuffling masks (low/high) for num_channels 4
  // Mask low casts lower half of each lane to epi16 and reorder RGBARGBA -> RRGGBBAA:
  //   [r1 g1 b1 a1  r2 g2 b2 a2  ... | R1 G1 B1 A1  R2 G2 B2 A2 ... ] ->
  //   [r1 0 r2 0  g1 0 g2 0  b1 0 b2 0  a1 0 a2 0 | R1 0 R2 0  G1 0 G2 0  B1 0 B2 0  A1 0 A2 0]
  // Mask high casts upper half of each lane to epi16 and reorder RGBARGBA -> RRGGBBAA::
  //   [ ... r3 g3 b3 a3  r4 g4 b4 a4 | ... R3 G3 B3 A3  R4 G4 B4 A4 ] ->
  //   [r3 0 r4 0  g3 0 g4 0  b3 0 b4 0  a3 0 a4 0 | R3 0 R4 0  G3 0 G4 0  B3 0 B4 0  A3 0 A4 0]

  const auto mask_low_c4 = _mm256_set_epi8(
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0,
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0);
  const auto mask_high_c4 = _mm256_set_epi8(
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8,
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8);

  const auto mask_low = mask_low_c4;
  const auto mask_high = mask_high_c4;

  const auto zero = _mm256_setzero_si256();
  const auto initial = _mm256_set1_epi32(1 << (coefs_precision - 1));

  for (const auto out_x : c10::irange(out_xsize)) {
    const auto ids_min = idx_ptr_xmin[out_x];
    const auto ids_size = idx_ptr_size[out_x];
    const auto * k = &kk[out_x * kmax];
    int64_t i = 0;

    auto sss0 = initial;
    auto sss1 = initial;

    const auto * lineIn0_min = lineIn0 + ids_min;
    const auto * lineIn1_min = lineIn1 + ids_min;
    const auto * lineIn2_min = lineIn2 + ids_min;
    const auto * lineIn3_min = lineIn3 + ids_min;

    // block 4
    for (; i < ids_size - 3; i += 4) {
      // Load 4 values from weight vector
      // mmk0 = [wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ...]
      // mmk1 = [wl_2 wh_2 wl_3 wh_3  wl_2 wh_2 wl_3 wh_3  ...]
      const auto mmk0 = _mm256_set1_epi32(*(int32_t*)&k[i]);
      const auto mmk1 = _mm256_set1_epi32(*(int32_t*)&k[i + 2]);

      // Load 8 pixels (4 per line) from input lines 0 and 1:
      // source = [
      //   r0 g0 b0 a0  r1 g1 b1 a1  r2 g2 b2 a2  r3 g3 b3 a3
      //   R0 G0 B0 A0  R1 G1 B1 A1  R2 G2 B2 A2  R3 G3 B3 A3
      // ]
      auto source = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadu_si128((__m128i*)&lineIn0_min[i])),
          _mm_loadu_si128((__m128i*)&lineIn1_min[i]), 1);

      // Apply mask_low:
      //   [r0 g0 b0 a0  r1 g1 b1 a1  ... | R0 G0 B0 A0  R1 G1 B1 A1 ... ] ->
      //   [r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  a0 0 a1 0 | R0 0 R1 0  G0 0 G1 0  B0 0 B1 0  A0 0 A1 0]
      auto pix1 = _mm256_shuffle_epi8(source, mask_low);
      // Compute output value as C += w0 * C0 + w1 * C1 for each channel in 32-bit precision
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix1, mmk0));

      // Apply mask_high:
      //   [ ... r2 g2 b2 a2  r3 g3 b3 a3 | ... R2 G2 B2 A2  R3 G3 B3 A3 ] ->
      //   [r2 0 r3 0  g2 0 g3 0  b2 0 b3 0  a2 0 a3 0 | R2 0 R3 0  G2 0 G3 0  B2 0 B3 0  A2 0 A3 0]
      auto pix2 = _mm256_shuffle_epi8(source, mask_high);
      // Compute output value as C += w2 * C2 + w3 * C3 for each channel in 32-bit precision
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix2, mmk1));

      // Same as above to next lines 2 and 3:
      auto source2 = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadu_si128((__m128i*)&lineIn2_min[i])),
          _mm_loadu_si128((__m128i*)&lineIn3_min[i]), 1);
      auto pix3 = _mm256_shuffle_epi8(source2, mask_low);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix3, mmk0));
      auto pix4 = _mm256_shuffle_epi8(source2, mask_high);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix4, mmk1));
    }

    // block 2
    for (; i < ids_size - 1; i += 2) {
      // Load 2 values from weight vector
      // mmk = [wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ...]
      const auto mmk = _mm256_set1_epi32(*(int32_t*)&k[i]);

      // Load 4 pixels (2 per line) from input lines 0 and 1:
      // source = [
      //   r0 g0 b0 a0  r1 g1 b1 a1  0 0 0 0  0 0 0 0
      //   R0 G0 B0 A0  R1 G1 B1 A1  0 0 0 0  0 0 0 0
      // ]
      auto source1 = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadl_epi64((__m128i*)&lineIn0_min[i])),
          _mm_loadl_epi64((__m128i*)&lineIn1_min[i]), 1);

      // Apply mask_low:
      //   [r0 g0 b0 a0  r1 g1 b1 a1  ... | R0 G0 B0 A0  R1 G1 B1 A1 ... ] ->
      //   [r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  a0 0 a1 0 | R0 0 R1 0  G0 0 G1 0  B0 0 B1 0  A0 0 A1 0]
      auto pix1 = _mm256_shuffle_epi8(source1, mask_low);
      // Compute output value as C += w0 * C0 + w1 * C1 for each channel in 32-bit precision
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix1, mmk));

      // Same as above for lines 2 and 3:
      auto source2 = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadl_epi64((__m128i*)&lineIn2_min[i])),
          _mm_loadl_epi64((__m128i*)&lineIn3_min[i]), 1);
      auto pix2 = _mm256_shuffle_epi8(source2, mask_low);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix2, mmk));
    }

    // block 1
    for (; i < ids_size; i++) {
      // Load 1 value from weight vector
      // mmk = [wl_0 wh_0 0 0  wl_0 wh_0 0 0  ...]
      const auto mmk = _mm256_set1_epi32(k[i]);

      // Load 2 pixels (one per line) from input lines 0 and 1:
      // source = [
      //   r0 g0 b0 a0  0 0 0 0  0 0 0 0  0 0 0 0
      //   R0 G0 B0 A0  0 0 0 0  0 0 0 0  0 0 0 0
      // ]
      auto pix1 = _mm256_inserti128_si256(_mm256_castsi128_si256(
          mm_cvtepu8_epi32(&lineIn0_min[i])),
          mm_cvtepu8_epi32(&lineIn1_min[i]), 1);
      // Compute output value as C += w0 * C0 for each channel in 32-bit precision
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix1, mmk));

      // Same as above for lines 2 and 3
      auto pix2 = _mm256_inserti128_si256(_mm256_castsi128_si256(
          mm_cvtepu8_epi32(&lineIn2_min[i])),
          mm_cvtepu8_epi32(&lineIn3_min[i]), 1);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix2, mmk));
    }

    // Convert fixed point values back to integers (truncating)
    sss0 = _mm256_srai_epi32(sss0, coefs_precision);
    sss1 = _mm256_srai_epi32(sss1, coefs_precision);
    // Convert packed signed 32-bit integers to packed 16-bit integers using signed saturation
    // (a a a a b b b b c c c c d d d d) -> (a a b b c c d d 0 0 0 0 0 0 0 0)
    sss0 = _mm256_packs_epi32(sss0, zero);
    sss1 = _mm256_packs_epi32(sss1, zero);
    // Convert packed signed 16-bit integers to packed 8-bit integers using unsigned saturation
    // (a a b b c c d d) -> (a b c d 0 0 0 0)
    sss0 = _mm256_packus_epi16(sss0, zero);
    sss1 = _mm256_packus_epi16(sss1, zero);

    // Write the output into single uint32
    // (a b c d) -> x_uint32
    lineOut0[out_x] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss0, 0));
    lineOut1[out_x] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss0, 1));
    lineOut2[out_x] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss1, 0));
    lineOut3[out_x] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss1, 1));
  }
}

void ImagingResampleHorizontalConvolution8u(
    uint32_t* C10_RESTRICT lineOut,
    int64_t out_xsize,
    const uint32_t* C10_RESTRICT lineIn,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision) {

  // Interpolation horizontal pass processing only one vertical line.
  // - Input data format is RGBA with R,G,B,A being uint8, we can encode 4 values as a single uint32 value.
  // - We split the size of weight vector for a given output index as a sum:
  //   ids_size = num_blocks_8 * 8 + num_blocks_4 * 4 + num_blocks_2 * 2 + num_blocks_1
  // - We load and process 8 weights values in a loop ("block 8") then 4 weights and 2 weights values in
  // in another loops ("block 4" and "block 2") and finally we process 1 weight value in the final loop ("block 1").

  // Define various shuffling masks
  const auto kmask_low = _mm256_set_epi8(
      11, 10, 9, 8, 11, 10, 9, 8, 11, 10, 9, 8, 11, 10, 9, 8,
      3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0);
  const auto kmask_high = _mm256_set_epi8(
      15, 14, 13, 12, 15, 14, 13, 12, 15, 14, 13, 12, 15, 14, 13, 12,
      7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4);
  const auto kmask_hl = _mm256_set_epi8(
      7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4,
      3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0);

  const auto mask_low_c4 = _mm256_set_epi8(
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0,
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0);
  const auto mask_high_c4 = _mm256_set_epi8(
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8,
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8);
  const auto mask_hl_c4 = _mm256_set_epi8(
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8,
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0);
  const auto mask_low128_c4 = _mm_set_epi8(
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0);

  const auto mask_low = mask_low_c4;
  const auto mask_high = mask_high_c4;
  const auto mask_hl = mask_hl_c4;
  const auto mask_low128 = mask_low128_c4;

  // out_xsize = output width, out_x = output x index
  // ids_min is the input offset index corresponding to out_x
  // ids_size is the interpolation size for out_x

  const auto zero = _mm_setzero_si128();

  for (const auto out_x : c10::irange(out_xsize)) {
    __m128i sss;
    const auto ids_min = idx_ptr_xmin[out_x];
    const auto ids_size = idx_ptr_size[out_x];
    const auto * k = &kk[out_x * kmax];
    int64_t i = 0;

    const auto * lineIn_min = lineIn + ids_min;

    if (ids_size < 8) {
      sss = _mm_set1_epi32(1 << (coefs_precision - 1));
    } else {
      // Lower part will be added to higher, use only half of the error
      auto sss256 = _mm256_set1_epi32(1 << (coefs_precision - 2));

      // block 8
      for (; i < ids_size - 7; i += 8) {
        // Load 8 values from weight vector
        auto tmp = _mm_loadu_si128((__m128i*)&k[i]);
        // ksource = [
        //    wl_0 wh_0 wl_1 wh_1  wl_2 wh_2 wl_3 wh_3  wl_4 wh_4 wl_5 wh_5  wl_6 wh_6 wl_7 wh_7
        //    wl_0 wh_0 wl_1 wh_1  wl_2 wh_2 wl_3 wh_3  wl_4 wh_4 wl_5 wh_5  wl_6 wh_6 wl_7 wh_7
        // ]
        auto ksource = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        // Load 8 pixels from input:
        // source = [
        //    r0 g0 b0 a0  r1 g1 b1 a1  r2 g2 b2 a2  r3 g3 b3 a3
        //    r4 g4 b4 a4  r5 g5 b5 a5  r6 g6 b6 a6  r7 g7 b7 a7
        // ]
        auto source = _mm256_loadu_si256((__m256i*)&lineIn_min[i]);
        // Extract lower part of each lane, cast to epi16 and reoder RGBARGBA -> RRGGBBAA
        // pix1 = [
        //   r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  a0 0 a1 0
        //   r4 0 r5 0  g4 0 g5 0  b4 0 b5 0  a4 0 a5 0
        // ]
        auto pix1 = _mm256_shuffle_epi8(source, mask_low);
        // mmk1 = [
        //   wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ...  ...
        //   wl_4 wh_4 wl_5 wh_5  wl_4 wh_4 wl_5 wh_5  ...  ...
        // ]
        auto mmk1 = _mm256_shuffle_epi8(ksource, kmask_low);
        // Compute output value as
        //   C += w0 * C0 + w1 * C1
        //   C += w4 * C4 + w5 * C5 for each channel in 32-bit precision
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix1, mmk1));

        // Same as above for higher part of each lane
        auto pix2 = _mm256_shuffle_epi8(source, mask_high);
        auto mmk2 = _mm256_shuffle_epi8(ksource, kmask_high);
        // Compute output value as
        //    C += w2 * C2 + w3 * C3
        //    C += w6 * C6 + w7 * C7 for each channel in 32-bit precision
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix2, mmk2));
      }

      // block 4
      for (; i < ids_size - 3; i += 4) {
        // Load 4 values from weight vector
        auto tmp = _mm_loadl_epi64((__m128i *) &k[i]);
        // ksource = [
        //    wl_0 wh_0 wl_1 wh_1  wl_2 wh_2 wl_3 wh_3  0 0 0 0  0 0 0 0
        //    wl_0 wh_0 wl_1 wh_1  wl_2 wh_2 wl_3 wh_3  0 0 0 0  0 0 0 0
        // ]
        auto ksource = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        // Load 4 pixels from input line
        tmp = _mm_loadu_si128((__m128i*)&lineIn_min[i]);
        // source = [
        //   r0 g0 b0 a0  r1 g1 b1 a1  r2 g2 b2 a2  r3 g3 b3 a3
        //   r0 g0 b0 a0  r1 g1 b1 a1  r2 g2 b2 a2  r3 g3 b3 a3
        // ]
        auto source = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        // Cast source to epi16 and reorder RGBARGBA -> RRGGBBAA
        // pix = [
        //   r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  a0 0 a1 0
        //   r2 0 r3 0  g2 0 g3 0  b2 0 b3 0  a2 0 a3 0
        // ]
        auto pix = _mm256_shuffle_epi8(source, mask_hl);
        // mmk = [
        //   wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ... ...
        //   wl_2 wh_2 wl_3 wh_3  wl_2 wh_2 wl_3 wh_3  ... ...
        // ]
        auto mmk = _mm256_shuffle_epi8(ksource, kmask_hl);
        // Compute output value as
        //   C += w0 * C0 + w1 * C1
        //   C += w2 * C2 + w3 * C3 for each channel in 32-bit precision
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));
      }

      // Sum results between the lanes
      sss = _mm_add_epi32(
          _mm256_extracti128_si256(sss256, 0),
          _mm256_extracti128_si256(sss256, 1));
    }

    // block 2
    for (; i < ids_size - 1; i += 2) {
      // Load 2 values from weight vector
      // mmk = [wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ...]
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[i]);
      // Load 2 pixels from input line
      // source = [
      //   r0 g0 b0 a0  r1 g1 b1 a1  0 0 0 0  0 0 0 0
      // ]
      auto source = _mm_loadl_epi64((__m128i*)&lineIn_min[i]);
      // Cast source to epi16 and reorder RGBARGBA -> RRGGBBAA
      auto pix = _mm_shuffle_epi8(source, mask_low128);
      // Compute output value as C += w0 * C0 + w1 * C1 for each channel in 32-bit precision
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    // block 1
    for (; i < ids_size; i++) {
      // Load 1 value from weight vector
      // mmk = [wl_0 wh_0 0 0  wl_0 wh_0 0 0  ...]
      auto mmk = _mm_set1_epi32(k[i]);
      // Load one pixel from input line
      // pix = [
      //   r0 0 0 0  g0 0 0 0  b0 0 0 0  a0 0 0 0
      // ]
      auto pix = mm_cvtepu8_epi32(&lineIn_min[i]);
      // Compute output value as C += w0 * C0 for each channel in 32-bit precision
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    // Convert fixed point values back to integers (truncating)
    sss = _mm_srai_epi32(sss, coefs_precision);
    // Convert packed signed 32-bit integers to packed 16-bit integers using signed saturation
    // (a a a a b b b b c c c c d d d d) -> (a a b b c c d d 0 0 0 0 0 0 0 0)
    sss = _mm_packs_epi32(sss, zero);
    // Convert packed signed 16-bit integers to packed 8-bit integers using unsigned saturation
    // (a a b b c c d d) -> (a b c d 0 0 0 0)
    sss = _mm_packus_epi16(sss, zero);
    // Write the output into single uint32
    // (a b c d) -> x_uint32
    lineOut[out_x] = _mm_cvtsi128_si32(sss);
  }
}

void ImagingResampleVerticalConvolution8u(
    uint32_t* C10_RESTRICT lineOut,
    const uint32_t* C10_RESTRICT lineIn,
    int64_t xsize,
    int64_t ids_min,
    int64_t ids_size,
    const int16_t* k,
    unsigned int coefs_precision) {
  // Interpolation vertical pass processing one line.
  // - We process x-axis data with blocks of 8, 2 and 1
  // - We split the size of weight vector for a given output index as a sum: K = n * 2 + m.

  // xsize = output width, also equals to input width
  // ids_size = interpolation size
  // ids_min = input y start index

  const auto initial = _mm_set1_epi32(1 << (coefs_precision - 1));
  const auto initial_256 = _mm256_set1_epi32(1 << (coefs_precision - 1));
  const auto zero = _mm_setzero_si128();
  const auto zero_256 = _mm256_setzero_si256();

  int64_t xx = 0;
  // block 8
  for (; xx < xsize - 7; xx += 8) {
    auto sss0 = initial_256;
    auto sss1 = initial_256;
    auto sss2 = initial_256;
    auto sss3 = initial_256;
    int64_t i = 0;
    const auto * lineIn_min = lineIn + xx + ids_min;

    for (; i < ids_size - 1; i += 2) {
      // Load 2 values from weight vector
      auto mmk = _mm256_set1_epi32(*(int32_t*)&k[i]);

      // Load 8 pixels per line
      // source1 = [
      //    r0 g0 b0 a0  r1 g1 b1 a1  r2 g2 b2 a2  r3 g3 b3 a3
      //    r4 g4 b4 a4  r5 g5 b5 a5  r6 g6 b6 a6  r7 g7 b7 a7
      // ]
      auto source1 = _mm256_loadu_si256((__m256i*)(lineIn_min + i * xsize));
      auto source2 = _mm256_loadu_si256((__m256i*)(lineIn_min + (i + 1) * xsize));

      // Interleave source1 and source2 from the low half of each 128-bit lane
      // and cast the result to epi16
      // pix1 = [
      //    r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  a0 0 A0 0
      //    r1 0 R1 0  g1 0 G1 0  b1 0 B1 0  a1 0 A1 0
      // ]
      auto source_lo = _mm256_unpacklo_epi8(source1, source2);
      auto pix1 = _mm256_unpacklo_epi8(source_lo, zero_256);
      // Compute output value as
      //   C += w0 * c0 + w1 * C0
      //   C += w0 * c1 + w1 * C1 for each channel in 32-bit precision
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix1, mmk));

      // pix2 = [
      //    r2 0 R2 0  g2 0 G2 0  b2 0 B2 0  a2 0 A2 0
      //    r3 0 R3 0  g3 0 G3 0  b3 0 B3 0  a3 0 A3 0
      // ]
      auto pix2 = _mm256_unpackhi_epi8(source_lo, zero_256);
      // Compute output value as
      //   C += w0 * c2 + w1 * C2
      //   C += w0 * c3 + w1 * C3 for each channel in 32-bit precision
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix2, mmk));

      // Same as above for the high half of each 128-bit lane
      auto source_hi = _mm256_unpackhi_epi8(source1, source2);
      auto pix3 = _mm256_unpacklo_epi8(source_hi, zero_256);
      sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix3, mmk));
      auto pix4 = _mm256_unpackhi_epi8(source_hi, zero_256);
      sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix4, mmk));
    }
    // Same processing as above but with a single weight value
    for (; i < ids_size; i += 1) {
      auto mmk = _mm256_set1_epi32(k[i]);

      auto source1 = _mm256_loadu_si256((__m256i*)(lineIn_min + i * xsize));

      auto source_lo = _mm256_unpacklo_epi8(source1, zero_256);
      auto pix1 = _mm256_unpacklo_epi8(source_lo, zero_256);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix1, mmk));
      auto pix2 = _mm256_unpackhi_epi8(source_lo, zero_256);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix2, mmk));

      auto source_hi = _mm256_unpackhi_epi8(source1, zero_256);
      auto pix3 = _mm256_unpacklo_epi8(source_hi, _mm256_setzero_si256());
      sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix3, mmk));
      auto pix4 = _mm256_unpackhi_epi8(source_hi, _mm256_setzero_si256());
      sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix4, mmk));
    }
    // Convert fixed point values back to integers (truncating)
    sss0 = _mm256_srai_epi32(sss0, coefs_precision);
    sss1 = _mm256_srai_epi32(sss1, coefs_precision);
    sss2 = _mm256_srai_epi32(sss2, coefs_precision);
    sss3 = _mm256_srai_epi32(sss3, coefs_precision);
    // Convert packed signed 32-bit integers to packed 16-bit integers using signed saturation
    // (a a a a b b b b c c c c d d d d) -> (a a b b c c d d)
    sss0 = _mm256_packs_epi32(sss0, sss1);
    sss2 = _mm256_packs_epi32(sss2, sss3);
    // Convert packed signed 16-bit integers to packed 8-bit integers using unsigned saturation
    // (a a b b c c d d) -> (a b c d)
    sss0 = _mm256_packus_epi16(sss0, sss2);

    // Store 8 pixels to the output
    _mm256_storeu_si256((__m256i*)&lineOut[xx], sss0);
  }

  // block 2
  for (; xx < xsize - 1; xx += 2) {
    auto sss0 = initial;
    auto sss1 = initial;
    int64_t i = 0;
    const auto * lineIn_min = lineIn + xx + ids_min;

    for (; i < ids_size - 1; i += 2) {
      // Load 2 values from weight vector
      // mmk = [wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ... ]
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[i]);

      // Load 2 pixels per line
      // source1 = [
      //    r0 g0 b0 a0  r1 g1 b1 a1  0 0 0 0  0 0 0 0
      // ]
      auto source1 = _mm_loadl_epi64((__m128i*)(lineIn_min + i * xsize));
      auto source2 = _mm_loadl_epi64((__m128i*)(lineIn_min + (i + 1) * xsize));

      // Interleave source1 and source2 and cast the result to epi16
      // pix = [
      //    r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  a0 0 A0 0
      // ]
      auto source = _mm_unpacklo_epi8(source1, source2);
      auto pix = _mm_unpacklo_epi8(source, zero);
      // Compute output value as C += w0 * c0 + w1 * C0 for each channel in 32-bit precision
      sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
      // pix = [
      //    r1 0 R1 0  g1 0 G1 0  b1 0 B1 0  a1 0 A1 0
      // ]
      pix = _mm_unpackhi_epi8(source, zero);
      // Compute output value as C += w0 * c1 + w1 * C1 for each channel in 32-bit precision
      sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
    }
    // Same processing as above but with a single weight value
    for (; i < ids_size; i += 1) {
      auto mmk = _mm_set1_epi32(k[i]);

      auto source1 = _mm_loadl_epi64((__m128i*)(lineIn_min + i * xsize));

      auto source = _mm_unpacklo_epi8(source1, zero);
      auto pix1 = _mm_unpacklo_epi8(source, zero);
      sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix1, mmk));
      auto pix2 = _mm_unpackhi_epi8(source, zero);
      sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix2, mmk));
    }
    // Convert fixed point values back to integers (truncating)
    sss0 = _mm_srai_epi32(sss0, coefs_precision);
    sss1 = _mm_srai_epi32(sss1, coefs_precision);
    // Convert packed signed 32-bit integers to packed 16-bit integers using signed saturation
    // (a a a a b b b b c c c c d d d d) -> (a a b b c c d d)
    sss0 = _mm_packs_epi32(sss0, sss1);
    // Convert packed signed 16-bit integers to packed 8-bit integers using unsigned saturation
    // (a a b b c c d d) -> (a b c d)
    sss0 = _mm_packus_epi16(sss0, sss0);
    // Store 2 pixels to the output
    _mm_storel_epi64((__m128i*)&lineOut[xx], sss0);
  }

  // block 1
  for (; xx < xsize; xx++) {
    auto sss = initial;
    int64_t i = 0;
    const auto * lineIn_min = lineIn + xx + ids_min;

    for (; i < ids_size - 1; i += 2) {
      // Load 2 values from weight vector
      // mmk = [wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ... ]
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[i]);

      // Load one pixel per line
      // source1 = [
      //    r0 g0 b0 a0  0 0 0 0  0 0 0 0  0 0 0 0
      // ]
      auto source1 = _mm_cvtsi32_si128(*(int32_t*)(lineIn_min + i * xsize));
      auto source2 = _mm_cvtsi32_si128(*(int32_t*)(lineIn_min + (i + 1) * xsize));

      // Interleave source1 and source2 and cast the result to epi16
      // pix = [
      //    r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  a0 0 A0 0
      // ]
      auto source = _mm_unpacklo_epi8(source1, source2);
      auto pix = _mm_unpacklo_epi8(source, zero);
      // Compute output value as C += w0 * c0 + w1 * C0 for each channel in 32-bit precision
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }
    // Same processing as above but with a single weight value
    for (; i < ids_size; i++) {
      auto mmk = _mm_set1_epi32(k[i]);
      auto pix = mm_cvtepu8_epi32(lineIn_min + i * xsize);
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    // Convert fixed point values back to integers (truncating)
    sss = _mm_srai_epi32(sss, coefs_precision);
    // Convert packed signed 32-bit integers to packed 16-bit integers using signed saturation
    // (a a a a b b b b c c c c d d d d) -> (a a b b c c d d)
    sss = _mm_packs_epi32(sss, zero);
    // Convert packed signed 16-bit integers to packed 8-bit integers using unsigned saturation
    // (a a b b c c d d) -> (a b c d)
    sss = _mm_packus_epi16(sss, zero);
    // Store one pixel to the output
    lineOut[xx] = _mm_cvtsi128_si32(sss);
  }
}

} // anonymous namespace
#endif // CPU_CAPABILITY_AVX2
