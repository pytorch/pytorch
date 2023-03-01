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

static __m128i inline mm_cvtepu8_epi32(const uint8_t* C10_RESTRICT ptr) {
  return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)ptr));
}

// TODO: We may want to hard-code an unrolled version for the case where
// num_channels=3 to hint the compiler to vectorize this (looks at original
// PIL-SIMD's code).
at::Tensor unpack_rgb(const at::Tensor& packed_tensor) {
  // Convert a "packed" tensor (typically RGBRGBRGB if channels_last) into
  // RGBARGBARGBA format where A is hard-coded to 255. Each pixel is encoded
  // into as 32bits. This generalizes to num_channels <= 4 and also works for
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
    uint8_t* C10_RESTRICT lineOut0,
    uint8_t* C10_RESTRICT lineOut1,
    uint8_t* C10_RESTRICT lineOut2,
    uint8_t* C10_RESTRICT lineOut3,
    const uint8_t* C10_RESTRICT lineIn0,
    const uint8_t* C10_RESTRICT lineIn1,
    const uint8_t* C10_RESTRICT lineIn2,
    const uint8_t* C10_RESTRICT lineIn3,
    int64_t xsize,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    int64_t num_channels);

void ImagingResampleHorizontalConvolution8u(
    uint8_t* C10_RESTRICT lineOut,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t xsize,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    int64_t num_channels);

void ImagingResampleVerticalConvolution8u(
    uint8_t* C10_RESTRICT lineOut,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t xsize,
    int64_t ymin,
    int64_t ymax,
    const int16_t* k,
    unsigned int coefs_precision,
    int64_t num_channels);

template<int num_channels>
void ImagingResampleHorizontal(
    const at::Tensor & unpacked_output,
    const at::Tensor & unpacked_input,
    int ksize,
    const std::vector<at::Tensor>& horiz_indices_weights,
    unsigned int horiz_weights_precision) {
  // TODO: we may want to merge that into the fallback code (currently called
  // basic_loop_aa_horizontal<uint8_t>)
  // Although this may not be needed if / when we port all this code to use
  // Vec.h since this would potentially give us another fall-back implem

  const int16_t* kk = (int16_t*)(horiz_indices_weights[3].data_ptr<double>());

  auto xout = unpacked_output.size(2);
  auto yout = unpacked_output.size(1);
  auto xin = unpacked_input.size(2);
  // const auto num_channels = unpacked_input.size(0);
  // TORCH_INTERNAL_ASSERT(num_channels == 3 || num_channels == 4);
  TORCH_INTERNAL_ASSERT(num_channels == unpacked_input.size(0));

  const int64_t* idx_ptr_xmin = horiz_indices_weights[0].data_ptr<int64_t>();
  const int64_t* idx_ptr_size = horiz_indices_weights[1].data_ptr<int64_t>();

  uint8_t* unpacked_output_p = unpacked_output.data_ptr<uint8_t>();
  const uint8_t* unpacked_input_p = unpacked_input.data_ptr<uint8_t>();

  int64_t yy = 0;
  auto xout_stride = xout * num_channels;
  auto xin_stride = xin * num_channels;
  for (; yy < yout - 3; yy += 4) {
    ImagingResampleHorizontalConvolution8u4x(
        unpacked_output_p + yy * xout_stride,
        unpacked_output_p + (yy + 1) * xout_stride,
        unpacked_output_p + (yy + 2) * xout_stride,
        unpacked_output_p + (yy + 3) * xout_stride,
        unpacked_input_p + yy * xin_stride,
        unpacked_input_p + (yy + 1) * xin_stride,
        unpacked_input_p + (yy + 2) * xin_stride,
        unpacked_input_p + (yy + 3) * xin_stride,
        xout,
        idx_ptr_xmin,
        idx_ptr_size,
        kk,
        ksize,
        horiz_weights_precision,
        num_channels);
  }
  for (; yy < yout; yy++) {
    ImagingResampleHorizontalConvolution8u(
        unpacked_output_p + yy * xout_stride,
        unpacked_input_p + yy * xin_stride,
        xout,
        idx_ptr_xmin,
        idx_ptr_size,
        kk,
        ksize,
        horiz_weights_precision,
        num_channels);
  }
}

template<int num_channels>
void ImagingResampleVertical(
    const at::Tensor & unpacked_output,
    const at::Tensor & unpacked_input,
    int ksize,
    const std::vector<at::Tensor>& vert_indices_weights,
    unsigned int vert_weights_precision) {
  // TODO: we may want to merge that into the fallback code (currently called
  // basic_loop_aa_vertical<uint8_t>)
  // Although this may not be needed if / when we port all this code to use
  // Vec.h since this would potentially give us another fall-back implem
  const int16_t* kk = (int16_t*)(vert_indices_weights[3].data_ptr<double>());

  const int64_t* idx_ptr_xmin = vert_indices_weights[0].data_ptr<int64_t>();
  const int64_t* idx_ptr_size = vert_indices_weights[1].data_ptr<int64_t>();

  uint8_t* unpacked_output_p = unpacked_output.data_ptr<uint8_t>();
  const uint8_t* unpacked_input_p = unpacked_input.data_ptr<uint8_t>();

  auto xout = unpacked_output.size(2);
  auto yout = unpacked_output.size(1);
  // const auto num_channels = unpacked_input.size(0);
  // TORCH_INTERNAL_ASSERT(num_channels == 3 || num_channels == 4);
  TORCH_INTERNAL_ASSERT(num_channels == unpacked_input.size(0));

  auto xout_stride = xout * num_channels;
  for (const auto yy : c10::irange(yout)) {
    const auto* k = &kk[yy * ksize];

    auto ymin = idx_ptr_xmin[yy];
    auto ymax = idx_ptr_size[yy];
    ImagingResampleVerticalConvolution8u(
        unpacked_output_p + yy * xout_stride,
        unpacked_input_p,
        xout,
        ymin,
        ymax,
        k,
        vert_weights_precision,
        num_channels);
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
            /*stride=*/1,
            /*ndims=*/4,
            /*reshape_dim=*/interp_dim,
            /*align_corners=*/align_corners,
            /*opt_scale=*/scales[interp_dim - 2],
            /*antialias=*/antialias,
            /*align_i32=*/true);
  }

  bool is_rgb_or_rgba = (num_channels == 3 || num_channels == 4) && input.is_contiguous(at::MemoryFormat::ChannelsLast);

  at::Tensor buffer_horiz, buffer_vert;
  if (need_horizontal && !(is_rgb_or_rgba && !need_vertical)) {
    auto c = (is_rgb_or_rgba) ? num_channels : 4;
    buffer_horiz = at::empty({c, yin, xout}, input.options());
  }
  if (need_vertical && !is_rgb_or_rgba) {
    auto c = (is_rgb_or_rgba) ? num_channels : 4;
    buffer_vert = at::empty({c, yout, xout}, input.options());
  }

  // TODO: The unpack / pack operations create a copy of the original input and
  // output tensor. There should be a way to avoid these copies by instead
  // modifying the low-level kernels. Or maybe at least avoid copying the entire
  // tensors and just copy part of them (line by line).
  for (const auto i : c10::irange(batch_size)) {

    at::Tensor unpacked_input = (is_rgb_or_rgba) ? input[i] : unpack_rgb(input[i]);
    at::Tensor unpacked_output;

    if (need_horizontal) {
      at::Tensor unpacked_output_temp = (is_rgb_or_rgba && !need_vertical) ? output[i] : buffer_horiz;

      if (is_rgb_or_rgba && num_channels == 3) {
        ImagingResampleHorizontal<3>(
            unpacked_output_temp,
            unpacked_input,
            ksize_horiz,
            horiz_indices_weights,
            horiz_weights_precision);
      } else {
        ImagingResampleHorizontal<4>(
            unpacked_output_temp,
            unpacked_input,
            ksize_horiz,
            horiz_indices_weights,
            horiz_weights_precision);
      }
      unpacked_output = unpacked_input = unpacked_output_temp;
    }
    if (need_vertical) {
      unpacked_output = (is_rgb_or_rgba) ? output[i] : buffer_vert;

      if (is_rgb_or_rgba && num_channels == 3) {
        ImagingResampleVertical<3>(
            unpacked_output,
            unpacked_input,
            ksize_vert,
            vert_indices_weights,
            vert_weights_precision);
      } else {
        ImagingResampleVertical<4>(
            unpacked_output,
            unpacked_input,
            ksize_vert,
            vert_indices_weights,
            vert_weights_precision);
      }
    }

    TORCH_INTERNAL_ASSERT(unpacked_output.defined());

    if (!is_rgb_or_rgba) {
      pack_rgb(unpacked_output, output[i]);
    }
  }
}

void ImagingResampleHorizontalConvolution8u4x(
    uint8_t* C10_RESTRICT lineOut0,
    uint8_t* C10_RESTRICT lineOut1,
    uint8_t* C10_RESTRICT lineOut2,
    uint8_t* C10_RESTRICT lineOut3,
    const uint8_t* C10_RESTRICT lineIn0,
    const uint8_t* C10_RESTRICT lineIn1,
    const uint8_t* C10_RESTRICT lineIn2,
    const uint8_t* C10_RESTRICT lineIn3,
    int64_t xsize,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    int64_t num_channels) {

  // Define shuffling masks (low/high) for num_channels 4 and 3
  const __m256i masks_low_high_c4[2] = {
    _mm256_set_epi8(
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0,
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0
    ),
    _mm256_set_epi8(
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8,
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8
    )
  };
  const __m256i masks_low_high_c3[2] = {
    _mm256_set_epi8(
      -1, -1, -1, -1, -1, 5, -1, 2, -1, 4, -1, 1, -1, 3, -1, 0,
      -1, -1, -1, -1, -1, 5, -1, 2, -1, 4, -1, 1, -1, 3, -1, 0
    ),
    _mm256_set_epi8(
      -1, -1, -1, -1, -1, 11, -1, 8, -1, 10, -1, 7, -1, 9, -1, 6,
      -1, -1, -1, -1, -1, 11, -1, 8, -1, 10, -1, 7, -1, 9, -1, 6
    )
  };

  const auto mask_low = (num_channels == 3) ? masks_low_high_c3[0] : masks_low_high_c4[0];
  const auto mask_high = (num_channels == 3) ? masks_low_high_c3[1] : masks_low_high_c4[1];

  const auto mask_ch = _mm256_set_epi8(
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 1, 0,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 1, 0
  );

  const auto stride = num_channels * 1;  // num channels * sizeof(uint8)

  TORCH_INTERNAL_ASSERT(stride == 3 || stride == 4);

  // Precompute xmax limits for block 4 and block 2
  // lineIn0 + stride * (x + xmin) + 16 <= lineIn0 + stride * (xmax + xmin)
  // --> x <= xmax - 16.0 / stride
  // --> x < xmax + 1 - int(16.0 / stride)
  const auto b4_delta = int(16.0 / stride) - 1;

  // lineIn0 + stride * (x + xmin) + 8 <= lineIn0 + stride * (xmax + xmin)
  // --> x <= xmax - 8.0 / stride
  // --> x < xmax + 1 - int(8.0 / stride)
  const auto b2_delta = int(8.0 / stride) - 1;

  // xsize = output width, xx = output x index
  // xmax = interpolation size, x = interpolation index (horizontal <-> x dimension)
  // xmin = input x start index corresponding to output x index (xx)

  const auto zero = _mm256_setzero_si256();
  const auto initial = _mm256_set1_epi32(1 << (coefs_precision - 1));

  for (const auto xx : c10::irange(xsize)) {
    const auto xmin = idx_ptr_xmin[xx];
    const auto xmax = idx_ptr_size[xx];
    const auto * k = &kk[xx * kmax];
    int64_t x = 0;

    auto sss0 = initial;
    auto sss1 = initial;

    for (; x < xmax - b4_delta; x += 4) {
      auto mmk0 = _mm256_set1_epi32(*(int32_t*)&k[x]);
      auto mmk1 = _mm256_set1_epi32(*(int32_t*)&k[x + 2]);

      auto source = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadu_si128((__m128i *) (lineIn0 + stride * (x + xmin)))),
          _mm_loadu_si128((__m128i *) (lineIn1 + stride * (x + xmin))), 1);

      auto pix = _mm256_shuffle_epi8(source, mask_low);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk0));
      pix = _mm256_shuffle_epi8(source, mask_high);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk1));

      source = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadu_si128((__m128i *) (lineIn2 + stride * (x + xmin)))),
          _mm_loadu_si128((__m128i *) (lineIn3 + stride * (x + xmin))), 1);
      pix = _mm256_shuffle_epi8(source, mask_low);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk0));
      pix = _mm256_shuffle_epi8(source, mask_high);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk1));
    }

    for (; x < xmax - b2_delta; x += 2) {
      auto mmk = _mm256_set1_epi32(*(int32_t*)&k[x]);

      auto pix = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadu_si64(lineIn0 + stride * (x + xmin))),
          _mm_loadu_si64(lineIn1 + stride * (x + xmin)), 1);
      pix = _mm256_shuffle_epi8(pix, mask_low);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));

      pix = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadu_si64(lineIn2 + stride * (x + xmin))),
          _mm_loadu_si64(lineIn3 + stride * (x + xmin)), 1);
      pix = _mm256_shuffle_epi8(pix, mask_low);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));
    }

    for (; x < xmax - 1; x++) {
      auto mmk = _mm256_set1_epi32(k[x]);
      auto pix = _mm256_inserti128_si256(_mm256_castsi128_si256(
          mm_cvtepu8_epi32(lineIn0 + stride * (x + xmin))),
          mm_cvtepu8_epi32(lineIn1 + stride * (x + xmin)), 1);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));

      pix = _mm256_inserti128_si256(_mm256_castsi128_si256(
          mm_cvtepu8_epi32(lineIn2 + stride * (x + xmin))),
          mm_cvtepu8_epi32(lineIn3 + stride * (x + xmin)), 1);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));
    }

    if (x == xmax - 1) {
      // last element x = xmax - 1
      auto mmk = _mm256_set1_epi32(k[x]);
      __m256i pix;
      __m128i p0, p1;
      if (num_channels == 3) {
        uint8_t output0[4];
        uint8_t output1[4];
        std::memcpy(output0, lineIn0 + stride * (x + xmin), 3);
        std::memcpy(output1, lineIn1 + stride * (x + xmin), 3);
        p0 = mm_cvtepu8_epi32(output0);
        p1 = mm_cvtepu8_epi32(output1);
      } else {
        p0 = mm_cvtepu8_epi32(lineIn0 + stride * (x + xmin));
        p1 = mm_cvtepu8_epi32(lineIn1 + stride * (x + xmin));
      }
      pix = _mm256_inserti128_si256(_mm256_castsi128_si256(p0), p1, 1);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));

      if (num_channels == 3) {
        uint8_t output0[4];
        uint8_t output1[4];
        std::memcpy(output0, lineIn2 + stride * (x + xmin), 3);
        std::memcpy(output1, lineIn3 + stride * (x + xmin), 3);
        p0 = mm_cvtepu8_epi32(output0);
        p1 = mm_cvtepu8_epi32(output1);
      } else {
        p0 = mm_cvtepu8_epi32(lineIn2 + stride * (x + xmin));
        p1 = mm_cvtepu8_epi32(lineIn3 + stride * (x + xmin));
      }
      pix = _mm256_inserti128_si256(_mm256_castsi128_si256(p0), p1, 1);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));
    }

    sss0 = _mm256_srai_epi32(sss0, coefs_precision);
    sss1 = _mm256_srai_epi32(sss1, coefs_precision);
    sss0 = _mm256_packs_epi32(sss0, zero);
    sss1 = _mm256_packs_epi32(sss1, zero);
    sss0 = _mm256_packus_epi16(sss0, zero);
    sss1 = _mm256_packus_epi16(sss1, zero);
    if (num_channels == 3) {
        sss0 = _mm256_shuffle_epi8(sss0, mask_ch);
        sss1 = _mm256_shuffle_epi8(sss1, mask_ch);
    }

    auto o0 = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss0, 0));
    auto o1 = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss0, 1));
    auto o2 = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss1, 0));
    auto o3 = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss1, 1));
    if (num_channels == 3 && C10_UNLIKELY(stride * xx + 4 >= xsize * stride)) {
        std::memcpy(lineOut0 + stride * xx, (unsigned char *) &o0, num_channels);
        std::memcpy(lineOut1 + stride * xx, (unsigned char *) &o1, num_channels);
        std::memcpy(lineOut2 + stride * xx, (unsigned char *) &o2, num_channels);
        std::memcpy(lineOut3 + stride * xx, (unsigned char *) &o3, num_channels);
    } else {
      *(uint32_t *)(lineOut0 + stride * xx) = o0;
      *(uint32_t *)(lineOut1 + stride * xx) = o1;
      *(uint32_t *)(lineOut2 + stride * xx) = o2;
      *(uint32_t *)(lineOut3 + stride * xx) = o3;
    }
  }
}

void ImagingResampleHorizontalConvolution8u(
    uint8_t* C10_RESTRICT lineOut,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t xsize,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    int64_t num_channels) {

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

  const auto mask_ch = _mm_set_epi8(
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 1, 0);

  const __m256i masks_low_high_c4[2] = {
    _mm256_set_epi8(
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0,
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0
    ),
    _mm256_set_epi8(
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8,
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8
    )
  };
  const __m256i masks_low_high_c3[2] = {
    _mm256_set_epi8(
      -1, -1, -1, -1, -1, 5, -1, 2, -1, 4, -1, 1, -1, 3, -1, 0,
      -1, -1, -1, -1, -1, 5, -1, 2, -1, 4, -1, 1, -1, 3, -1, 0
    ),
    _mm256_set_epi8(
      -1, -1, -1, -1, -1, 11, -1, 8, -1, 10, -1, 7, -1, 9, -1, 6,
      -1, -1, -1, -1, -1, 11, -1, 8, -1, 10, -1, 7, -1, 9, -1, 6
    )
  };
  const __m256i masks_lh_c3_c4[2] = {
    _mm256_set_epi8(
      -1, -1, -1, -1, -1, 11, -1, 8, -1, 10, -1, 7, -1, 9, -1, 6,
      -1, -1, -1, -1, -1, 5, -1, 2, -1, 4, -1, 1, -1, 3, -1, 0
    ),
    _mm256_set_epi8(
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8,
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0
    )
  };

  const __m128i masks_low128_c3_c4[2] = {
    _mm_set_epi8(
      -1, -1, -1, -1, -1, 5, -1, 2, -1, 4, -1, 1, -1, 3, -1, 0
    ),
    _mm_set_epi8(
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0
    )
  };

  const auto mask_low = (num_channels == 3) ? masks_low_high_c3[0] : masks_low_high_c4[0];
  const auto mask_high = (num_channels == 3) ? masks_low_high_c3[1] : masks_low_high_c4[1];
  const auto mask_hl = (num_channels == 3) ? masks_lh_c3_c4[0] : masks_lh_c3_c4[1];
  const auto mask_low128 = (num_channels == 3) ? masks_low128_c3_c4[0] : masks_low128_c3_c4[1];

  // xsize = output width, xx = output x index
  // xmax = interpolation size, x = interpolation index (horizontal <-> x dimension)
  // xmin = input x start index corresponding to output x index (xx)

  const auto stride = num_channels * 1;  // num channels * sizeof(uint8)
  const auto zero = _mm_setzero_si128();

  TORCH_INTERNAL_ASSERT(stride == 3 || stride == 4);

  // Precompute xmax limits for block 8, block 4 and block 2
  // lineIn + stride * (x + xmin) + 32 <= lineIn + stride * (xmax + xmin)
  // --> x <= xmax - 32.0 / stride
  // --> x < xmax + 1 - int(32.0 / stride)
  const auto b8_delta = int(32.0 / stride) - 1;

  // lineIn + stride * (x + xmin) + 16 <= lineIn0 + stride * (xmax + xmin)
  // --> x <= xmax - 16.0 / stride
  // --> x < xmax + 1 - int(16.0 / stride)
  const auto b4_delta = int(16.0 / stride) - 1;

  // lineIn0 + stride * (x + xmin) + 8 <= lineIn0 + stride * (xmax + xmin)
  // --> x <= xmax - 8.0 / stride
  // --> x < xmax + 1 - int(8.0 / stride)
  const auto b2_delta = int(8.0 / stride) - 1;

  for (const auto xx : c10::irange(xsize)) {
    __m128i sss;
    const auto xmin = idx_ptr_xmin[xx];
    const auto xmax = idx_ptr_size[xx];
    const auto * k = &kk[xx * kmax];
    int64_t x = 0;

    if (xmax < 8) {
      sss = _mm_set1_epi32(1 << (coefs_precision - 1));
    } else {
      // Lower part will be added to higher, use only half of the error
      auto sss256 = _mm256_set1_epi32(1 << (coefs_precision - 2));

      for (; x < xmax - b8_delta; x += 8) {
        auto tmp = _mm_loadu_si128((__m128i*)&k[x]);
        auto ksource = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        auto source = _mm256_inserti128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *) (lineIn + stride * (x + 0 + xmin)))),
            _mm_loadu_si128((__m128i *) (lineIn + stride * (x + 4 + xmin))), 1);

        auto pix = _mm256_shuffle_epi8(source, mask_low);
        auto mmk = _mm256_shuffle_epi8(ksource, kmask_low);
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));

        pix = _mm256_shuffle_epi8(source, mask_high);
        mmk = _mm256_shuffle_epi8(ksource, kmask_high);
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));
      }

      for (; x < xmax - b4_delta; x += 4) {
        auto tmp = _mm_loadu_si64(&k[x]);
        auto ksource = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        tmp = _mm_loadu_si128((__m128i *) (lineIn + stride * (x + xmin)));
        auto source = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        auto pix = _mm256_shuffle_epi8(source, mask_hl);
        auto mmk = _mm256_shuffle_epi8(ksource, kmask_hl);
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));
      }

      sss = _mm_add_epi32(
          _mm256_extracti128_si256(sss256, 0),
          _mm256_extracti128_si256(sss256, 1));
    }

    for (; x < xmax - b2_delta; x += 2) {
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[x]);
      auto source = _mm_loadu_si64(lineIn + stride * (x + xmin));
      auto pix = _mm_shuffle_epi8(source, mask_low128);
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    for (; x < xmax - 1; x++) {
      auto mmk = _mm_set1_epi32(k[x]);
      auto pix = mm_cvtepu8_epi32(lineIn + stride * (x + xmin));
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    if (x == xmax - 1) {
      // last element x = xmax - 1
      auto mmk = _mm_set1_epi32(k[x]);
      __m128i pix;
      auto p = lineIn + stride * (x + xmin);
      if (num_channels == 3) {
        unsigned char output[4];
        std::memcpy(output, p, 3);
        pix = mm_cvtepu8_epi32(output);
      } else {
        pix = mm_cvtepu8_epi32(p);
      }
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }
    sss = _mm_srai_epi32(sss, coefs_precision);
    sss = _mm_packs_epi32(sss, zero);
    sss = _mm_packus_epi16(sss, zero);
    if (num_channels == 3) {
      sss = _mm_shuffle_epi8(sss, mask_ch);
    }

    auto o = _mm_cvtsi128_si32(sss);
    if (num_channels == 3 && C10_UNLIKELY(stride * xx + 4 >= xsize * stride)) {
      std::memcpy(lineOut + stride * xx, (unsigned char *) &o, num_channels);
    } else {
      *(uint32_t *)(lineOut + stride * xx) = o;
    }
  }
}

void ImagingResampleVerticalConvolution8u(
    uint8_t* C10_RESTRICT lineOut,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t xsize,
    int64_t ymin,
    int64_t ymax,
    const int16_t* k,
    unsigned int coefs_precision,
    int64_t num_channels) {

  // xsize = output width and input width, xx = output x index
  // ymax = interpolation size, y = interpolation index (vertical <-> y dimension)
  // ymin = input y start index

  const auto stride = num_channels * 1;  // num channels * sizeof(uint8)

  TORCH_INTERNAL_ASSERT(stride == 3 || stride == 4);

  const int64_t data_size = xsize * stride;
  const int64_t data_stride = stride;
  constexpr auto vec_size = 256 / 8;

  const auto initial = _mm_set1_epi32(1 << (coefs_precision - 1));
  const auto initial_256 = _mm256_set1_epi32(1 << (coefs_precision - 1));
  const auto zero = _mm_setzero_si128();
  const auto zero_256 = _mm256_setzero_si256();

  const auto mask_ch = _mm_set_epi8(
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 1, 0);

  int64_t i = 0;
  const auto b8_usable_vec_stride = (vec_size / data_stride) * data_stride;
  for (; i < data_size - vec_size; i += b8_usable_vec_stride) {
    auto sss0 = initial_256;
    auto sss1 = initial_256;
    auto sss2 = initial_256;
    auto sss3 = initial_256;
    int64_t y = 0;
    for (; y < ymax - 1; y += 2) {
      // Load two coefficients at once
      auto mmk = _mm256_set1_epi32(*(int32_t*)&k[y]);

      // Load 2 lines
      auto source1 =
          _mm256_loadu_si256((__m256i*)(lineIn + i + data_size * (y + ymin)));
      auto source2 =
          _mm256_loadu_si256((__m256i*)(lineIn + i + data_size * (y + 1 + ymin)));

      auto source = _mm256_unpacklo_epi8(source1, source2);
      auto pix = _mm256_unpacklo_epi8(source, zero_256);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));
      pix = _mm256_unpackhi_epi8(source, zero_256);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));

      source = _mm256_unpackhi_epi8(source1, source2);
      pix = _mm256_unpacklo_epi8(source, zero_256);
      sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix, mmk));
      pix = _mm256_unpackhi_epi8(source, zero_256);
      sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix, mmk));
    }
    for (; y < ymax; y += 1) {
      auto mmk = _mm256_set1_epi32(k[y]);

      auto source1 = _mm256_loadu_si256((__m256i*)(lineIn + i + data_size * (y + ymin)));

      auto source = _mm256_unpacklo_epi8(source1, zero_256);
      auto pix = _mm256_unpacklo_epi8(source, zero_256);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));
      pix = _mm256_unpackhi_epi8(source, zero_256);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));

      source = _mm256_unpackhi_epi8(source1, zero_256);
      pix = _mm256_unpacklo_epi8(source, zero_256);
      sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix, mmk));
      pix = _mm256_unpackhi_epi8(source, zero_256);
      sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix, mmk));
    }
    sss0 = _mm256_srai_epi32(sss0, coefs_precision);
    sss1 = _mm256_srai_epi32(sss1, coefs_precision);
    sss2 = _mm256_srai_epi32(sss2, coefs_precision);
    sss3 = _mm256_srai_epi32(sss3, coefs_precision);

    sss0 = _mm256_packs_epi32(sss0, sss1);
    sss2 = _mm256_packs_epi32(sss2, sss3);
    sss0 = _mm256_packus_epi16(sss0, sss2);

    // Stores 32 bytes
    _mm256_storeu_si256((__m256i*)(lineOut + i), sss0);
  }

  const auto b2_usable_vec_stride = (8 / data_stride) * data_stride;
  // TODO: can we make b2_usable_vec_stride as (16 / data_stride) * data_stride ?
  for (; i < data_size - vec_size / 4; i += b2_usable_vec_stride) {
    auto sss0 = initial; // left row
    auto sss1 = initial; // right row
    int64_t y = 0;
    for (; y < ymax - 1; y += 2) {
      // Load two coefficients at once
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[y]);

      // Load 2 lines
      auto source1 = _mm_loadu_si64(lineIn + i + data_size * (y + ymin));
      auto source2 = _mm_loadu_si64(lineIn + i + data_size * (y + 1 + ymin));

      auto source = _mm_unpacklo_epi8(source1, source2);
      auto pix = _mm_unpacklo_epi8(source, zero);
      sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
      pix = _mm_unpackhi_epi8(source, zero);
      sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
    }
    for (; y < ymax; y += 1) {
      auto mmk = _mm_set1_epi32(k[y]);

      auto source1 = _mm_loadu_si64(lineIn + i + data_size * (y + ymin));
      auto source = _mm_unpacklo_epi8(source1, zero);
      auto pix = _mm_unpacklo_epi8(source, zero);
      sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
      pix = _mm_unpackhi_epi8(source, zero);
      sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
    }
    sss0 = _mm_srai_epi32(sss0, coefs_precision);
    sss1 = _mm_srai_epi32(sss1, coefs_precision);

    sss0 = _mm_packs_epi32(sss0, sss1);
    sss0 = _mm_packus_epi16(sss0, sss0);

    _mm_storel_epi64((__m128i*)(lineOut + i), sss0);
  }

  const auto b1_usable_vec_stride = (4 / data_stride) * data_stride;
  for (; i < data_size - 4; i += b1_usable_vec_stride) {
    auto sss = initial;
    int64_t y = 0;
    for (; y < ymax - 1; y += 2) {
      // Load two coefficients at once
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[y]);

      // Load 2 lines
      auto source1 = _mm_cvtsi32_si128(*(int32_t*)(lineIn + i + data_size * (y + ymin)));
      auto source2 = _mm_cvtsi32_si128(*(int32_t*)(lineIn + i + data_size * (y + 1 + ymin)));

      auto source = _mm_unpacklo_epi8(source1, source2);
      auto pix = _mm_unpacklo_epi8(source, zero);
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    for (; y < ymax; y++) {
      auto mmk = _mm_set1_epi32(k[y]);
      auto pix = mm_cvtepu8_epi32(lineIn + i + data_size * (y + ymin));
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }
    sss = _mm_srai_epi32(sss, coefs_precision);
    sss = _mm_packs_epi32(sss, zero);
    sss = _mm_packus_epi16(sss, zero);

    if (num_channels == 3) {
      sss = _mm_shuffle_epi8(sss, mask_ch);
    }

    auto o = _mm_cvtsi128_si32(sss);

    // Here we write 4 bytes to the output even if num_channels < 4, e.g o = {r,g,b,0} for num_channels=3
    // It is OK to write 4th byte (e.g. 0) as on the next step we will overwrite it with new data.
    // We also wont go out of bounds of lineOut memory allocation
    *(uint32_t *)(lineOut + i) = o;
  }

  for (; i < data_size; i += data_stride) {
    auto sss = initial;
    int64_t y = 0;
    for (; y < ymax - 1; y += 2) {
      // Load two coefficients at once
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[y]);

      // Load 2 lines
      auto source1 = _mm_cvtsi32_si128(*(int32_t*)(lineIn + i + data_size * (y + ymin)));
      auto source2 = _mm_cvtsi32_si128(*(int32_t*)(lineIn + i + data_size * (y + 1 + ymin)));

      auto source = _mm_unpacklo_epi8(source1, source2);
      auto pix = _mm_unpacklo_epi8(source, zero);
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    for (; y < ymax; y++) {
      auto mmk = _mm_set1_epi32(k[y]);

      const uint8_t * p = lineIn + i + data_size * (y + ymin);
      __m128i pix;
      // TODO: Update condition to apply on the last pixel only
      if (num_channels == 3) {
        unsigned char output[4];
        std::memcpy(output, p, 3);
        pix = mm_cvtepu8_epi32(output);
      } else {
        pix = mm_cvtepu8_epi32(p);
      }
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    sss = _mm_srai_epi32(sss, coefs_precision);
    sss = _mm_packs_epi32(sss, zero);
    sss = _mm_packus_epi16(sss, zero);
    if (num_channels == 3) {
      sss = _mm_shuffle_epi8(sss, mask_ch);
    }

    auto o = _mm_cvtsi128_si32(sss);
    if (num_channels == 3 && C10_UNLIKELY(i + 4 >= data_size)) {
      std::memcpy(lineOut + i, (unsigned char *) &o, num_channels);
    } else {
      *(uint32_t *)(lineOut + i) = o;
    }
  }

}

} // anonymous namespace
#endif // CPU_CAPABILITY_AVX2
