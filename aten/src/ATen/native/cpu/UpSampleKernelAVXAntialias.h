#pragma once
#ifdef CPU_CAPABILITY_AVX2

#include <ATen/core/Tensor.h>
#include <ATen/cpu/vec/intrinsics.h>
#include <c10/util/irange.h>

#include <stdio.h> // TODO: remove
#include <iostream> // TODO: remove

namespace {

static __m128i inline mm_cvtepu8_epi32(void* ptr) {
  return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)ptr));
}

typedef double (*aa_filter_fn_t)(
    double x); // TODO: use aa_filter_fn_t from UpSampleKernel.cpp file instead?

void unpack_rgb(
    uint8_t* unpacked, // OUT
    const at::Tensor& packed_tensor, // IN
    bool is_channels_last) {
  const uint8_t* packed = (const uint8_t*)packed_tensor.data_ptr<uint8_t>();
  auto num_pixels = packed_tensor.size(1) * packed_tensor.size(2);
  auto num_channels = packed_tensor.size(0);
  auto packed_stride = is_channels_last ? 1 : num_pixels;
  auto packed_increment = is_channels_last ? num_channels : 1;

  for (const auto i : c10::irange(num_pixels)) {
    for (const auto j : c10::irange(num_channels)) {
      unpacked[j] = packed[j * packed_stride];
    }
    for (const auto j : c10::irange(num_channels, 4)) {
      unpacked[j] = 255;
    }
    unpacked += 4;
    packed += packed_increment;
  }
}

void pack_rgb(
    const uint8_t* unpacked, // IN
    const at::Tensor& packed_tensor, // OUT
    bool is_channels_last) {
  uint8_t* packed = (uint8_t*)packed_tensor.data_ptr<uint8_t>();
  auto num_pixels = packed_tensor.size(1) * packed_tensor.size(2);
  auto num_channels = packed_tensor.size(0);
  auto packed_stride = is_channels_last ? 1 : num_pixels;
  auto packed_increment = is_channels_last ? num_channels : 1;
  for (const auto i : c10::irange(num_pixels)) {
    for (const auto j : c10::irange(num_channels)) {
      packed[j * packed_stride] = unpacked[j];
    }
    unpacked += 4;
    packed += packed_increment;
  }
}

int precompute_coeffs( // TODO: This is redundant with
                       // `compute_indices_weights_aa()`
    int inSize,
    int outSize,
    int** boundsp,
    double** kkp,
    aa_filter_fn_t filter,
    int interp_size) {
  double support, scale, filterscale;
  double center, ww, ss;
  int xx, x, ksize, xmin, xmax;
  int* bounds;
  double *kk, *k;

  /* prepare for horizontal stretch */
  filterscale = scale = (double)inSize / outSize;
  if (filterscale < 1.0) {
    filterscale = 1.0;
  }
  /* determine support size (length of resampling filter) */
  support = filterscale * interp_size / 2.;

  /* maximum number of coeffs */
  ksize = (int)ceil(support) * 2 + 1;

  // check for overflow
  if (outSize > INT_MAX / (ksize * (int)sizeof(double))) {
    // ImagingError_MemoryError();
    AT_ERROR("BLOP");
    return 0;
  }

  /* coefficient buffer */
  /* malloc check ok, overflow checked above */
  kk = (double*)malloc(outSize * ksize * sizeof(double));
  if (!kk) {
    // ImagingError_MemoryError();
    AT_ERROR("BLIP");
    return 0;
  }

  /* malloc check ok, ksize*sizeof(double) > 2*sizeof(int) */
  bounds = (int*)malloc(outSize * 2 * sizeof(int));
  if (!bounds) {
    free(kk);
    // ImagingError_MemoryError();
    AT_ERROR("BLOOP");
    return 0;
  }

  for (xx = 0; xx < outSize; xx++) {
    center = (xx + 0.5) * scale;
    ww = 0.0;
    ss = 1.0 / filterscale;
    // Round the value
    xmin = (int)(center - support + 0.5);
    if (xmin < 0) {
      xmin = 0;
    }
    // Round the value
    xmax = (int)(center + support + 0.5);
    if (xmax > inSize) {
      xmax = inSize;
    }
    xmax -= xmin;
    k = &kk[xx * ksize];
    for (x = 0; x < xmax; x++) {
      double w = filter((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (x = 0; x < xmax; x++) {
      if (ww != 0.0) {
        k[x] /= ww;
      }
    }
    // Remaining values should stay empty if they are used despite of xmax.
    for (; x < ksize; x++) {
      k[x] = 0;
    }
    bounds[xx * 2 + 0] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  *boundsp = bounds;
  *kkp = kk;
  return ksize;
}

#define PRECISION_BITS (32 - 8 - 2)
#define MAX_COEFS_PRECISION (16 - 1)

int normalize_coeffs_8bpc(int outSize, int ksize, double* prekk) {
  int x;
  int16_t* kk;
  int coefs_precision;
  double maxkk;

  // use the same buffer for normalized coefficients
  kk = (int16_t*)prekk;

  maxkk = prekk[0];
  for (x = 0; x < outSize * ksize; x++) {
    if (maxkk < prekk[x]) {
      maxkk = prekk[x];
    }
  }

  for (coefs_precision = 0; coefs_precision < PRECISION_BITS;
       coefs_precision += 1) {
    int next_value = (int)(0.5 + maxkk * (1 << (coefs_precision + 1)));
    // The next value will be outside of the range, so just stop
    if (next_value >= (1 << MAX_COEFS_PRECISION))
      break;
  }

  for (x = 0; x < outSize * ksize; x++) {
    if (prekk[x] < 0) {
      kk[x] = (int)(-0.5 + prekk[x] * (1 << coefs_precision));
    } else {
      kk[x] = (int)(0.5 + prekk[x] * (1 << coefs_precision));
    }
  }
  return coefs_precision;
}

void ImagingResampleHorizontalConvolution8u4x(
    uint32_t* lineOut0,
    uint32_t* lineOut1,
    uint32_t* lineOut2,
    uint32_t* lineOut3,
    uint32_t* lineIn0,
    uint32_t* lineIn1,
    uint32_t* lineIn2,
    uint32_t* lineIn3,
    int xsize,
    int* xbounds,
    int16_t* kk,
    int kmax,
    int coefs_precision);
void ImagingResampleHorizontalConvolution8u(
    uint32_t* lineOut,
    uint32_t* lineIn,
    int xsize,
    int* xbounds,
    int16_t* kk,
    int kmax,
    int coefs_precision);
void ImagingResampleVerticalConvolution8u(
    uint32_t* lineOut,
    uint32_t* imIn,
    int xmin,
    int xmax,
    int16_t* k,
    int coefs_precision,
    int xin);

void ImagingResampleHorizontal_8bpc(
    uint32_t* unpacked_output_p,
    uint32_t* unpacked_input_p,
    int ksize,
    int* bounds,
    double* prekk,
    int xout,
    int yout,
    int xin) {
  int yy;
  int16_t* kk;
  int coefs_precision;

  // use the same buffer for normalized coefficients
  kk = (int16_t*)prekk;
  coefs_precision = normalize_coeffs_8bpc(xout, ksize, prekk);

  yy = 0;
  for (; yy < yout - 3; yy += 4) {
    ImagingResampleHorizontalConvolution8u4x(
        unpacked_output_p + yy * xout,
        unpacked_output_p + (yy + 1) * xout,
        unpacked_output_p + (yy + 2) * xout,
        unpacked_output_p + (yy + 3) * xout,
        unpacked_input_p + yy * xin,
        unpacked_input_p + (yy + 1) * xin,
        unpacked_input_p + (yy + 2) * xin,
        unpacked_input_p + (yy + 3) * xin,
        xout,
        bounds,
        kk,
        ksize,
        coefs_precision);
  }
  for (; yy < yout; yy++) {
    ImagingResampleHorizontalConvolution8u(
        unpacked_output_p + yy * xout,
        unpacked_input_p + yy * xin,
        xout,
        bounds,
        kk,
        ksize,
        coefs_precision);
  }
}

void ImagingResampleVertical_8bpc(
    uint32_t* unpacked_output_p,
    uint32_t* unpacked_input_p,
    int ksize,
    int* bounds,
    double* prekk,
    int xout,
    int yout) {
  int ymin, ymax;
  int16_t *k, *kk;
  int coefs_precision;

  // use the same buffer for normalized coefficients
  kk = (int16_t*)prekk;
  coefs_precision = normalize_coeffs_8bpc(yout, ksize, prekk);

  for (const auto yy : c10::irange(yout)) {
    k = &kk[yy * ksize];
    ymin = bounds[yy * 2 + 0];
    ymax = bounds[yy * 2 + 1];
    ImagingResampleVerticalConvolution8u(
        unpacked_output_p + yy * xout,
        unpacked_input_p,
        ymin,
        ymax,
        k,
        coefs_precision,
        xout);
  }
}

// TODO: Cleanup error checks (as in comments)
// TODO: probably use pytorch's allocator instead of malloc
uint32_t* ImagingResampleInner(
    uint32_t* unpacked_input_p,
    int xin,
    int yin,
    int xout,
    int yout,
    aa_filter_fn_t filter,
    int interp_size) {
  int need_horizontal, need_vertical;
  int ksize_horiz, ksize_vert;
  int *bounds_horiz, *bounds_vert;
  double *kk_horiz, *kk_vert;
  uint32_t* unpacked_output_p = NULL;
  uint32_t* unpacked_output_temp_p = NULL;

  need_horizontal = xout != xin;
  need_vertical = yout != yin;

  /* two-pass resize, horizontal pass */
  if (need_horizontal) {
    ksize_horiz = precompute_coeffs(
        xin, xout, &bounds_horiz, &kk_horiz, filter, interp_size);
    // if (!ksize_horiz) {
    //     return NULL;
    // }

    // imTemp = ImagingNewDirty(imIn->mode, xout, ybox_last - ybox_first);
    // if (imTemp) {
    //     ResampleHorizontal(
    //         imTemp, imIn, ybox_first, ksize_horiz, bounds_horiz, kk_horiz);
    // }
    unpacked_output_temp_p = (uint32_t*)malloc(sizeof(uint32_t) * xout * yin);
    ImagingResampleHorizontal_8bpc(
        unpacked_output_temp_p,
        unpacked_input_p,
        ksize_horiz,
        bounds_horiz,
        kk_horiz,
        xout,
        yin,
        xin);
    free(bounds_horiz);
    free(kk_horiz);
    // if (!imTemp) {
    //     free(bounds_vert);
    //     free(kk_vert);
    //     return NULL;
    // }
    // imOut = imIn = imTemp;
    unpacked_output_p = unpacked_input_p = unpacked_output_temp_p;
  }

  /* vertical pass */
  if (need_vertical) {
    ksize_vert = precompute_coeffs(
        yin, yout, &bounds_vert, &kk_vert, filter, interp_size);
    // if (!ksize_vert) {
    //     free(bounds_horiz);
    //     free(kk_horiz);
    //     return NULL;
    // }

    // imOut = ImagingNewDirty(imIn->mode, imIn->xsize, ysize);
    unpacked_output_p = (uint32_t*)malloc(sizeof(uint32_t) * xout * yout);
    // if (imOut) {
    ImagingResampleVertical_8bpc(
        unpacked_output_p,
        unpacked_input_p,
        ksize_vert,
        bounds_vert,
        kk_vert,
        xout,
        yout);
    // }
    // /* it's safe to call ImagingDelete with empty value
    //    if previous step was not performed. */
    // ImagingDelete(imTemp);
    free(unpacked_output_temp_p);
    free(bounds_vert);
    free(kk_vert);
    // if (!imOut) {
    //     return NULL;
    // }
  }

  // /* none of the previous steps are performed, copying */
  // if (!imOut) {
  //     imOut = ImagingCopy(imIn);
  // }

  return unpacked_output_p;
}

void upsample_avx_bilinear_or_bicubic(
    const at::Tensor& input,
    const at::Tensor& output,
    aa_filter_fn_t filter,
    int interp_size) {
  auto batch_size = input.size(0);
  auto xin = input.size(3);
  auto yin = input.size(2);
  auto xout = output.size(3);
  auto yout = output.size(2);
  auto num_pixels_input = xin * yin;

  uint32_t* unpacked_input_p =
      (uint32_t*)malloc(sizeof(uint32_t) * num_pixels_input);

  for (const auto i : c10::irange(batch_size)) {
    unpack_rgb(
        (uint8_t*)unpacked_input_p,
        input[i],
        input.is_contiguous(at::MemoryFormat::ChannelsLast));

    uint32_t* unpacked_output_p = ImagingResampleInner(
        unpacked_input_p, xin, yin, xout, yout, filter, interp_size);

    pack_rgb(
        (const uint8_t*)unpacked_output_p,
        output[i],
        output.is_contiguous(at::MemoryFormat::ChannelsLast));
  }
  free(unpacked_input_p);
}

void ImagingResampleHorizontalConvolution8u4x(
    uint32_t* lineOut0,
    uint32_t* lineOut1,
    uint32_t* lineOut2,
    uint32_t* lineOut3,
    uint32_t* lineIn0,
    uint32_t* lineIn1,
    uint32_t* lineIn2,
    uint32_t* lineIn3,
    int xsize,
    int* xbounds,
    int16_t* kk,
    int kmax,
    int coefs_precision) {
  int xmin, xmax, x;
  int16_t* k;

  for (const auto xx : c10::irange(xsize)) {
    xmin = xbounds[xx * 2 + 0];
    xmax = xbounds[xx * 2 + 1];
    k = &kk[xx * kmax];
    x = 0;

    __m256i sss0, sss1;
    __m256i zero = _mm256_setzero_si256();
    __m256i initial = _mm256_set1_epi32(1 << (coefs_precision - 1));
    sss0 = initial;
    sss1 = initial;

    for (; x < xmax - 3; x += 4) {
      __m256i pix, mmk0, mmk1, source;

      mmk0 = _mm256_set1_epi32(*(int32_t*)&k[x]);
      mmk1 = _mm256_set1_epi32(*(int32_t*)&k[x + 2]);

      source = _mm256_inserti128_si256(
          _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)&lineIn0[x + xmin])),
          _mm_loadu_si128((__m128i*)&lineIn1[x + xmin]),
          1);
      // clang-format off
      pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
        -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
        -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk0));
      pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
        -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8,
        -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8));
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk1));

      source = _mm256_inserti128_si256(
          _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)&lineIn2[x + xmin])),
          _mm_loadu_si128((__m128i*)&lineIn3[x + xmin]),
          1);
      pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
        -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
        -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk0));
      pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
        -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8,
        -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8));
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk1));
    }

    for (; x < xmax - 1; x += 2) {
      __m256i pix, mmk;

      mmk = _mm256_set1_epi32(*(int32_t*)&k[x]);

      pix = _mm256_inserti128_si256(
          _mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)&lineIn0[x + xmin])),
          _mm_loadl_epi64((__m128i*)&lineIn1[x + xmin]),
          1);
      pix = _mm256_shuffle_epi8(pix, _mm256_set_epi8(
        -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
        -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));

      pix = _mm256_inserti128_si256(
          _mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)&lineIn2[x + xmin])),
          _mm_loadl_epi64((__m128i*)&lineIn3[x + xmin]),
          1);
      pix = _mm256_shuffle_epi8(pix, _mm256_set_epi8(
        -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
        -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));
      // clang-format on
    }

    for (; x < xmax; x++) {
      __m256i pix, mmk;

      // [16] xx k0 xx k0 xx k0 xx k0 xx k0 xx k0 xx k0 xx k0
      mmk = _mm256_set1_epi32(k[x]);

      // [16] xx a0 xx b0 xx g0 xx r0 xx a0 xx b0 xx g0 xx r0
      pix = _mm256_inserti128_si256(
          _mm256_castsi128_si256(mm_cvtepu8_epi32(&lineIn0[x + xmin])),
          mm_cvtepu8_epi32(&lineIn1[x + xmin]),
          1);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));

      pix = _mm256_inserti128_si256(
          _mm256_castsi128_si256(mm_cvtepu8_epi32(&lineIn2[x + xmin])),
          mm_cvtepu8_epi32(&lineIn3[x + xmin]),
          1);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));
    }

    sss0 = _mm256_srai_epi32(sss0, coefs_precision);
    sss1 = _mm256_srai_epi32(sss1, coefs_precision);
    sss0 = _mm256_packs_epi32(sss0, zero);
    sss1 = _mm256_packs_epi32(sss1, zero);
    sss0 = _mm256_packus_epi16(sss0, zero);
    sss1 = _mm256_packus_epi16(sss1, zero);
    lineOut0[xx] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss0, 0));
    lineOut1[xx] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss0, 1));
    lineOut2[xx] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss1, 0));
    lineOut3[xx] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss1, 1));
  }
}

void ImagingResampleHorizontalConvolution8u(
    uint32_t* lineOut,
    uint32_t* lineIn,
    int xsize,
    int* xbounds,
    int16_t* kk,
    int kmax,
    int coefs_precision) {
  int xmin, xmax, x;
  int16_t* k;

  for (const auto xx : c10::irange(xsize)) {
    __m128i sss;
    xmin = xbounds[xx * 2 + 0];
    xmax = xbounds[xx * 2 + 1];
    k = &kk[xx * kmax];
    x = 0;

    if (xmax < 8) {
      sss = _mm_set1_epi32(1 << (coefs_precision - 1));
    } else {
      // Lower part will be added to higher, use only half of the error
      __m256i sss256 = _mm256_set1_epi32(1 << (coefs_precision - 2));

      for (; x < xmax - 7; x += 8) {
        __m256i pix, mmk, source;
        __m128i tmp = _mm_loadu_si128((__m128i*)&k[x]);
        __m256i ksource =
            _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        // clang-format off
        source = _mm256_loadu_si256((__m256i*)&lineIn[x + xmin]);
        pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
          -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
          -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
        mmk = _mm256_shuffle_epi8(ksource, _mm256_set_epi8(
          11,10, 9,8, 11,10, 9,8, 11,10, 9,8, 11,10, 9,8,
          3,2, 1,0, 3,2, 1,0, 3,2, 1,0, 3,2, 1,0));
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));

        pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
          -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8,
          -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8));
        mmk = _mm256_shuffle_epi8(ksource, _mm256_set_epi8(
          15,14, 13,12, 15,14, 13,12, 15,14, 13,12, 15,14, 13,12,
          7,6, 5,4, 7,6, 5,4, 7,6, 5,4, 7,6, 5,4));
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));
        // clang-format on
      }

      for (; x < xmax - 3; x += 4) {
        __m256i pix, mmk, source;
        __m128i tmp = _mm_loadl_epi64((__m128i*)&k[x]);
        __m256i ksource =
            _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        tmp = _mm_loadu_si128((__m128i*)&lineIn[x + xmin]);
        source = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        // clang-format off
        pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
          -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8,
          -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
        mmk = _mm256_shuffle_epi8(ksource, _mm256_set_epi8(
          7,6, 5,4, 7,6, 5,4, 7,6, 5,4, 7,6, 5,4, 
          3,2, 1,0, 3,2, 1,0, 3,2, 1,0, 3,2, 1,0));
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));
        // clang-format on
      }

      sss = _mm_add_epi32(
          _mm256_extracti128_si256(sss256, 0),
          _mm256_extracti128_si256(sss256, 1));
    }

    for (; x < xmax - 1; x += 2) {
      __m128i mmk = _mm_set1_epi32(*(int32_t*)&k[x]);
      __m128i source = _mm_loadl_epi64((__m128i*)&lineIn[x + xmin]);
      __m128i pix = _mm_shuffle_epi8(
          source,
          _mm_set_epi8(-1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0));
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    for (; x < xmax; x++) {
      __m128i pix = mm_cvtepu8_epi32(&lineIn[x + xmin]);
      __m128i mmk = _mm_set1_epi32(k[x]);
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }
    sss = _mm_srai_epi32(sss, coefs_precision);
    sss = _mm_packs_epi32(sss, sss);
    lineOut[xx] = _mm_cvtsi128_si32(_mm_packus_epi16(sss, sss));
  }
}

void ImagingResampleVerticalConvolution8u(
    uint32_t* lineOut,
    uint32_t* imIn,
    int xmin,
    int xmax,
    int16_t* k,
    int coefs_precision,
    int xin) {
  int x;
  int xx = 0;
  int xsize = xin;

  __m128i initial = _mm_set1_epi32(1 << (coefs_precision - 1));
  __m256i initial_256 = _mm256_set1_epi32(1 << (coefs_precision - 1));

  for (; xx < xsize - 7; xx += 8) {
    __m256i sss0 = initial_256;
    __m256i sss1 = initial_256;
    __m256i sss2 = initial_256;
    __m256i sss3 = initial_256;
    x = 0;
    for (; x < xmax - 1; x += 2) {
      __m256i source, source1, source2;
      __m256i pix, mmk;

      // Load two coefficients at once
      mmk = _mm256_set1_epi32(*(int32_t*)&k[x]);

      // Load 2 lines
      //                           (__m256i *) &imIn->image32[x + xmin][xx]
      source1 = _mm256_loadu_si256((__m256i*)(imIn + (x + xmin) * xin + xx));
      //                           (__m256i *) &imIn->image32[x + 1 + xmin][xx]
      source2 =
          _mm256_loadu_si256((__m256i*)(imIn + (x + 1 + xmin) * xin + xx));

      source = _mm256_unpacklo_epi8(source1, source2);
      pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));
      pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));

      source = _mm256_unpackhi_epi8(source1, source2);
      pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
      sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix, mmk));
      pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
      sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix, mmk));
    }
    for (; x < xmax; x += 1) {
      __m256i source, source1, pix, mmk;
      mmk = _mm256_set1_epi32(k[x]);

      //                           (__m256i *) &imIn->image32[x + xmin][xx])
      source1 = _mm256_loadu_si256((__m256i*)(imIn + (x + xmin) * xin + xx));

      source = _mm256_unpacklo_epi8(source1, _mm256_setzero_si256());
      pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));
      pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));

      source = _mm256_unpackhi_epi8(source1, _mm256_setzero_si256());
      pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
      sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix, mmk));
      pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
      sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix, mmk));
    }
    sss0 = _mm256_srai_epi32(sss0, coefs_precision);
    sss1 = _mm256_srai_epi32(sss1, coefs_precision);
    sss2 = _mm256_srai_epi32(sss2, coefs_precision);
    sss3 = _mm256_srai_epi32(sss3, coefs_precision);

    sss0 = _mm256_packs_epi32(sss0, sss1);
    sss2 = _mm256_packs_epi32(sss2, sss3);
    sss0 = _mm256_packus_epi16(sss0, sss2);
    _mm256_storeu_si256((__m256i*)&lineOut[xx], sss0);
  }

  for (; xx < xsize - 1; xx += 2) {
    __m128i sss0 = initial; // left row
    __m128i sss1 = initial; // right row
    x = 0;
    for (; x < xmax - 1; x += 2) {
      __m128i source, source1, source2;
      __m128i pix, mmk;

      // Load two coefficients at once
      mmk = _mm_set1_epi32(*(int32_t*)&k[x]);

      // Load 2 lines
      //                        (__m128i *) &imIn->image32[x + xmin][xx])
      source1 = _mm_loadl_epi64((__m128i*)(imIn + (x + xmin) * xin + xx));
      //                        (__m128i *) &imIn->image32[x + 1 + xmin][xx]
      source2 = _mm_loadl_epi64((__m128i*)(imIn + (x + 1 + xmin) * xin + xx));

      source = _mm_unpacklo_epi8(source1, source2);
      pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
      sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
      pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
      sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
    }
    for (; x < xmax; x += 1) {
      __m128i source, source1, pix, mmk;
      mmk = _mm_set1_epi32(k[x]);

      //                        (__m128i *) &imIn->image32[x + xmin][xx]);
      source1 = _mm_loadl_epi64((__m128i*)(imIn + (x + xmin) * xin + xx));

      source = _mm_unpacklo_epi8(source1, _mm_setzero_si128());
      pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
      sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
      pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
      sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
    }
    sss0 = _mm_srai_epi32(sss0, coefs_precision);
    sss1 = _mm_srai_epi32(sss1, coefs_precision);

    sss0 = _mm_packs_epi32(sss0, sss1);
    sss0 = _mm_packus_epi16(sss0, sss0);
    _mm_storel_epi64((__m128i*)&lineOut[xx], sss0);
  }

  for (; xx < xsize; xx++) {
    __m128i sss = initial;
    x = 0;
    for (; x < xmax - 1; x += 2) {
      __m128i source, source1, source2;
      __m128i pix, mmk;

      // Load two coefficients at once
      mmk = _mm_set1_epi32(*(int32_t*)&k[x]);

      // Load 2 lines
      //                           *(int *) &imIn->image32[x + xmin][xx]
      source1 = _mm_cvtsi32_si128(*(int*)(imIn + (x + xmin) * xin + xx));
      //                          *(int *) &imIn->image32[x + 1 + xmin][xx]
      source2 = _mm_cvtsi32_si128(*(int*)(imIn + (x + 1 + xmin) * xin + xx));

      source = _mm_unpacklo_epi8(source1, source2);
      pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    for (; x < xmax; x++) {
      //                             &imIn->image32[x + xmin][xx]
      __m128i pix = mm_cvtepu8_epi32(imIn + (x + xmin) * xin + xx);
      __m128i mmk = _mm_set1_epi32(k[x]);
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }
    sss = _mm_srai_epi32(sss, coefs_precision);
    sss = _mm_packs_epi32(sss, sss);
    lineOut[xx] = _mm_cvtsi128_si32(_mm_packus_epi16(sss, sss));
  }
}

} // anonymous namespace
#endif // CPU_CAPABILITY_AVX2