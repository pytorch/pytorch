#include "./fused_nbit_rowwise_conversion.h"

#include <immintrin.h>
#include <algorithm>
#include <cfloat> // for FLT_MAX
#include <cmath>

#include "./cvtsh_ss_bugfix.h"

namespace caffe2 {

constexpr int VLEN = 8;

void FloatToFused8BitRowwiseQuantized__avx2_fma(
    const float* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output) {
  constexpr float kEpsilon = 1e-8f;

  __m256i permute_mask1_v =
      _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);
  __m256i shuffle_mask_v = _mm256_set_epi8(
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0x0c,
      0x08,
      0x04,
      0x00,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0x0c,
      0x08,
      0x04,
      0x00);
  __m256i permute_mask2_v =
      _mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);

  int output_columns = input_columns + 2 * sizeof(float);
  for (std::size_t row = 0; row < input_rows; ++row) {
    const float* input_row = input + row * input_columns;
    std::uint8_t* output_row = output + row * output_columns;
    float* output_row_scale_bias =
        reinterpret_cast<float*>(output_row + input_columns);

    float minimum_element = FLT_MAX;
    float maximum_element = -FLT_MAX;
    __m256 min_v = _mm256_set1_ps(minimum_element);
    __m256 max_v = _mm256_set1_ps(maximum_element);
    std::size_t col;
    for (col = 0; col < input_columns / VLEN * VLEN; col += VLEN) {
      __m256 in_v = _mm256_loadu_ps(input_row + col);
      min_v = _mm256_min_ps(min_v, in_v);
      max_v = _mm256_max_ps(max_v, in_v);
    }
    alignas(64) float min_buf[VLEN], max_buf[VLEN];
    _mm256_store_ps(min_buf, min_v);
    _mm256_store_ps(max_buf, max_v);
    for (int i = 0; i < VLEN; ++i) {
      minimum_element = std::min(minimum_element, min_buf[i]);
      maximum_element = std::max(maximum_element, max_buf[i]);
    }
    for (; col < input_columns; ++col) {
      minimum_element = std::min(minimum_element, input_row[col]);
      maximum_element = std::max(maximum_element, input_row[col]);
    }

    float range = maximum_element - minimum_element;

    output_row_scale_bias[0] = range / 255.0f;
    output_row_scale_bias[1] = minimum_element;
    const auto inverse_scale = 255.0f / (range + kEpsilon);
    min_v = _mm256_set1_ps(minimum_element);
    __m256 inverse_scale_v = _mm256_set1_ps(inverse_scale);

    for (col = 0; col < input_columns / (4 * VLEN) * (4 * VLEN);
         col += 4 * VLEN) {
      __m256i x_rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
          _mm256_sub_ps(_mm256_loadu_ps(input_row + col), min_v),
          inverse_scale_v));
      __m256i y_rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
          _mm256_sub_ps(_mm256_loadu_ps(input_row + col + VLEN), min_v),
          inverse_scale_v));
      __m256i z_rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
          _mm256_sub_ps(_mm256_loadu_ps(input_row + col + 2 * VLEN), min_v),
          inverse_scale_v));
      __m256i w_rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
          _mm256_sub_ps(_mm256_loadu_ps(input_row + col + 3 * VLEN), min_v),
          inverse_scale_v));

      // An instruction sequence to save 32 32-bit integers as 8-bit integers
      __m256i xy_packed_v = _mm256_packs_epi32(x_rounded_v, y_rounded_v);
      __m256i zw_packed_v = _mm256_packs_epi32(z_rounded_v, w_rounded_v);
      __m256i xyzw_packed_v = _mm256_packus_epi16(xy_packed_v, zw_packed_v);
      xyzw_packed_v =
          _mm256_permutevar8x32_epi32(xyzw_packed_v, permute_mask1_v);
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(output_row + col), xyzw_packed_v);
    }
    for (; col < input_columns / VLEN * VLEN; col += VLEN) {
      __m256i rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
          _mm256_sub_ps(_mm256_loadu_ps(input_row + col), min_v),
          inverse_scale_v));

      // An instruction sequence to save 8 32-bit integers as 8-bit integers
      rounded_v = _mm256_shuffle_epi8(rounded_v, shuffle_mask_v);
      rounded_v = _mm256_permutevar8x32_epi32(rounded_v, permute_mask2_v);
      _mm_storel_epi64(
          reinterpret_cast<__m128i*>(output_row + col),
          _mm256_castsi256_si128(rounded_v));
    }
    for (; col < input_columns; ++col) {
      output_row[col] =
          std::lrintf((input_row[col] - minimum_element) * inverse_scale);
    }
  }
}

void Fused8BitRowwiseQuantizedToFloat__avx2_fma(
    const std::uint8_t* input,
    int input_rows,
    int input_columns,
    float* output) {
  int output_columns = input_columns - 2 * sizeof(float);

  for (std::size_t row = 0; row < input_rows; ++row) {
    const std::uint8_t* input_row = input + row * input_columns;
    const float* input_row_scale_bias =
        reinterpret_cast<const float*>(input_row + output_columns);
    float* output_row = output + row * output_columns;

    __m256 scale_v = _mm256_set1_ps(input_row_scale_bias[0]);
    __m256 bias_v = _mm256_set1_ps(input_row_scale_bias[1]);

    std::size_t col;
    for (col = 0; col < output_columns / VLEN * VLEN; col += VLEN) {
      __m256 in_v = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
          _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input_row + col))));
      _mm256_storeu_ps(
          output_row + col,
          _mm256_add_ps(_mm256_mul_ps(in_v, scale_v), bias_v));
    }

    for (; col < output_columns; ++col) {
      output_row[col] =
          input_row[col] * input_row_scale_bias[0] + input_row_scale_bias[1];
    }
  }
}

namespace {

template <int BIT_RATE>
void FloatToFusedNBitRowwiseQuantizedSBHalf_(
    const float* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output) {
  __m256i permute_mask1_v =
      _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);

  int NUM_ELEM_PER_BYTE = 8 / BIT_RATE;
  int output_columns =
      (input_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE +
      2 * sizeof(std::uint16_t);
  for (std::size_t row = 0; row < input_rows; ++row) {
    const float* input_row = input + row * input_columns;
    std::uint8_t* output_row = output + row * output_columns;
    std::uint16_t* output_row_scale_bias = reinterpret_cast<std::uint16_t*>(
        output_row +
        (input_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE);

    float minimum_element = FLT_MAX;
    float maximum_element = -FLT_MAX;
    __m256 min_v = _mm256_set1_ps(minimum_element);
    __m256 max_v = _mm256_set1_ps(maximum_element);
    std::size_t col;
    for (col = 0; col < input_columns / VLEN * VLEN; col += VLEN) {
      __m256 in_v = _mm256_loadu_ps(input_row + col);
      min_v = _mm256_min_ps(min_v, in_v);
      max_v = _mm256_max_ps(max_v, in_v);
    }
    alignas(64) float min_buf[VLEN], max_buf[VLEN];
    _mm256_store_ps(min_buf, min_v);
    _mm256_store_ps(max_buf, max_v);
    for (int i = 0; i < VLEN; ++i) {
      minimum_element = std::min(minimum_element, min_buf[i]);
      maximum_element = std::max(maximum_element, max_buf[i]);
    }
    for (; col < input_columns; ++col) {
      minimum_element = std::min(minimum_element, input_row[col]);
      maximum_element = std::max(maximum_element, input_row[col]);
    }

    output_row_scale_bias[1] = _cvtss_sh(
        minimum_element, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    minimum_element = _cvtsh_ss(output_row_scale_bias[1]);
    const float range = maximum_element - minimum_element;

    float scale = range == 0 ? 1.0f : range / ((1 << BIT_RATE) - 1);
    std::uint16_t scale_fp16 =
        _cvtss_sh(scale, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    scale = _cvtsh_ss(scale_fp16);
    if (scale == 0) {
      // Corner case handling when maximum_element == minimum_element
      // Any scale would work because maximum_element - minimum_element will be
      // 0 for all X
      scale = 1.0f;
    }
    float inverse_scale = 1.0f / scale;
    if (std::isinf(inverse_scale)) {
      scale = 1.0f;
      inverse_scale = 1.0f;
    }

    output_row_scale_bias[0] =
        _cvtss_sh(scale, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    __m256 inverse_scale_v = _mm256_set1_ps(inverse_scale);
    min_v = _mm256_set1_ps(minimum_element);

    col = 0;

    if (BIT_RATE == 2 || BIT_RATE == 4) {
      for (; col + 4 * VLEN <= input_columns; col += 4 * VLEN) {
        __m256i x_rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
            _mm256_sub_ps(_mm256_loadu_ps(input_row + col), min_v),
            inverse_scale_v));
        __m256i y_rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
            _mm256_sub_ps(_mm256_loadu_ps(input_row + col + VLEN), min_v),
            inverse_scale_v));
        __m256i z_rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
            _mm256_sub_ps(_mm256_loadu_ps(input_row + col + 2 * VLEN), min_v),
            inverse_scale_v));
        __m256i w_rounded_v = _mm256_cvtps_epi32(_mm256_mul_ps(
            _mm256_sub_ps(_mm256_loadu_ps(input_row + col + 3 * VLEN), min_v),
            inverse_scale_v));

        // An instruction sequence to save 32 32-bit integers as 8-bit integers
        __m256i xy_packed_v = _mm256_packs_epi32(x_rounded_v, y_rounded_v);
        __m256i zw_packed_v = _mm256_packs_epi32(z_rounded_v, w_rounded_v);
        __m256i xyzw_packed_v = _mm256_packus_epi16(xy_packed_v, zw_packed_v);
        xyzw_packed_v =
            _mm256_permutevar8x32_epi32(xyzw_packed_v, permute_mask1_v);

        // saturate to BIT_RATE
        xyzw_packed_v = _mm256_min_epu8(
            xyzw_packed_v,
            _mm256_set1_epi8(static_cast<char>((1 << BIT_RATE) - 1)));

        if (BIT_RATE == 4) {
          // pack into lower 8-bit of each 16-bit
          xyzw_packed_v = _mm256_and_si256(
              _mm256_or_si256(
                  xyzw_packed_v, _mm256_srli_epi16(xyzw_packed_v, 4)),
              _mm256_set1_epi16(0x00ff));
        } else {
          // pack into lower 8-bit of each 32-bit
          xyzw_packed_v = _mm256_and_si256(
              _mm256_or_si256(
                  _mm256_or_si256(
                      xyzw_packed_v, _mm256_srli_epi32(xyzw_packed_v, 6)),
                  _mm256_or_si256(
                      _mm256_srli_epi32(xyzw_packed_v, 8 + 4),
                      _mm256_srli_epi32(xyzw_packed_v, 2 * 8 + 2))),
              _mm256_set1_epi32(0x00ff));
        }

        __m128i out_v;
        if (BIT_RATE == 4) {
          // avx2 doesn't have _mm256_cvtepi16_epi8
          out_v = _mm_packus_epi16(
              _mm256_castsi256_si128(xyzw_packed_v),
              _mm256_extractf128_si256(xyzw_packed_v, 1));
          _mm_storeu_si128(
              reinterpret_cast<__m128i*>(output_row + col / NUM_ELEM_PER_BYTE),
              out_v);
        } else {
          // avx2 doesn't have _mm256_cvtepi32_epi8
          out_v = _mm_packus_epi32(
              _mm256_castsi256_si128(xyzw_packed_v),
              _mm256_extractf128_si256(xyzw_packed_v, 1));
          out_v = _mm_packus_epi16(out_v, out_v);
          _mm_storel_epi64(
              reinterpret_cast<__m128i*>(output_row + col / NUM_ELEM_PER_BYTE),
              out_v);
        }
      }
    }

    for (; col < input_columns; ++col) {
      float X = input_row[col];
      std::uint8_t quantized = std::max(
          0,
          std::min<int>(
              std::lrintf((X - minimum_element) * inverse_scale),
              (1 << BIT_RATE) - 1));
      if (col % NUM_ELEM_PER_BYTE == 0) {
        output_row[col / NUM_ELEM_PER_BYTE] = quantized;
      } else {
        output_row[col / NUM_ELEM_PER_BYTE] |=
            (quantized << ((col % NUM_ELEM_PER_BYTE) * BIT_RATE));
      }
    }
  }
}

template <int BIT_RATE>
void FusedNBitRowwiseQuantizedSBHalfToFloat_(
    const std::uint8_t* input,
    int input_rows,
    int input_columns,
    float* output) {
  constexpr int NUM_ELEM_PER_BYTE = 8 / BIT_RATE;
  int output_columns =
      (input_columns - 2 * sizeof(std::uint16_t)) * NUM_ELEM_PER_BYTE;

  // mask can be accessed by avx2_ps_or_epi32_combined_mask[(8 - remainder) % 8]
  static const int avx2_ps_or_epi32_combined_mask[16] = {
      -1,
      -1,
      -1,
      -1,
      -1,
      -1,
      -1,
      -1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
  };

  // Compute a remainder for vector load
  // Since every row is followed by 2 fp16 (scale and bias), luckily
  // we don't need mask at bit-rate granularity but just at 32-bit
  // granularity.
  constexpr int NUM_ELEM_PER_32BIT = 32 / BIT_RATE;
  // multiply by 4 because we're handling 4 vlen per iteration
  constexpr int NUM_OF_32BIT_PER_VLOAD = VLEN * 4 / NUM_ELEM_PER_32BIT;
  int remainder_32bit_granularity = (output_columns + NUM_ELEM_PER_32BIT - 1) /
      NUM_ELEM_PER_32BIT % NUM_OF_32BIT_PER_VLOAD;
  __m128i vmask_load = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(
      avx2_ps_or_epi32_combined_mask + NUM_OF_32BIT_PER_VLOAD +
      (NUM_OF_32BIT_PER_VLOAD - remainder_32bit_granularity) %
          NUM_OF_32BIT_PER_VLOAD));
  int remainder = output_columns % (4 * VLEN);
  __m256i vmask_store0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
      avx2_ps_or_epi32_combined_mask +
      (VLEN - std::min(output_columns % (4 * VLEN), VLEN) % (VLEN + 1))));
  __m256i vmask_store1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
      avx2_ps_or_epi32_combined_mask +
      (VLEN -
       std::max(0, std::min(output_columns % (4 * VLEN) - VLEN, VLEN)) %
           (VLEN + 1))));
  __m256i vmask_store2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
      avx2_ps_or_epi32_combined_mask +
      (VLEN -
       std::max(0, std::min(output_columns % (4 * VLEN) - 2 * VLEN, VLEN)) %
           (VLEN + 1))));
  __m256i vmask_store3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
      avx2_ps_or_epi32_combined_mask +
      (VLEN -
       std::max(0, std::min(output_columns % (4 * VLEN) - 3 * VLEN, VLEN)) %
           (VLEN + 1))));

  for (std::size_t row = 0; row < input_rows; ++row) {
    const std::uint8_t* input_row = input + row * input_columns;
    const std::uint16_t* input_row_scale_bias =
        reinterpret_cast<const std::uint16_t*>(
            input_row +
            (output_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE);
    float scale = _cvtsh_ss(input_row_scale_bias[0]);
    float bias = _cvtsh_ss(input_row_scale_bias[1]);
    float* output_row = output + row * output_columns;

    std::size_t col = 0;
    if (BIT_RATE == 4 || BIT_RATE == 2) {
      __m256 vscale = _mm256_set1_ps(scale);
      __m256 vbias = _mm256_set1_ps(bias);
      for (; col + 4 * VLEN <= output_columns; col += 4 * VLEN) {
        __m256i vinq;
        // unpack to 8-bit integers
        if (BIT_RATE == 4) {
          vinq = _mm256_cvtepu8_epi16(
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                  input_row + col / NUM_ELEM_PER_BYTE)));
          vinq = _mm256_and_si256(
              _mm256_or_si256(vinq, _mm256_slli_epi32(vinq, 4)),
              _mm256_set1_epi16(0x0f0f));
        } else {
          vinq = _mm256_cvtepu8_epi32(
              _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
                  input_row + col / NUM_ELEM_PER_BYTE)));
          vinq = _mm256_and_si256(
              _mm256_or_si256(
                  _mm256_or_si256(
                      _mm256_slli_epi32(vinq, 2 * 8 + 2),
                      _mm256_slli_epi32(vinq, 8 + 4)),
                  _mm256_or_si256(_mm256_slli_epi32(vinq, 6), vinq)),
              _mm256_set1_epi32(0x03030303));
        }
        __m256 vinq0 = _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm256_castsi256_si128(vinq)));
        __m256 vinq1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_set1_epi64x(_mm256_extract_epi64(vinq, 1))));
        __m256 vinq2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_set1_epi64x(_mm256_extract_epi64(vinq, 2))));
        __m256 vinq3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_set1_epi64x(_mm256_extract_epi64(vinq, 3))));
        vinq0 = _mm256_fmadd_ps(vscale, vinq0, vbias);
        vinq1 = _mm256_fmadd_ps(vscale, vinq1, vbias);
        vinq2 = _mm256_fmadd_ps(vscale, vinq2, vbias);
        vinq3 = _mm256_fmadd_ps(vscale, vinq3, vbias);
        _mm256_storeu_ps(output_row + col, vinq0);
        _mm256_storeu_ps(output_row + col + VLEN, vinq1);
        _mm256_storeu_ps(output_row + col + 2 * VLEN, vinq2);
        _mm256_storeu_ps(output_row + col + 3 * VLEN, vinq3);
      }

      if (remainder) {
        __m256i vinq;
        if (BIT_RATE == 4) {
          vinq = _mm256_cvtepu8_epi16(_mm_maskload_epi32(
              reinterpret_cast<const int*>(input_row + col / NUM_ELEM_PER_BYTE),
              vmask_load));
          vinq = _mm256_and_si256(
              _mm256_or_si256(vinq, _mm256_slli_epi32(vinq, 4)),
              _mm256_set1_epi16(0x0f0f));
        } else {
          vinq = _mm256_cvtepu8_epi32(_mm_maskload_epi32(
              reinterpret_cast<const int*>(input_row + col / NUM_ELEM_PER_BYTE),
              vmask_load));
          vinq = _mm256_and_si256(
              _mm256_or_si256(
                  _mm256_or_si256(
                      _mm256_slli_epi32(vinq, 2 * 8 + 2),
                      _mm256_slli_epi32(vinq, 8 + 4)),
                  _mm256_or_si256(_mm256_slli_epi32(vinq, 6), vinq)),
              _mm256_set1_epi32(0x03030303));
        }

        __m256 vinq0 = _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm256_castsi256_si128(vinq)));
        __m256 vinq1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_set1_epi64x(_mm256_extract_epi64(vinq, 1))));
        __m256 vinq2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_set1_epi64x(_mm256_extract_epi64(vinq, 2))));
        __m256 vinq3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_set1_epi64x(_mm256_extract_epi64(vinq, 3))));

        vinq0 = _mm256_fmadd_ps(vscale, vinq0, vbias);
        vinq1 = _mm256_fmadd_ps(vscale, vinq1, vbias);
        vinq2 = _mm256_fmadd_ps(vscale, vinq2, vbias);
        vinq3 = _mm256_fmadd_ps(vscale, vinq3, vbias);

        _mm256_maskstore_ps(output_row + col, vmask_store0, vinq0);
        _mm256_maskstore_ps(output_row + col + VLEN, vmask_store1, vinq1);
        _mm256_maskstore_ps(output_row + col + 2 * VLEN, vmask_store2, vinq2);
        _mm256_maskstore_ps(output_row + col + 3 * VLEN, vmask_store3, vinq3);
      }
    } else {
      for (; col < output_columns; ++col) {
        std::uint8_t quantized = input_row[col / NUM_ELEM_PER_BYTE];
        quantized >>= (col % NUM_ELEM_PER_BYTE) * BIT_RATE;
        quantized &= (1 << BIT_RATE) - 1;
        output_row[col] = scale * quantized + bias;
      }
    }
  }
}
} // namespace

void FloatToFusedNBitRowwiseQuantizedSBHalf__avx2_fma(
    int bit_rate,
    const float* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output) {
  if (bit_rate == 2) {
    FloatToFusedNBitRowwiseQuantizedSBHalf_<2>(
        input, input_rows, input_columns, output);
  } else if (bit_rate == 4) {
    FloatToFusedNBitRowwiseQuantizedSBHalf_<4>(
        input, input_rows, input_columns, output);
  } else if (bit_rate == 8) {
    FloatToFusedNBitRowwiseQuantizedSBHalf_<8>(
        input, input_rows, input_columns, output);
  }
}

void FusedNBitRowwiseQuantizedSBHalfToFloat__avx2_fma(
    int bit_rate,
    const std::uint8_t* input,
    int input_rows,
    int input_columns,
    float* output) {
  if (bit_rate == 2) {
    FusedNBitRowwiseQuantizedSBHalfToFloat_<2>(
        input, input_rows, input_columns, output);
  } else if (bit_rate == 4) {
    FusedNBitRowwiseQuantizedSBHalfToFloat_<4>(
        input, input_rows, input_columns, output);
  } else {
    FusedNBitRowwiseQuantizedSBHalfToFloat_<8>(
        input, input_rows, input_columns, output);
  }
}

} // namespace caffe2
