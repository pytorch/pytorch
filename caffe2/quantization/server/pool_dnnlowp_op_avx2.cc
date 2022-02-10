#include "caffe2/quantization/server/pool_dnnlowp_op_avx2.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>

namespace caffe2 {

using namespace std;

void max_pool_avx2(
    const uint8_t* Xdata,
    int n,
    int height,
    int width,
    int channels,
    int pooled_height,
    int pooled_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_t,
    int pad_l,
    uint8_t* Ydata) {
  const uint8_t* Xdata_temp = Xdata + n * height * width * channels;
  uint8_t* Ydata_temp = Ydata + n * pooled_height * pooled_width * channels;
  for (int ph = 0; ph < pooled_height; ++ph) {
    int hstart = ph * stride_h - pad_t;
    int hend = hstart + kernel_h < height ? hstart + kernel_h : height;
    hstart = hstart > 0 ? hstart : 0;
    for (int pw = 0; pw < pooled_width; ++pw) {
      int wstart = pw * stride_w - pad_l;
      int wend = wstart + kernel_w < width ? wstart + kernel_w : width;
      wstart = wstart > 0 ? wstart : 0;

      uint8_t* Yh = Ydata_temp + (ph * pooled_width + pw) * channels;
      constexpr int VLEN = 32;
      // vectorized loop
      for (int c = 0; c < channels / VLEN * VLEN; c += VLEN) {
        __m256i Y_v = _mm256_setzero_si256();
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            const int input_idx = (h * width + w) * channels + c;
            Y_v = _mm256_max_epu8(
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(Xdata_temp + input_idx)),
                Y_v);
          }
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(Yh + c), Y_v);
      }

      // remainder
      for (int c = channels / VLEN * VLEN; c < channels; ++c) {
        Yh[c] = 0;
      }
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          for (int c = channels / VLEN * VLEN; c < channels; ++c) {
            const int input_idx = (h * width + w) * channels + c;
            Yh[c] =
                Xdata_temp[input_idx] > Yh[c] ? Xdata_temp[input_idx] : Yh[c];
          }
        }
      }
    } // pw loop
  } // ph loop
}

void average_pool_avx2(
    const uint8_t* Xdata,
    int n,
    int height,
    int width,
    int channels,
    int pooled_height,
    int pooled_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_t,
    int pad_l,
    uint8_t* Ydata,
    float in_scale,
    float out_scale,
    int32_t in_zero_point,
    int32_t out_zero_point,
    int32_t minimum,
    int32_t maximum) {
  const uint8_t* Xdata_temp = Xdata + n * height * width * channels;
  uint8_t* Ydata_temp = Ydata + n * pooled_height * pooled_width * channels;

  const __m256i shuffle_mask_v = _mm256_set_epi8(
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      0x0c,
      0x08,
      0x04,
      0x00,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      0x0c,
      0x08,
      0x04,
      0x00);
  const __m256i permute_mask_v =
      _mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);

  const __m256i min_v = _mm256_set1_epi32(minimum);
  const __m256i max_v = _mm256_set1_epi32(maximum);
  __m256 out_zero_point_v = _mm256_set1_ps(out_zero_point);

  for (int ph = 0; ph < pooled_height; ++ph) {
    int hstart = ph * stride_h - pad_t;
    int hend = hstart + kernel_h < height ? hstart + kernel_h : height;
    hstart = hstart > 0 ? hstart : 0;
    for (int pw = 0; pw < pooled_width; ++pw) {
      int wstart = pw * stride_w - pad_l;
      int wend = wstart + kernel_w < width ? wstart + kernel_w : width;
      wstart = wstart > 0 ? wstart : 0;

      int size = (hend - hstart) * (wend - wstart);
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      float multiplier = in_scale / out_scale / size;
      __m256 multiplier_v = _mm256_set1_ps(multiplier);

      uint8_t* Yh = Ydata_temp + (ph * pooled_width + pw) * channels;
      constexpr int VLEN = 8;
      int32_t Yh0 = -in_zero_point * size;

      // vectorized loop
      for (int c = 0; c < channels / VLEN * VLEN; c += VLEN) {
        __m256i Yh0_v = _mm256_set1_epi32(Yh0);

        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            const int input_idx = (h * width + w) * channels + c;
            const __m256i temp_v = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(Xdata_temp + input_idx)));
            Yh0_v = _mm256_add_epi32(Yh0_v, temp_v);
          }
        }

        __m256 Yh0_fp = _mm256_cvtepi32_ps(Yh0_v);
        __m256 Y_float_v =
            _mm256_fmadd_ps(Yh0_fp, multiplier_v, out_zero_point_v);
        __m256i Y_rounded_v = _mm256_cvtps_epi32(Y_float_v);
        __m256i Y_clipped_v =
            _mm256_max_epi32(min_v, _mm256_min_epi32(max_v, Y_rounded_v));

        Y_clipped_v = _mm256_shuffle_epi8(Y_clipped_v, shuffle_mask_v);
        Y_clipped_v = _mm256_permutevar8x32_epi32(Y_clipped_v, permute_mask_v);
        *reinterpret_cast<int64_t*>(Yh + c) =
            _mm256_extract_epi64(Y_clipped_v, 0);
      }

      // remainder
      for (int c = channels / VLEN * VLEN; c < channels; ++c) {
        Yh[c] = 0;
      }

      for (int c = channels / VLEN * VLEN; c < channels; ++c) {
        const int pool_idx = (ph * pooled_width + pw) * channels + c;
        int32_t Yh_t = -in_zero_point * size;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            const int input_idx = (h * width + w) * channels + c;
            Yh_t += Xdata_temp[input_idx];
          }
        }

        Ydata_temp[pool_idx] = std::min<int32_t>(
            std::max<int32_t>(
                nearbyint(Yh_t * multiplier + out_zero_point), minimum),
            maximum);
      }
    } // pw loop
  } // ph loop
}

void average_pool_3d_avx2(
    const uint8_t* Xdata,
    int n,
    int height,
    int width,
    int depth,
    int channels,
    int pooled_height,
    int pooled_width,
    int pooled_depth,
    int kernel_h,
    int kernel_w,
    int kernel_d,
    int stride_h,
    int stride_w,
    int stride_d,
    int pad_t,
    int pad_l,
    int pad_d,
    uint8_t* Ydata,
    float in_scale,
    float out_scale,
    int32_t in_zero_point,
    int32_t out_zero_point,
    int32_t minimum,
    int32_t maximum) {
  const uint8_t* Xdata_temp = Xdata + n * height * width * depth * channels;
  uint8_t* Ydata_temp =
      Ydata + n * pooled_height * pooled_width * pooled_depth * channels;

  const __m256i shuffle_mask_v = _mm256_set_epi8(
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      0x0c,
      0x08,
      0x04,
      0x00,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      0xff,
      0x0c,
      0x08,
      0x04,
      0x00);
  const __m256i permute_mask_v =
      _mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);

  const __m256i min_v = _mm256_set1_epi32(minimum);
  const __m256i max_v = _mm256_set1_epi32(maximum);
  __m256 out_zero_point_v = _mm256_set1_ps(out_zero_point);

  for (int ph = 0; ph < pooled_height; ++ph) {
    int hstart = ph * stride_h - pad_t;
    int hend = hstart + kernel_h < height ? hstart + kernel_h : height;
    hstart = hstart > 0 ? hstart : 0;
    for (int pw = 0; pw < pooled_width; ++pw) {
      int wstart = pw * stride_w - pad_l;
      int wend = wstart + kernel_w < width ? wstart + kernel_w : width;
      wstart = wstart > 0 ? wstart : 0;
      for (int pd = 0; pd < pooled_depth; ++pd) {
        int dstart = pd * stride_d - pad_d;
        int dend = dstart + kernel_d < depth ? dstart + kernel_d : depth;
        dstart = max(dstart, 0);

        int size = (hend - hstart) * (wend - wstart) * (dend - dstart);
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
        float multiplier = in_scale / out_scale / size;
        __m256 multiplier_v = _mm256_set1_ps(multiplier);

        uint8_t* Yh = Ydata_temp +
            ((ph * pooled_width + pw) * pooled_depth + pd) * channels;
        constexpr int VLEN = 8;
        int32_t Yh0 = -in_zero_point * size;

        // vectorized loop
        for (int c = 0; c < channels / VLEN * VLEN; c += VLEN) {
          __m256i Yh0_v = _mm256_set1_epi32(Yh0);

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              for (int d = dstart; d < dend; ++d) {
                const int input_idx =
                    ((h * width + w) * depth + d) * channels + c;
                const __m256i temp_v = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(Xdata_temp + input_idx)));
                Yh0_v = _mm256_add_epi32(Yh0_v, temp_v);
              }
            }
          }

          __m256 Yh0_fp = _mm256_cvtepi32_ps(Yh0_v);
          __m256 Y_float_v =
              _mm256_fmadd_ps(Yh0_fp, multiplier_v, out_zero_point_v);
          __m256i Y_rounded_v = _mm256_cvtps_epi32(Y_float_v);
          __m256i Y_clipped_v =
              _mm256_max_epi32(min_v, _mm256_min_epi32(max_v, Y_rounded_v));

          Y_clipped_v = _mm256_shuffle_epi8(Y_clipped_v, shuffle_mask_v);
          Y_clipped_v =
              _mm256_permutevar8x32_epi32(Y_clipped_v, permute_mask_v);
          *reinterpret_cast<int64_t*>(Yh + c) =
              _mm256_extract_epi64(Y_clipped_v, 0);
        }

        // remainder
        for (int c = channels / VLEN * VLEN; c < channels; ++c) {
          Yh[c] = 0;
        }

        for (int c = channels / VLEN * VLEN; c < channels; ++c) {
          const int pool_idx =
              ((ph * pooled_width + pw) * pooled_depth + pd) * channels + c;

          int32_t Yh_t = -in_zero_point * size;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              for (int d = dstart; d < dend; ++d) {
                const int input_idx =
                    ((h * width + w) * depth + d) * channels + c;

                Yh_t += Xdata_temp[input_idx];
              }
            }
          }

          Ydata_temp[pool_idx] = std::min<int32_t>(
              std::max<int32_t>(
                  nearbyint(Yh_t * multiplier + out_zero_point), minimum),
              maximum);
        }

      } // pd loop
    } // pw loop
  } // ph loop
}

} // namespace caffe2
