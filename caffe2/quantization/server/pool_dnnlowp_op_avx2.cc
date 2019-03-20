#include "caffe2/quantization/server/pool_dnnlowp_op_avx2.h"

#include <immintrin.h>
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
      constexpr int VLEN = 8;
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

} // namespace caffe2
