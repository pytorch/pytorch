#include "THCUNN.h"
#include "THCTensor.hpp"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "common.h"

template <typename Dtype, typename Acctype, bool COUNT_INCLUDE_PAD>
__global__ void AvePoolForward(const int64_t nthreads,
    const Dtype* const bottom_data, const int64_t num, const int64_t channels,
    const int64_t height, const int64_t width, const int64_t pooled_height,
    const int64_t pooled_width, const int64_t kernel_h, const int64_t kernel_w,
    const int64_t stride_h, const int64_t stride_w, const int64_t pad_h, const int64_t pad_w,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int64_t pw = index % pooled_width;
    const int64_t ph = (index / pooled_width) % pooled_height;
    const int64_t c = (index / pooled_width / pooled_height) % channels;
    const int64_t n = index / pooled_width / pooled_height / channels;
    int64_t hstart = ph * stride_h - pad_h;
    int64_t wstart = pw * stride_w - pad_w;
    int64_t hend = min(hstart + kernel_h, height + pad_h);
    int64_t wend = min(wstart + kernel_w, width + pad_w);
    const int64_t pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Acctype aveval = Acctype(0);
    const Dtype* const bottom_slice = bottom_data + (n * channels + c) * height * width;
    for (int64_t h = hstart; h < hend; ++h) {
      for (int64_t w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    if(COUNT_INCLUDE_PAD)
      top_data[index] = ScalarConvert<Acctype, Dtype>::to(aveval / pool_size);
    else
      top_data[index] = ScalarConvert<Acctype, Dtype>::to(aveval / ((hend - hstart) * (wend - wstart)));
  }
}

template <typename Dtype, typename Acctype, bool COUNT_INCLUDE_PAD>
__global__ void AvePoolBackward(const int64_t nthreads, const Dtype* const top_diff,
    const int64_t num, const int64_t channels, const int64_t height,
    const int64_t width, const int64_t pooled_height, const int64_t pooled_width,
    const int64_t kernel_h, const int64_t kernel_w, const int64_t stride_h,
    const int64_t stride_w, const int64_t pad_h, const int64_t pad_w,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int64_t w = index % width + pad_w;
    const int64_t h = (index / width) % height + pad_h;
    const int64_t c = (index / width / height) % channels;
    const int64_t n = index / width / height / channels;
    const int64_t phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int64_t phend = min(h / stride_h + 1, pooled_height);
    const int64_t pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int64_t pwend = min(w / stride_w + 1, pooled_width);
    Acctype gradient = Acctype(0);
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int64_t ph = phstart; ph < phend; ++ph) {
      for (int64_t pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int64_t hstart = ph * stride_h - pad_h;
        int64_t wstart = pw * stride_w - pad_w;
        int64_t hend = min(hstart + kernel_h, height + pad_h);
        int64_t wend = min(wstart + kernel_w, width + pad_w);
        int64_t pool_size = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, height);
        wend = min(wend, width);
        if(COUNT_INCLUDE_PAD)
          gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
        else
          gradient += top_diff_slice[ph * pooled_width + pw] / ((hend - hstart) * (wend - wstart));
      }
    }
    bottom_diff[index] = ScalarConvert<Acctype, Dtype>::to(gradient);
  }
}

#include "generic/SpatialAveragePooling.cu"
#include "THCGenerateFloatTypes.h"
