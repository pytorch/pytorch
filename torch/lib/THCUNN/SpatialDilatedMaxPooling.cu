#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "common.h"

// kernels borrowed from Caffe
template <typename Dtype, typename AccType>
__global__ void MaxPoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, Dtype* top_data,
    int64_t* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + (kernel_h - 1) * dilation_h + 1, height);
    int wend = min(wstart + (kernel_w - 1) * dilation_w + 1, width);
    while(hstart < 0)
      hstart += dilation_h;
    while(wstart < 0)
      wstart += dilation_w;
    AccType maxval = THCNumerics<AccType>::min();
    int maxidx = -1;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; h += dilation_h) {
      for (int w = wstart; w < wend; w += dilation_w) {
        if (ScalarConvert<Dtype, AccType>::to(bottom_data[h * width + w]) > maxval) {
          maxidx = h * width + w;
          maxval = ScalarConvert<Dtype, AccType>::to(bottom_data[maxidx]);
        }
      }
    }
    top_data[index] = ScalarConvert<AccType, Dtype>::to(maxval);
    top_mask[index] = maxidx + TH_INDEX_BASE;
  }
}


template <typename Dtype, typename AccType>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* top_diff,
    const int64_t* top_mask, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart =
        (h + pad_h < ((kernel_h - 1) * dilation_h + 1)) ? 0 : (h + pad_h - ((kernel_h - 1) * dilation_h + 1)) / stride_h + 1;
    int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    int pwstart =
        (w + pad_w < ((kernel_w - 1) * dilation_w + 1)) ? 0 : (w + pad_w - ((kernel_w - 1) * dilation_w + 1)) / stride_w + 1;
    int pwend = min((w + pad_w) / stride_w + 1, pooled_width);

    AccType gradient = AccType(0);
    int offset = (n * channels + c) * pooled_height * pooled_width;
    top_diff += offset;
    top_mask += offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if (top_mask[ph * pooled_width + pw] - TH_INDEX_BASE == h * width + w) {
          gradient += ScalarConvert<Dtype, AccType>::to(top_diff[ph * pooled_width + pw]);
        }
      }
    }
    bottom_diff[index] = ScalarConvert<AccType, Dtype>::to(gradient);
  }
}

#include "generic/SpatialDilatedMaxPooling.cu"
#include "THCGenerateFloatTypes.h"
