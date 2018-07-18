#include "THCUNN.h"
#include "THCTensor.hpp"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCNumerics.cuh"
#include "common.h"

// kernels borrowed from Caffe
template <typename Dtype, typename AccType>
__global__ void MaxPoolForward(const int64_t nthreads, const Dtype* bottom_data,
    const int64_t num, const int64_t channels, const int64_t height,
    const int64_t width, const int64_t pooled_height, const int64_t pooled_width,
    const int64_t kernel_h, const int64_t kernel_w, const int64_t stride_h,
    const int64_t stride_w, const int64_t pad_h, const int64_t pad_w,
    const int64_t dilation_h, const int64_t dilation_w, Dtype* top_data,
    int64_t* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int64_t pw = index % pooled_width;
    int64_t ph = (index / pooled_width) % pooled_height;
    int64_t c = (index / pooled_width / pooled_height) % channels;
    int64_t n = index / pooled_width / pooled_height / channels;
    int64_t hstart = ph * stride_h - pad_h;
    int64_t wstart = pw * stride_w - pad_w;
    int64_t hend = min(hstart + (kernel_h - 1) * dilation_h + 1, height);
    int64_t wend = min(wstart + (kernel_w - 1) * dilation_w + 1, width);
    while(hstart < 0)
      hstart += dilation_h;
    while(wstart < 0)
      wstart += dilation_w;
    AccType maxval = THCNumerics<AccType>::min();
    int64_t maxidx = -1;
    bottom_data += (n * channels + c) * height * width;
    for (int64_t h = hstart; h < hend; h += dilation_h) {
      for (int64_t w = wstart; w < wend; w += dilation_w) {
        Dtype val = bottom_data[h * width + w];
        if ((ScalarConvert<Dtype, AccType>::to(val) > maxval) || THCNumerics<Dtype>::isnan(val)) {
          maxidx = h * width + w;
          maxval = ScalarConvert<Dtype, AccType>::to(val);
        }
      }
    }
    top_data[index] = ScalarConvert<AccType, Dtype>::to(maxval);
    top_mask[index] = maxidx + TH_INDEX_BASE;
  }
}

const int64_t BACKWARD_THREADS = 256;

template <typename Dtype, typename AccType>
__launch_bounds__(BACKWARD_THREADS,2048/BACKWARD_THREADS)
__global__ void MaxPoolBackward(const int64_t nthreads, const Dtype* top_diff,
    const int64_t* top_mask, const int64_t num, const int64_t channels,
    const int64_t height, const int64_t width, const int64_t pooled_height,
    const int64_t pooled_width, const int64_t kernel_h, const int64_t kernel_w,
    const int64_t stride_h, const int64_t stride_w, const int64_t pad_h, const int64_t pad_w,
    const int64_t dilation_h, const int64_t dilation_w,
    Dtype* bottom_diff) {
    CUDA_KERNEL_LOOP(index, height*width) {
    int64_t h = index/width;
    int64_t w = index - h * width;
//get some templating performance benefits without actually templating
    int64_t phstart, phend, pwstart, pwend;
    if (stride_h == 1) {
       phstart =
        (h + pad_h < ((kernel_h - 1) * dilation_h + 1)) ? 0 : (h + pad_h - ((kernel_h - 1) * dilation_h + 1))  + 1;
       phend = min((h + pad_h)  + 1, pooled_height);
    } else if (stride_h == 2) {
       phstart =
        (h + pad_h < ((kernel_h - 1) * dilation_h + 1)) ? 0 : (h + pad_h - ((kernel_h - 1) * dilation_h + 1)) / 2  + 1;
       phend = min((h + pad_h) / 2  + 1, pooled_height);
    } else {
       phstart =
        (h + pad_h < ((kernel_h - 1) * dilation_h + 1)) ? 0 : (h + pad_h - ((kernel_h - 1) * dilation_h + 1)) / stride_h  + 1;
       phend = min((h + pad_h) / stride_h  + 1, pooled_height);
    }
    if (stride_w == 1) {
        pwstart =
        (w + pad_w < ((kernel_w - 1) * dilation_w + 1)) ? 0 : (w + pad_w - ((kernel_w - 1) * dilation_w + 1)) + 1;
        pwend = min((w + pad_w) + 1, pooled_width);
    } else if (stride_w == 2) {
        pwstart =
        (w + pad_w < ((kernel_w - 1) * dilation_w + 1)) ? 0 : (w + pad_w - ((kernel_w - 1) * dilation_w + 1)) / 2 + 1;
        pwend = min((w + pad_w) / 2 + 1, pooled_width);
    } else {
        pwstart =
        (w + pad_w < ((kernel_w - 1) * dilation_w + 1)) ? 0 : (w + pad_w - ((kernel_w - 1) * dilation_w + 1)) / stride_w + 1;
        pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    }
    for (int64_t n = blockIdx.y; n < num; n += gridDim.y)
       for (int64_t c = blockIdx.z; c < channels; c+= gridDim.z) { 

        AccType gradient = AccType(0);
        int64_t offset = (n * channels + c) * pooled_height * pooled_width;
        top_diff += offset;
        top_mask += offset;
//get some templating performance benefits without actually templating
        if ((phstart + 1 != phend) || (pwstart + 1 != pwend)) {
        for (int64_t ph = phstart; ph < phend; ++ph) {
          for (int64_t pw = pwstart; pw < pwend; ++pw) {
            if (top_mask[ph * pooled_width + pw] - TH_INDEX_BASE == h * width + w) {
              gradient += ScalarConvert<Dtype, AccType>::to(top_diff[ph * pooled_width + pw]);
            }
          }
        }
        } else {
            if (top_mask[phstart * pooled_width + pwstart] - TH_INDEX_BASE == h * width + w) {
              gradient += ScalarConvert<Dtype, AccType>::to(top_diff[phstart * pooled_width + pwstart]);
            }  
        }
        bottom_diff[(n*channels+c)*height*width+index] = ScalarConvert<AccType, Dtype>::to(gradient);
      }
  }
}

#include "generic/SpatialDilatedMaxPooling.cu"
#include "THCGenerateFloatTypes.h"
