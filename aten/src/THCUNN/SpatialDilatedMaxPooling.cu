#include <THCUNN/THCUNN.h>
#include <THC/THCTensor.hpp>
#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>
#include <THC/THCNumerics.cuh>
#include <THCUNN/common.h>
#include <c10/macros/Macros.h>

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
        Dtype val = bottom_data[h * width + w];
        if ((ScalarConvert<Dtype, AccType>::to(val) > maxval) || THCNumerics<Dtype>::isnan(val)) {
          maxidx = h * width + w;
          maxval = ScalarConvert<Dtype, AccType>::to(val);
        }
      }
    }
    top_data[index] = ScalarConvert<AccType, Dtype>::to(maxval);
    top_mask[index] = maxidx;
  }
}

const int BACKWARD_THREADS = 256;

template <typename Dtype, typename AccType>
#if defined (__HIP_PLATFORM_HCC__)
C10_LAUNCH_BOUNDS_2(BACKWARD_THREADS, 4)
#else
C10_LAUNCH_BOUNDS_2(BACKWARD_THREADS, 8)
#endif
__global__ void MaxPoolBackward(const int nthreads, const Dtype* top_diff,
    const int64_t* top_mask, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    Dtype* bottom_diff) {
    CUDA_KERNEL_LOOP(index, height*width) {
    int h = index/width;
    int w = index - h * width;
//get some templating performance benefits without actually templating
    int phstart, phend, pwstart, pwend;
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
    for (int n = blockIdx.y; n < num; n += gridDim.y)
       for (int c = blockIdx.z; c < channels; c+= gridDim.z) {

        AccType gradient = AccType(0);
        int offset = (n * channels + c) * pooled_height * pooled_width;
        top_diff += offset;
        top_mask += offset;
//get some templating performance benefits without actually templating
        if ((phstart + 1 != phend) || (pwstart + 1 != pwend)) {
        for (int ph = phstart; ph < phend; ++ph) {
          for (int pw = pwstart; pw < pwend; ++pw) {
            if (top_mask[ph * pooled_width + pw] == h * width + w) {
              gradient += ScalarConvert<Dtype, AccType>::to(top_diff[ph * pooled_width + pw]);
            }
          }
        }
        } else {
            if (top_mask[phstart * pooled_width + pwstart] == h * width + w) {
              gradient += ScalarConvert<Dtype, AccType>::to(top_diff[phstart * pooled_width + pwstart]);
            }
        }
        bottom_diff[(n*channels+c)*height*width+index] = ScalarConvert<AccType, Dtype>::to(gradient);
      }
  }
}

#include <THCUNN/generic/SpatialDilatedMaxPooling.cu>
#include <THC/THCGenerateFloatTypes.h>
