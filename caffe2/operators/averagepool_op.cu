#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/averagepool_op.h"

namespace caffe2 {

namespace {
template <typename T>
__global__ void AveragePoolForwardNCHW(
    const int nthreads, const T* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_t, const int pad_l, T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_t;
    int wstart = pw * stride_w - pad_l;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    T output = 0;
    bottom_data += n * channels * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int idx = c * height * width + h * width + w;
        output += bottom_data[idx];
      }
    }
    int pool_size = (hend - hstart) * (wend - wstart);
    top_data[index] = output / pool_size;
  }
}

template <typename T>
__global__ void AveragePoolForwardNHWC(
    const int nthreads, const T* bottom_data,
    const int num, const int height, const int width,
    const int channels, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_t, const int pad_l, T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int c = index % channels;
    int pw = (index / channels) % pooled_width;
    int ph = (index / channels / pooled_width) % pooled_height;
    int n = index / channels / pooled_width / pooled_height;
    int hstart = ph * stride_h - pad_t;
    int wstart = pw * stride_w - pad_l;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    T output = 0;
    bottom_data += n * height * width * channels;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        output += bottom_data[(h * width + w) * channels + c];
      }
    }
    int pool_size = (hend - hstart) * (wend - wstart);
    top_data[index] = output / pool_size;
  }
}

template <typename T>
__global__ void AvePoolBackwardNCHW(const int nthreads,
    const T* const top_diff, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_t,
    const int pad_l, T* const bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_l;
    const int h = (index / width) % height + pad_t;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    T gradient = 0;
    const T* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_t;
        int wstart = pw * stride_w - pad_l;
        int hend = min(hstart + kernel_h, height);
        int wend = min(wstart + kernel_w, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename T>
__global__ void AvePoolBackwardNHWC(const int nthreads,
    const T* const top_diff, const int num, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_t,
    const int pad_l, T* const bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int c = index % channels;
    const int w = index / channels % width + pad_l;
    const int h = (index / channels / width) % height + pad_t;
    const int n = index / channels / width / height;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    T gradient = 0;
    const T* const top_diff_slice =
        top_diff + n * pooled_height * pooled_width * channels + c;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_t;
        int wstart = pw * stride_w - pad_l;
        int hend = min(hstart + kernel_h, height);
        int wend = min(wstart + kernel_w, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient +=
            top_diff_slice[(ph * pooled_width + pw) * channels] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}

}  // namespace

template <>
bool AveragePoolOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase<CUDAContext>::SetOutputSize(X, Y, X.dim(1));
  int output_size = Y->size();
  AveragePoolForwardNCHW<float><<<CAFFE_GET_BLOCKS(output_size),
                              CAFFE_CUDA_NUM_THREADS,
                              0, device_context_.cuda_stream()>>>(
      output_size, X.data<float>(), X.dim(0), X.dim(1), X.dim(2), X.dim(3),
      Y->dim(2), Y->dim(3), kernel_h_, kernel_w_, stride_h_, stride_w_,
      pad_t_, pad_l_, Y->mutable_data<float>());
  return true;
}

template <>
bool AveragePoolOp<float, CUDAContext>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase<CUDAContext>::SetOutputSize(X, Y, X.dim(3));
  int output_size = Y->size();
  AveragePoolForwardNHWC<float><<<CAFFE_GET_BLOCKS(output_size),
                              CAFFE_CUDA_NUM_THREADS,
                              0, device_context_.cuda_stream()>>>(
      output_size, X.data<float>(), X.dim(0), X.dim(1), X.dim(2), X.dim(3),
      Y->dim(1), Y->dim(2), kernel_h_, kernel_w_, stride_h_, stride_w_,
      pad_t_, pad_l_, Y->mutable_data<float>());
  return true;
}

template <>
bool AveragePoolGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto& dY = Input(1);
  CAFFE_CHECK_EQ(dY.ndim(), 4);
  auto* dX = Output(0);
  dX->ReshapeLike(X);
  ConvPoolOpBase<CUDAContext>::ComputePads(X.dim(2), X.dim(3));
  AvePoolBackwardNCHW<float><<<CAFFE_GET_BLOCKS(X.size()),
                               CAFFE_CUDA_NUM_THREADS,
                               0, device_context_.cuda_stream()>>>(
      X.size(), dY.data<float>(), X.dim(0), X.dim(1), X.dim(2), X.dim(3),
      dY.dim(2), dY.dim(3), kernel_h_, kernel_w_, stride_h_, stride_w_,
      pad_t_, pad_l_, dX->mutable_data<float>());
  return true;
}

template <>
bool AveragePoolGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto& dY = Input(1);
  CAFFE_CHECK_EQ(dY.ndim(), 4);
  auto* dX = Output(0);
  dX->ReshapeLike(X);
  ConvPoolOpBase<CUDAContext>::ComputePads(X.dim(1), X.dim(2));
  AvePoolBackwardNHWC<float><<<CAFFE_GET_BLOCKS(X.size()),
                               CAFFE_CUDA_NUM_THREADS,
                               0, device_context_.cuda_stream()>>>(
      X.size(), dY.data<float>(), X.dim(0), X.dim(1), X.dim(2), X.dim(3),
      dY.dim(1), dY.dim(2), kernel_h_, kernel_w_, stride_h_, stride_w_,
      pad_t_, pad_l_, dX->mutable_data<float>());
  return true;
}


namespace {
REGISTER_CUDA_OPERATOR(AveragePool, AveragePoolOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(AveragePoolGradient,
                       AveragePoolGradientOp<float, CUDAContext>);
}  // namespace
}  // namespace caffe2
