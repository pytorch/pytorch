// TODO: reduce the apparent redundancy of all the code below.
#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/pool_op.h"

namespace caffe2 {
namespace {
class LpPool {};
} // namespace

namespace {
template <typename T>
inline __device__ T cuda_pow(T x, T y);

template <typename T>
inline __device__ T cuda_abs(T x);

template <>
inline __device__ float cuda_pow<float>(float x, float y) {
  return powf(x, y);
}
template <>
inline __device__ double cuda_pow<double>(double x, double y) {
  return pow(x, y);
}

template <>
inline __device__ float cuda_abs(float x) {
  return fabsf(x);
}
template <>
inline __device__ double cuda_abs(double x) {
  return fabs(x);
}
}

namespace {
template <typename T>
__global__ void LpPoolForwardNCHW(
    const int nthreads,
    const T* bottom_data,
    const int num,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    T* top_data,
    const T p) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int pw = n % pooled_width;
    n /= pooled_width;
    int ph = n % pooled_height;
    n /= pooled_height;
    int c = n % channels;
    n /= channels;
    int hstart = ph * stride_h - pad_t;
    int wstart = pw * stride_w - pad_l;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    top_data[index] = 0;
    int bottom_offset = (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        top_data[index] += cuda_pow<T>(
            cuda_abs(bottom_data[bottom_offset + h * width + w]), p);
      }
    }
    top_data[index] = cuda_pow<T>(top_data[index], 1.0 / p);
  }
}

template <typename T>
__global__ void LpPoolForwardNHWC(
    const int nthreads,
    const T* bottom_data,
    const int num,
    const int height,
    const int width,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    T* top_data,
    const T p) {
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
    int bottom_offset = n * height * width * channels + c;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        output += cuda_pow<T>(
            cuda_abs(bottom_data[bottom_offset + (h * width + w) * channels]),
            p);
      }
    }
    top_data[index] = cuda_pow<T>(output, 1.0 / p);
  }
}

template <typename T>
__global__ void LpPoolBackwardNCHW(
    const int nthreads,
    const T* const top_diff,
    const T* const top_data,
    const T* const bottom_data,
    const int num,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    T* const bottom_diff,
    const int p) {
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
    const T* const top_data_slice =
        top_data + (n * channels + c) * pooled_height * pooled_width;

    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_t;
        int wstart = pw * stride_w - pad_l;
        int hend = min(hstart + kernel_h, height);
        int wend = min(wstart + kernel_w, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        gradient += top_diff_slice[ph * pooled_width + pw] *
            bottom_data[index] *
            cuda_pow<T>(cuda_abs(bottom_data[index]), p - 2) /
            cuda_pow<T>(top_data_slice[ph * pooled_width + pw], p - 1);
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename T>
__global__ void LpPoolBackwardNHWC(
    const int nthreads,
    const T* const top_diff,
    const T* const top_data,
    const T* const bottom_data,
    const int num,
    const int height,
    const int width,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    T* const bottom_diff,
    const T p) {
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
    const T* const top_data_slice =
        top_data + n * pooled_height * pooled_width * channels + c;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_t;
        int wstart = pw * stride_w - pad_l;
        int hend = min(hstart + kernel_h, height);
        int wend = min(wstart + kernel_w, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        gradient += top_diff_slice[(ph * pooled_width + pw) * channels] *
            bottom_data[index] *
            cuda_pow<T>(cuda_abs(bottom_data[index]), p - 2) /
            cuda_pow<T>(top_data_slice[(ph * pooled_width + pw) * channels],
                        p - 1);
      }
    }
    bottom_diff[index] = gradient;
  }
}

} // namespace

template <>
bool PoolOp<float, CUDAContext, LpPool>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase<CUDAContext>::SetOutputSize(X, Y, X.dim32(1));
  int output_size = Y->size();
  LpPoolForwardNCHW<float><<<
      CAFFE_GET_BLOCKS(output_size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      output_size,
      X.data<float>(),
      X.dim32(0),
      X.dim32(1),
      X.dim32(2),
      X.dim32(3),
      Y->dim32(2),
      Y->dim32(3),
      kernel_h(),
      kernel_w(),
      stride_h(),
      stride_w(),
      pad_t(),
      pad_l(),
      Y->mutable_data<float>(),
      OperatorBase::GetSingleArgument<float>("p", 2.0));
  return true;
}

template <>
bool PoolOp<float, CUDAContext, LpPool>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase<CUDAContext>::SetOutputSize(X, Y, X.dim32(3));
  int output_size = Y->size();
  LpPoolForwardNHWC<float><<<
      CAFFE_GET_BLOCKS(output_size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      output_size,
      X.data<float>(),
      X.dim32(0),
      X.dim32(1),
      X.dim32(2),
      X.dim32(3),
      Y->dim32(1),
      Y->dim32(2),
      kernel_h(),
      kernel_w(),
      stride_h(),
      stride_w(),
      pad_t(),
      pad_l(),
      Y->mutable_data<float>(),
      OperatorBase::GetSingleArgument<float>("p", 2.0));
  return true;
}

template <>
bool PoolGradientOp<float, CUDAContext, LpPool>::
    RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dY = Input(2);
  CAFFE_ENFORCE_EQ(dY.ndim(), 4);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  ConvPoolOpBase<CUDAContext>::ComputePads({X.dim32(2), X.dim32(3)});
  LpPoolBackwardNCHW<float><<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(),
      dY.data<float>(),
      Y.data<float>(),
      X.data<float>(),
      X.dim32(0),
      X.dim32(1),
      X.dim32(2),
      X.dim32(3),
      dY.dim32(2),
      dY.dim32(3),
      kernel_h(),
      kernel_w(),
      stride_h(),
      stride_w(),
      pad_t(),
      pad_l(),
      dX->mutable_data<float>(),
      OperatorBase::GetSingleArgument<float>("p", 2.0));
  return true;
}

template <>
bool PoolGradientOp<float, CUDAContext, LpPool>::
    RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dY = Input(2);
  CAFFE_ENFORCE_EQ(dY.ndim(), 4);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  ConvPoolOpBase<CUDAContext>::ComputePads({X.dim32(1), X.dim32(2)});
  LpPoolBackwardNHWC<float><<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(),
      dY.data<float>(),
      Y.data<float>(),
      X.data<float>(),
      X.dim32(0),
      X.dim32(1),
      X.dim32(2),
      X.dim32(3),
      dY.dim32(1),
      dY.dim32(2),
      kernel_h(),
      kernel_w(),
      stride_h(),
      stride_w(),
      pad_t(),
      pad_l(),
      dX->mutable_data<float>(),
      OperatorBase::GetSingleArgument<float>("p", 2.0));
  return true;
}

REGISTER_CUDA_OPERATOR(LpPool, PoolOp<float, CUDAContext, LpPool>);
REGISTER_CUDA_OPERATOR(
    LpPoolGradient,
    PoolGradientOp<float, CUDAContext, LpPool>);
}
