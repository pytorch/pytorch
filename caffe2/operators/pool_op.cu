// TODO: reduce the apparent redundancy of all the code below.
#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/pool_op.h"

namespace caffe2 {
namespace {
class AveragePool {};
class MaxPool {};
}  // namespace

namespace {
template <typename T>
__global__ void AveragePoolForwardNCHW(
    const int nthreads, const T* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_t, const int pad_l, T* top_data) {
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
        top_data[index] += bottom_data[bottom_offset + h * width + w];
      }
    }
    top_data[index] /= (hend - hstart) * (wend - wstart);
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
    int bottom_offset = n * height * width * channels + c;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        output += bottom_data[bottom_offset + (h * width + w) * channels];
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
bool PoolOp<float, CUDAContext, AveragePool>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase<CUDAContext>::SetOutputSize(X, Y, X.dim32(1));
  int output_size = Y->size();
  AveragePoolForwardNCHW<float><<<
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
      Y->mutable_data<float>());
  return true;
}

template <>
bool PoolOp<float, CUDAContext, AveragePool>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase<CUDAContext>::SetOutputSize(X, Y, X.dim32(3));
  int output_size = Y->size();
  AveragePoolForwardNHWC<float><<<
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
      Y->mutable_data<float>());
  return true;
}

template <>
bool PoolGradientOp<float, CUDAContext, AveragePool>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto& dY = Input(2);
  CAFFE_ENFORCE_EQ(dY.ndim(), 4);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  ConvPoolOpBase<CUDAContext>::ComputePads({X.dim32(2), X.dim32(3)});
  AvePoolBackwardNCHW<float><<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(),
      dY.data<float>(),
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
      dX->mutable_data<float>());
  return true;
}

template <>
bool PoolGradientOp<float, CUDAContext, AveragePool>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto& dY = Input(2);
  CAFFE_ENFORCE_EQ(dY.ndim(), 4);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  ConvPoolOpBase<CUDAContext>::ComputePads({X.dim32(1), X.dim32(2)});
  AvePoolBackwardNHWC<float><<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(),
      dY.data<float>(),
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
      dX->mutable_data<float>());
  return true;
}


namespace {
template <typename T>
__global__ void MaxPoolForwardNCHW(const int nthreads, const T* bottom_data,
    const int channels, const int height,
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
    T maxval = -FLT_MAX;
    const T* bdata_offset = bottom_data + n * channels * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int idx = c * height * width + h * width + w;
        if (bdata_offset[idx] > maxval) {
          maxval = bdata_offset[idx];
        }
      }
    }
    top_data[index] = maxval;
  }
}

template <typename T>
__global__ void MaxPoolForwardNHWC(const int nthreads, const T* bottom_data,
    const int height, const int width,
    const int channels, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_t, const int pad_l, T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int wstart = (n % pooled_width) * stride_w - pad_l;
    n /= pooled_width;
    int hstart = (n % pooled_height) * stride_h - pad_t;
    n /= pooled_height;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    T maxval = -FLT_MAX;
    const T* bdata_offset = bottom_data + n * height * width * channels;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int idx = (h * width + w) * channels + c;
        if (bdata_offset[idx] > maxval) {
          maxval = bdata_offset[idx];
        }
      }
    }
    top_data[index] = maxval;
  }
}

template <typename T>
__global__ void MaxPoolBackwardNCHW(const int nthreads,
    const T* const bottom_data, const T* const top_data,
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
    const int top_offset =
        (n * channels + c) * pooled_height * pooled_width;
    bottom_diff[index] = 0;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        int top_local_offset = top_offset + ph * pooled_width + pw;
        if (bottom_data[index] == top_data[top_local_offset]) {
          bottom_diff[index] += top_diff[top_local_offset];
        }
      }
    }
  }
}

template <typename T>
__global__ void MaxPoolBackwardNHWC(const int nthreads,
    const T* const bottom_data, const T* const top_data,
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
    const int top_offset =
        n * pooled_height * pooled_width * channels + c;
    bottom_diff[index] = 0;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        int top_local_offset = top_offset + (ph * pooled_width + pw) * channels;
        if (bottom_data[index] == top_data[top_local_offset]) {
          bottom_diff[index] += top_diff[top_local_offset];
        }
      }
    }
  }
}
}  // namespace

template <>
bool PoolOp<float, CUDAContext, MaxPool>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase<CUDAContext>::SetOutputSize(X, Y, X.dim32(1));
  int output_size = Y->size();
  MaxPoolForwardNCHW<float><<<
      CAFFE_GET_BLOCKS(output_size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      output_size,
      X.data<float>(),
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
      Y->mutable_data<float>());
  return true;
}

template <>
bool PoolOp<float, CUDAContext, MaxPool>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase<CUDAContext>::SetOutputSize(X, Y, X.dim32(3));
  int output_size = Y->size();
  MaxPoolForwardNHWC<float><<<
      CAFFE_GET_BLOCKS(output_size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      output_size,
      X.data<float>(),
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
      Y->mutable_data<float>());
  return true;
}

template <>
bool PoolGradientOp<float, CUDAContext, MaxPool>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dY = Input(2);
  CAFFE_ENFORCE_EQ(dY.ndim(), 4);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  ConvPoolOpBase<CUDAContext>::ComputePads({X.dim32(2), X.dim32(3)});
  MaxPoolBackwardNCHW<float><<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(),
      X.data<float>(),
      Y.data<float>(),
      dY.data<float>(),
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
      dX->mutable_data<float>());
  return true;
}

template <>
bool PoolGradientOp<float, CUDAContext, MaxPool>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dY = Input(2);
  CAFFE_ENFORCE_EQ(dY.ndim(), 4);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  ConvPoolOpBase<CUDAContext>::ComputePads({X.dim32(1), X.dim32(2)});
  MaxPoolBackwardNHWC<float><<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(),
      X.data<float>(),
      Y.data<float>(),
      dY.data<float>(),
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
      dX->mutable_data<float>());
  return true;
}

namespace {
REGISTER_CUDA_OPERATOR(AveragePool, PoolOp<float, CUDAContext, AveragePool>);
REGISTER_CUDA_OPERATOR(AveragePoolGradient,
                       PoolGradientOp<float, CUDAContext, AveragePool>);
REGISTER_CUDA_OPERATOR(MaxPool, PoolOp<float, CUDAContext, MaxPool>);
REGISTER_CUDA_OPERATOR(MaxPoolGradient,
                       PoolGradientOp<float, CUDAContext, MaxPool>);
}  // namespace
}  // namespace caffe2
