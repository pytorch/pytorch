#include <algorithm>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/pad_op.h"

namespace caffe2 {

namespace {
template <typename T>
__global__ void PadImageConstNCHW(
    const int nthreads, const T* const bottom_data, const int num,
    const int channels, const int height, const int width,
    const int padded_height, const int padded_width,
    const int pad_t, const int pad_l, T value, T* const top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int nc = index / padded_width;
    const int pw = index % padded_width;
    const int ph = nc % padded_height;
    nc /= padded_height;
    const int h = ph - pad_t;
    const int w = pw - pad_l;
    top_data[index] = (h < 0 || w < 0 || h >= height || w >= width)
        ? value
        : bottom_data[(nc * height + h) * width + w];
  }
}

template <typename T>
__global__ void PadImageReflectNCHW(
    const int nthreads, const T* const bottom_data, const int num,
    const int channels, const int height, const int width,
    const int padded_height, const int padded_width,
    const int pad_t, const int pad_l, T* const top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int nc = index / padded_width;
    const int pw = index % padded_width;
    const int ph = nc % padded_height;
    nc /= padded_height;
    int h = ph - pad_t;
    int w = pw - pad_l;
    h = max(h, -h);
    w = max(w, -w);
    h = min(h, 2 * height - h - 2);
    w = min(w, 2 * width - w - 2);
    top_data[index] = bottom_data[(nc * height + h) * width + w];
  }
}

template <typename T>
__global__ void PadImageEdgeNCHW(
    const int nthreads, const T* const bottom_data, const int num,
    const int channels, const int height, const int width,
    const int padded_height, const int padded_width,
    const int pad_t, const int pad_l, T* const top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int nc = index / padded_width;
    const int pw = index % padded_width;
    const int ph = nc % padded_height;
    nc /= padded_height;
    const int h = min(height - 1, max(ph - pad_t, 0));
    const int w = min(width - 1, max(pw - pad_l, 0));
    top_data[index] = bottom_data[(nc * height + h) * width + w];
  }
}

template <typename T>
__global__ void PadImageConstNHWC(
    const int nthreads, const T* const bottom_data, const int num,
    const int height, const int width, const int channels,
    const int padded_height, const int padded_width,
    const int pad_t, const int pad_l, T value, T* const top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / channels;
    const int c = index % channels;
    const int pw = n % padded_width;
    n /= padded_width;
    const int ph = n % padded_height;
    n /= padded_height;
    const int h = ph - pad_t;
    const int w = pw - pad_l;
    top_data[index] = (h < 0 || w < 0 || h >= height || w >= width)
        ? value
        : bottom_data[((n * height + h) * width + w) * channels + c];
  }
}

template <typename T>
__global__ void PadImageReflectNHWC(
    const int nthreads, const T* const bottom_data, const int num,
    const int height, const int width, const int channels,
    const int padded_height, const int padded_width,
    const int pad_t, const int pad_l, T* const top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / channels;
    const int c = index % channels;
    const int pw = n % padded_width;
    n /= padded_width;
    const int ph = n % padded_height;
    n /= padded_height;
    int h = ph - pad_t;
    int w = pw - pad_l;
    h = max(h, -h);
    w = max(w, -w);
    h = min(h, 2 * height - h - 2);
    w = min(w, 2 * width - w - 2);
    top_data[index] =
        bottom_data[((n * height + h) * width + w) * channels + c];
  }
}

template <typename T>
__global__ void PadImageEdgeNHWC(
    const int nthreads, const T* const bottom_data, const int num,
    const int height, const int width, const int channels,
    const int padded_height, const int padded_width,
    const int pad_t, const int pad_l, T* const top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / channels;
    const int c = index % channels;
    const int pw = n % padded_width;
    n /= padded_width;
    const int ph = n % padded_height;
    n /= padded_height;
    const int h = min(height - 1, max(ph - pad_t, 0));
    const int w = min(width - 1, max(pw - pad_l, 0));
    top_data[index] =
        bottom_data[((n * height + h) * width + w) * channels + c];
  }
}

template <typename T>
__global__ void PadImageGradientConstNCHW(
    const int nthreads, const T* const top_diff, const int num,
    const int channels, const int height, const int width,
    const int padded_height, const int padded_width,
    const int pad_t, const int pad_l, T* const bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int nc = index / width;
    const int pw = index % width + pad_l;
    const int ph = nc % height + pad_t;
    nc /= height;
    bottom_diff[index] =
        top_diff[(nc * padded_height + ph) * padded_width + pw];
  }
}

template <typename T>
__global__ void PadImageGradientReflectNCHW(
    const int nthreads, const T* const top_diff, const int num,
    const int channels, const int height, const int width,
    const int padded_height, const int padded_width,
    const int pad_t, const int pad_l, T* const bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int nc = index / padded_width;
    const int pw = index % padded_width;
    const int ph = nc % padded_height;
    nc /= padded_height;
    int h = ph - pad_t;
    int w = pw - pad_l;
    h = max(h, -h);
    w = max(w, -w);
    h = min(h, 2 * height - h - 2);
    w = min(w, 2 * width - w - 2);
    atomicAdd(&bottom_diff[(nc * height + h) * width + w], top_diff[index]);
  }
}

template <typename T>
__global__ void PadImageGradientEdgeNCHW(
    const int nthreads, const T* const top_diff, const int num,
    const int channels, const int height, const int width,
    const int padded_height, const int padded_width,
    const int pad_t, const int pad_l, T* const bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int nc = index / padded_width;
    const int pw = index % padded_width;
    const int ph = nc % padded_height;
    nc /= padded_height;
    const int h = min(height - 1, max(ph - pad_t, 0));
    const int w = min(width - 1, max(pw - pad_l, 0));
    atomicAdd(&bottom_diff[(nc * height + h) * width + w], top_diff[index]);
  }
}

template <typename T>
__global__ void PadImageGradientConstNHWC(
    const int nthreads, const T* const top_diff, const int num,
    const int height, const int width, const int channels,
    const int padded_height, const int padded_width,
    const int pad_t, const int pad_l, T* const bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / channels;
    const int c = index % channels;
    const int pw = n % width + pad_l;
    n /= width;
    const int ph = n % height + pad_t;
    n /= height;
    bottom_diff[index] =
        top_diff[((n * padded_height + ph) * padded_width + pw) * channels + c];
  }
}

template <typename T>
__global__ void PadImageGradientReflectNHWC(
    const int nthreads, const T* const top_diff, const int num,
    const int height, const int width, const int channels,
    const int padded_height, const int padded_width,
    const int pad_t, const int pad_l, T* const bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / channels;
    const int c = index % channels;
    const int pw = n % padded_width;
    n /= padded_width;
    const int ph = n % padded_height;
    n /= padded_height;
    int h = ph - pad_t;
    int w = pw - pad_l;
    h = max(h, -h);
    w = max(w, -w);
    h = min(h, 2 * height - h - 2);
    w = min(w, 2 * width - w - 2);
    atomicAdd(
        &bottom_diff[((n * height + h) * width + w) * channels + c],
        top_diff[index]);
  }
}

template <typename T>
__global__ void PadImageGradientEdgeNHWC(
    const int nthreads, const T* const top_diff, const int num,
    const int height, const int width, const int channels,
    const int padded_height, const int padded_width,
    const int pad_t, const int pad_l, T* const bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / channels;
    const int c = index % channels;
    const int pw = n % padded_width;
    n /= padded_width;
    const int ph = n % padded_height;
    n /= padded_height;
    const int h = min(height - 1, max(ph - pad_t, 0));
    const int w = min(width - 1, max(pw - pad_l, 0));
    atomicAdd(
        &bottom_diff[((n * height + h) * width + w) * channels + c],
        top_diff[index]);
  }
}

}  // namespace

template <>
bool PadImageOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  const int num = X.dim32(0);
  const int channels = X.dim32(1);
  const int height = X.dim32(2);
  const int width = X.dim32(3);
  auto sizes = ConvPoolOpBase<CUDAContext>::GetOutputSize(X, channels);
  auto* Y = Output(0, sizes, at::dtype<float>());

  const int output_size = Y->numel();
  const int padded_height = Y->dim32(2);
  const int padded_width = Y->dim32(3);
  const float* Xdata = X.data<float>();
  float* Ydata = Y->template mutable_data<float>();

  switch (mode_) {
    case PadMode::CONSTANT:
      PadImageConstNCHW<float><<<
          CAFFE_GET_BLOCKS(output_size),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          output_size,
          Xdata,
          num,
          channels,
          height,
          width,
          padded_height,
          padded_width,
          pad_t(),
          pad_l(),
          value_,
          Ydata);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case PadMode::REFLECT:
      PadImageReflectNCHW<float><<<
          CAFFE_GET_BLOCKS(output_size),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          output_size,
          Xdata,
          num,
          channels,
          height,
          width,
          padded_height,
          padded_width,
          pad_t(),
          pad_l(),
          Ydata);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case PadMode::EDGE:
      PadImageEdgeNCHW<float><<<
          CAFFE_GET_BLOCKS(output_size),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          output_size,
          Xdata,
          num,
          channels,
          height,
          width,
          padded_height,
          padded_width,
          pad_t(),
          pad_l(),
          Ydata);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
  }

  return true;
}

template<>
bool PadImageOp<float, CUDAContext>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  const int num = X.dim32(0);
  const int height = X.dim32(1);
  const int width = X.dim32(2);
  const int channels = X.dim32(3);
  auto sizes = ConvPoolOpBase<CUDAContext>::GetOutputSize(X, channels);
  auto* Y = Output(0, sizes, at::dtype<float>());

  const int output_size = Y->numel();
  const int padded_height = Y->dim32(1);
  const int padded_width = Y->dim32(2);
  const float* Xdata = X.data<float>();
  float* Ydata = Y->template mutable_data<float>();

  switch (mode_) {
    case PadMode::CONSTANT:
      PadImageConstNHWC<float><<<
          CAFFE_GET_BLOCKS(output_size),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          output_size,
          Xdata,
          num,
          height,
          width,
          channels,
          padded_height,
          padded_width,
          pad_t(),
          pad_l(),
          value_,
          Ydata);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case PadMode::REFLECT:
      PadImageReflectNHWC<float><<<
          CAFFE_GET_BLOCKS(output_size),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          output_size,
          Xdata,
          num,
          height,
          width,
          channels,
          padded_height,
          padded_width,
          pad_t(),
          pad_l(),
          Ydata);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case PadMode::EDGE:
      PadImageEdgeNHWC<float><<<
          CAFFE_GET_BLOCKS(output_size),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          output_size,
          Xdata,
          num,
          height,
          width,
          channels,
          padded_height,
          padded_width,
          pad_t(),
          pad_l(),
          Ydata);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
  }

  return true;
}

template<>
bool PadImageGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW() {
  auto& dY = Input(0);

  auto* dX = Output(0, { dY.dim32(0),
      dY.dim32(1),
      dY.dim32(2) - pad_t() - pad_b(),
      dY.dim32(3) - pad_l() - pad_r()}, at::dtype<float>());
  const int input_size = dY.numel();
  const int padded_height = dY.dim32(2);
  const int padded_width = dY.dim32(3);
  const int output_size = dX->numel();
  const int num = dX->dim32(0);
  const int channels = dX->dim32(1);
  const int height = dX->dim32(2);
  const int width = dX->dim32(3);
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->template mutable_data<float>();
  math::Set<float, CUDAContext>(output_size, 0, dXdata, &context_);

  switch (mode_) {
    case PadMode::CONSTANT:
      PadImageGradientConstNCHW<float><<<
          CAFFE_GET_BLOCKS(output_size),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          output_size,
          dYdata,
          num,
          channels,
          height,
          width,
          padded_height,
          padded_width,
          pad_t(),
          pad_l(),
          dXdata);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case PadMode::REFLECT:
      PadImageGradientReflectNCHW<float><<<
          CAFFE_GET_BLOCKS(input_size),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          input_size,
          dYdata,
          num,
          channels,
          height,
          width,
          padded_height,
          padded_width,
          pad_t(),
          pad_l(),
          dXdata);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case PadMode::EDGE:
      PadImageGradientEdgeNCHW<float><<<
          CAFFE_GET_BLOCKS(input_size),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          input_size,
          dYdata,
          num,
          channels,
          height,
          width,
          padded_height,
          padded_width,
          pad_t(),
          pad_l(),
          dXdata);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
  }

  return true;
}

template<>
bool PadImageGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNHWC() {
  auto& dY = Input(0);

  auto* dX = Output(0, { dY.dim32(0),
      dY.dim32(1) - pad_t() - pad_b(),
      dY.dim32(2) - pad_l() - pad_r(),
      dY.dim32(3)}, at::dtype<float>());
  const int input_size = dY.numel();
  const int padded_height = dY.dim32(1);
  const int padded_width = dY.dim32(2);
  const int output_size = dX->numel();
  const int num = dX->dim32(0);
  const int height = dX->dim32(1);
  const int width = dX->dim32(2);
  const int channels = dX->dim32(3);
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->template mutable_data<float>();
  math::Set<float, CUDAContext>(output_size, 0, dXdata, &context_);

  switch (mode_) {
    case PadMode::CONSTANT:
      PadImageGradientConstNHWC<float><<<
          CAFFE_GET_BLOCKS(output_size),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          output_size,
          dYdata,
          num,
          height,
          width,
          channels,
          padded_height,
          padded_width,
          pad_t(),
          pad_l(),
          dXdata);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case PadMode::REFLECT:
      PadImageGradientReflectNHWC<float><<<
          CAFFE_GET_BLOCKS(input_size),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          input_size,
          dYdata,
          num,
          height,
          width,
          channels,
          padded_height,
          padded_width,
          pad_t(),
          pad_l(),
          dXdata);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case PadMode::EDGE:
      PadImageGradientEdgeNHWC<float><<<
          CAFFE_GET_BLOCKS(input_size),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          input_size,
          dYdata,
          num,
          height,
          width,
          channels,
          padded_height,
          padded_width,
          pad_t(),
          pad_l(),
          dXdata);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
  }

  return true;
}


REGISTER_CUDA_OPERATOR(PadImage, PadImageOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(PadImageGradient,
                       PadImageGradientOp<float, CUDAContext>);
}  // namespace caffe2
