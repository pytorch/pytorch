#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "resize_op.h"

namespace caffe2 {

namespace {

__global__ void NearestNeighborKernelNCHW(
    const int size,
    const int num_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const float height_scale,
    const float width_scale,
    const float* X,
    float* Y) {
  CUDA_1D_KERNEL_LOOP(index, size) {
    int indexTemp = index;
    const int w = indexTemp % output_width;
    indexTemp /= output_width;
    const int h = indexTemp % output_height;
    indexTemp /= output_height;
    const int c = indexTemp % num_channels;
    indexTemp /= num_channels;
    const int n = indexTemp;

    const int in_h = fminf(h / height_scale, input_height - 1);
    const int in_w = fminf(w / width_scale, input_width - 1);
    Y[index] =
        X[((n * num_channels + c) * input_height + in_h) * input_width + in_w];
  }
}

__global__ void NearestNeighborKernelNHWC(
    const int size,
    const int num_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const float height_scale,
    const float width_scale,
    const float* X,
    float* Y) {
  CUDA_1D_KERNEL_LOOP(index, size) {
    int indexTemp = index;
    const int c = indexTemp % num_channels;
    indexTemp /= num_channels;
    const int w = indexTemp % output_width;
    indexTemp /= output_width;
    const int h = indexTemp % output_height;
    indexTemp /= output_height;
    const int n = indexTemp;

    const int in_h = fminf(h / height_scale, input_height - 1);
    const int in_w = fminf(w / width_scale, input_width - 1);
    Y[index] =
        X[((n * input_height + in_h) * input_width + in_w) * num_channels + c];
  }
}
} // namespace

template <>
bool ResizeNearestOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW() {
  const auto& X = Input(0);
  auto* Y = Output(0);

  const auto& inputDims = X.dims();
  CAFFE_ENFORCE_EQ(4, inputDims.size());
  const int batch_size = X.dim32(0), num_channels = X.dim32(1),
            input_height = X.dim32(2), input_width = X.dim32(3);
  int output_width = input_width * width_scale_;
  int output_height = input_height * height_scale_;
  Y->Resize(batch_size, num_channels, output_height, output_width);

  const auto size = Y->size();
  NearestNeighborKernelNCHW<<<
      CAFFE_GET_BLOCKS(size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      size,
      num_channels,
      input_height,
      input_width,
      output_height,
      output_width,
      height_scale_,
      width_scale_,
      X.data<float>(),
      Y->mutable_data<float>());

  return true;
}

template <>
bool ResizeNearestOp<float, CUDAContext>::RunOnDeviceWithOrderNHWC() {
  const auto& X = Input(0);
  auto* Y = Output(0);

  const auto& inputDims = X.dims();
  CAFFE_ENFORCE_EQ(4, inputDims.size());
  const int batch_size = X.dim32(0), input_height = X.dim32(1),
            input_width = X.dim32(2), num_channels = X.dim32(3);
  int output_width = input_width * width_scale_;
  int output_height = input_height * height_scale_;
  Y->Resize(batch_size, output_height, output_width, num_channels);
  const auto size = Y->size();
  NearestNeighborKernelNHWC<<<
      CAFFE_GET_BLOCKS(size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      size,
      num_channels,
      input_height,
      input_width,
      output_height,
      output_width,
      height_scale_,
      width_scale_,
      X.data<float>(),
      Y->mutable_data<float>());

  return true;
}

template <>
bool ResizeNearestOp<float, CUDAContext>::RunOnDevice() {
  switch (order_) {
    case StorageOrder::NCHW:
      return RunOnDeviceWithOrderNCHW();
    case StorageOrder::NHWC:
      return RunOnDeviceWithOrderNHWC();
    default:
      CAFFE_THROW("Unknown storage order: ", order_);
  }
}

REGISTER_CUDA_OPERATOR(ResizeNearest, ResizeNearestOp<float, CUDAContext>);
} // namespace caffe2
