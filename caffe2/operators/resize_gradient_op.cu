#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "resize_op.h"

namespace caffe2 {

namespace {
__global__ void NearestNeighborGradientKernelNCHW(
    const int size,
    const int num_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const float height_scale,
    const float width_scale,
    const float* dY,
    float* dX) {
  CUDA_1D_KERNEL_LOOP(index, size) {
    int indexTemp = index;
    const int w = indexTemp % input_width;
    indexTemp /= input_width;
    const int h = indexTemp % input_height;
    indexTemp /= input_height;
    const int c = indexTemp % num_channels;
    indexTemp /= num_channels;
    const int n = indexTemp;

    const int out_h = fminf(h / height_scale, output_height - 1);
    const int out_w = fminf(w / width_scale, output_width - 1);
    const int out_index =
        ((n * num_channels + c) * output_height + out_h) * output_width + out_w;
#if __CUDA_ARCH__ >= 350
    atomicAdd(dX + out_index, __ldg(dY + index));
#else
    atomicAdd(dX + out_index, *(dY + index));
#endif
  }
}

__global__ void NearestNeighborGradientKernelNHWC(
    const int size,
    const int num_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const float height_scale,
    const float width_scale,
    const float* dY,
    float* dX) {
  CUDA_1D_KERNEL_LOOP(index, size) {
    int indexTemp = index;
    const int c = indexTemp % num_channels;
    indexTemp /= num_channels;
    const int w = indexTemp % input_width;
    indexTemp /= input_width;
    const int h = indexTemp % input_height;
    indexTemp /= input_height;
    const int n = indexTemp;

    const int out_h = fminf(h / height_scale, output_height - 1);
    const int out_w = fminf(w / width_scale, output_width - 1);
    const int out_index =
        ((n * output_height + out_h) * output_width + out_w) * num_channels + c;
#if __CUDA_ARCH__ >= 350
    atomicAdd(dX + out_index, __ldg(dY + index));
#else
    atomicAdd(dX + out_index, *(dY + index));
#endif
  }
}
} // namespace

template <>
bool ResizeNearestGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW() {
  const auto& dY = Input(0);
  const auto& X = Input(1);
  auto* dX = Output(0);

  const auto& inputDims = dY.dims();
  CAFFE_ENFORCE_EQ(4, inputDims.size());
  const int batch_size = dY.dim32(0), num_channels = dY.dim32(1),
            input_height = dY.dim32(2), input_width = dY.dim32(3);
  int output_height = X.dim32(2);
  int output_width = X.dim32(3);
  dX->Resize(batch_size, num_channels, output_height, output_width);
  math::Set<float, CUDAContext>(
      dX->size(), 0.0f, dX->mutable_data<float>(), &context_);

  const auto size = dY.size();
  NearestNeighborGradientKernelNCHW<<<
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
      dY.data<float>(),
      dX->mutable_data<float>());

  return true;
}

template <>
bool ResizeNearestGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNHWC() {
  const auto& dY = Input(0);
  const auto& X = Input(1);
  auto* dX = Output(0);

  const auto& inputDims = dY.dims();
  CAFFE_ENFORCE_EQ(4, inputDims.size());
  const int batch_size = dY.dim32(0), input_height = dY.dim32(1),
            input_width = dY.dim32(2), num_channels = dY.dim32(3);
  int output_height = X.dim32(1);
  int output_width = X.dim32(2);
  dX->Resize(batch_size, output_height, output_width, num_channels);
  math::Set<float, CUDAContext>(
      dX->size(), 0.0f, dX->mutable_data<float>(), &context_);
  const auto size = dY.size();
  NearestNeighborGradientKernelNHWC<<<
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
      dY.data<float>(),
      dX->mutable_data<float>());
  return true;
}

template <>
bool ResizeNearestGradientOp<float, CUDAContext>::RunOnDevice() {
  switch (order_) {
    case StorageOrder::NCHW:
      return RunOnDeviceWithOrderNCHW();
    case StorageOrder::NHWC:
      return RunOnDeviceWithOrderNHWC();
    default:
      CAFFE_THROW("Unknown storage order: ", order_);
  }
}

REGISTER_CUDA_OPERATOR(
    ResizeNearestGradient,
    ResizeNearestGradientOp<float, CUDAContext>);
} // namespace caffe2
