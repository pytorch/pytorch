#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/resize_3d_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

__global__ void NearestNeighbor3DKernel(
    const int size,
    const int num_channels,
    const int input_frames,
    const int input_height,
    const int input_width,
    const int output_frames,
    const int output_height,
    const int output_width,
    const float temporal_scale,
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
    const int f = indexTemp % output_frames;
    indexTemp /= output_frames;
    const int c = indexTemp % num_channels;
    indexTemp /= num_channels;
    const int n = indexTemp;

    const int in_f = fminf(f / temporal_scale, input_frames - 1);
    const int in_y = fminf(h / height_scale, input_height - 1);
    const int in_x = fminf(w / width_scale, input_width - 1);
    Y[index] =
        X[(((n * num_channels + c) * input_frames + in_f) * input_height + in_y)
          * input_width + in_x];
  }
}

__global__ void NearestNeighbor3DGradientKernel(
    const int size,
    const int num_channels,
    const int input_frames,
    const int input_height,
    const int input_width,
    const int output_frames,
    const int output_height,
    const int output_width,
    const float temporal_scale,
    const float height_scale,
    const float width_scale,
    const float* dY,
    float* dX) {
  CUDA_1D_KERNEL_LOOP(index, size) {
    int indexTemp = index;
    const int x = indexTemp % input_width;
    indexTemp /= input_width;
    const int y = indexTemp % input_height;
    indexTemp /= input_height;
    const int f = indexTemp % input_frames;
    indexTemp /= input_frames;
    const int c = indexTemp % num_channels;
    indexTemp /= num_channels;
    const int n = indexTemp;

    const int out_f = fminf(f / temporal_scale, output_frames - 1);
    const int out_y = fminf(y / height_scale, output_height - 1);
    const int out_x = fminf(x / width_scale, output_width - 1);
    const int out_index =
        (((n * num_channels + c) * output_frames + out_f) * output_height +
          out_y) * output_width + out_x;
#if __CUDA_ARCH__ >= 350
    atomicAdd(dX + out_index, __ldg(dY + index));
#else
    atomicAdd(dX + out_index, *(dY + index));
#endif
  }
}

} // namespace


template <>
bool ResizeNearest3DOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW() {
  const auto& X = Input(0);

  const auto inputDims = X.sizes();
  CAFFE_ENFORCE_EQ(5, inputDims.size());
  const int batch_size = X.dim32(0), num_channels = X.dim32(1),
            input_frames = X.dim32(2), input_height = X.dim32(3),
            input_width = X.dim32(4);

  CAFFE_ENFORCE_EQ(InputSize(), 1);

  int output_frames = input_frames * temporal_scale_;
  int output_height = input_height * height_scale_;
  int output_width = input_width * width_scale_;
  auto* Y = Output(
      0,
      {batch_size, num_channels, output_frames, output_height, output_width},
      at::dtype<float>());

  const auto size = Y->numel();
  NearestNeighbor3DKernel<<<
      CAFFE_GET_BLOCKS(size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      size,
      num_channels,
      input_frames,
      input_height,
      input_width,
      output_frames,
      output_height,
      output_width,
      temporal_scale_,
      height_scale_,
      width_scale_,
      X.data<float>(),
      Y->template mutable_data<float>());

  return true;
}

template <>
bool ResizeNearest3DOp<float, CUDAContext>::RunOnDevice() {
  switch (order_) {
    case StorageOrder::NHWC:
      CAFFE_THROW("Not implemented for storage order: ", order_);
    case StorageOrder::NCHW:
      return RunOnDeviceWithOrderNCHW();
    default:
      CAFFE_THROW("Unknown Storage order: ", order_);
  }
}


template <>
bool ResizeNearest3DGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW() {
  const auto& dY = Input(0);
  const auto& X = Input(1);

  const auto inputDims = dY.sizes();
  CAFFE_ENFORCE_EQ(5, inputDims.size());
  const int batch_size = dY.dim32(0), num_channels = dY.dim32(1),
            input_frames = dY.dim32(2), input_height = dY.dim32(3),
            input_width = dY.dim32(4);

  // X,dim32(2) can be different from int(input_frames / temporal_scale_)
  // We choose to compute output_frames=int(input_frames / temporal_scale_)

  // const int output_frames = X,dim32(2);
  // const int output_height = X.dim32(3);
  // const int output_width = X.dim32(4);

  const int output_frames = int(input_frames / temporal_scale_);
  const int output_height = int(input_height / height_scale_);
  const int output_width = int(input_width / width_scale_);

  CAFFE_ENFORCE_EQ(InputSize(), 2);

  auto* dX = Output(
      0,
      {batch_size, num_channels, output_frames, output_height, output_width},
      at::dtype<float>());
  math::Set<float, CUDAContext>(
      dX->numel(), 0.0f, dX->template mutable_data<float>(), &context_);

  const auto size = dY.numel();
  NearestNeighbor3DGradientKernel<<<
      CAFFE_GET_BLOCKS(size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      size,
      num_channels,
      input_frames,
      input_height,
      input_width,
      output_frames,
      output_height,
      output_width,
      temporal_scale_,
      height_scale_,
      width_scale_,
      dY.data<float>(),
      dX->template mutable_data<float>());

  return true;
}

template <>
bool ResizeNearest3DGradientOp<float, CUDAContext>::RunOnDevice() {
  switch (order_) {
    case StorageOrder::NHWC:
      CAFFE_THROW("Not implemented for storage order: ", order_);
    case StorageOrder::NCHW:
      return RunOnDeviceWithOrderNCHW();
    default:
      CAFFE_THROW("Unknown Storage order: ", order_);
  }
}

REGISTER_CUDA_OPERATOR(ResizeNearest3D, ResizeNearest3DOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    ResizeNearest3DGradient,
    ResizeNearest3DGradientOp<float, CUDAContext>);

} // namespace caffe2

using ResizeNearest3DOpFloatCUDA =
    caffe2::ResizeNearest3DOp<float, caffe2::CUDAContext>;

C10_EXPORT_CAFFE2_OP_TO_C10_CUDA(ResizeNearest3D, ResizeNearest3DOpFloatCUDA);
