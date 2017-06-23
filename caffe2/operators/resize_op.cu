#include "caffe2/core/context_gpu.h"
#include "resize_op.h"

namespace caffe2 {

namespace {

__global__ void NearestNeighborKernel(
    const int size,
    const int C,
    const int H,
    const int W,
    const float height_scale,
    const float width_scale,
    const float* X,
    float* Y) {
  CUDA_1D_KERNEL_LOOP(index, size) {
    const int output_width = W * width_scale;
    const int output_height = H * height_scale;

    int indexTemp = index;
    const int w = indexTemp % output_width;
    indexTemp /= output_width;
    const int h = indexTemp % output_height;
    indexTemp /= output_height;
    const int c = indexTemp % C;
    indexTemp /= C;
    const int n = indexTemp;

    const int input_h = h / height_scale;
    const int input_w = w / width_scale;
    Y[index] = X[((n * C + c) * H + input_h) * W + input_w];
  }
}

} // namespace

template <>
bool ResizeNearestOp<float, CUDAContext>::RunOnDevice() {
  const auto& X = Input(0);
  auto* Y = Output(0);

  const auto& inputDims = X.dims();
  CAFFE_ENFORCE_EQ(4, inputDims.size());
  const int N = X.dim32(0), C = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
  int output_width = W * width_scale_;
  int output_height = H * height_scale_;
  Y->Resize(N, C, output_height, output_width);

  const auto size = Y->size();
  NearestNeighborKernel<<<
      CAFFE_GET_BLOCKS(size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      size,
      C,
      H,
      W,
      height_scale_,
      width_scale_,
      X.data<float>(),
      Y->mutable_data<float>());

  return true;
}

namespace {
REGISTER_CUDA_OPERATOR(ResizeNearest, ResizeNearestOp<float, CUDAContext>);
} // namespace
} // namespace caffe2
