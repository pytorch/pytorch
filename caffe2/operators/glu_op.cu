#include "caffe2/operators/glu_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {
__global__ void glu_kernel(
    const int M,
    const int split_dim_size,
    const int N,
    const float* Xdata,
    float* Ydata) {
  const int xOffset = 2 * split_dim_size * N;
  const int yOffset = split_dim_size * N;
  CUDA_1D_KERNEL_LOOP(index, M * split_dim_size * N) {
    const int i = index / split_dim_size / N;
    const int j = index / N % split_dim_size;
    const int k = index % N;
    const float x1 = Xdata[i * xOffset + j * N + k];
    const float x2 = Xdata[i * xOffset + (j + split_dim_size) * N + k];
    Ydata[i * yOffset + j * N + k] = x1 * (1. / (1. + exp(-x2)));
  }
}
} // namespace

template <>
void GluOp<float, CUDAContext>::ComputeGlu(
    const int M,
    const int split_dim_size,
    const int N,
    const float* x_data,
    float* y_data) {
  glu_kernel<<<
      CAFFE_GET_BLOCKS(M * N * split_dim_size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(M, split_dim_size, N, x_data, y_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

REGISTER_CUDA_OPERATOR(Glu, GluOp<float, CUDAContext>);
} // namespace caffe2
