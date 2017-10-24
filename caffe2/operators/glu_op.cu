#include "caffe2/operators/glu_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {
__global__ void
glu_kernel(const int M, const int N, const float* Xdata, float* Ydata) {
  const int xOffset = 2 * N;
  CUDA_1D_KERNEL_LOOP(index, N * M) {
    const int i = index % M;
    const int j = index / M;
    const float x1 = Xdata[i * xOffset + j];
    const float x2 = Xdata[i * xOffset + j + N];
    Ydata[i * N + j] = x1 * (1. / (1. + exp(-x2)));
  }
}
} // namespace

template <>
void GluOp<float, CUDAContext>::ComputeGlu(
    const int M,
    const int N,
    const float* Xdata,
    float* Ydata) {
  glu_kernel<<<
      CAFFE_GET_BLOCKS(M * N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(M, N, Xdata, Ydata);
}

REGISTER_CUDA_OPERATOR(Glu, GluOp<float, CUDAContext>);
} // namespace caffe2
