#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/reduction_ops.h"

namespace caffe2 {
namespace {

REGISTER_CUDA_OPERATOR(SumElements, SumElementsOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SumSqrElements, SumSqrElementsOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(
    SumElementsGradient,
    SumElementsGradientOp<float, CUDAContext>);

template <typename T>
__global__ void
SumElementsGradientKernel(bool average, const int N, const T* dY, T* dX) {
  const T value = average ? (*dY) / N : *dY;
  CUDA_1D_KERNEL_LOOP(i, N) {
    dX[i] = value;
  }
}
} // namespace

template <>
bool SumElementsGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& dY = Input(1);
  DCHECK_EQ(dY.size(), 1);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  SumElementsGradientKernel<float><<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      average_, X.size(), dY.data<float>(), dX->mutable_data<float>());
  return true;
}

} // namespace caffe2
