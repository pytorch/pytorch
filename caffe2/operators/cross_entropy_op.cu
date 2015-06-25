#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/cross_entropy_op.h"

namespace caffe2 {

namespace {
__global__ void LabelCrossEntropyKernel(
    const int N, const int D, const float* Xdata, const int* labeldata,
    const float log_threshold, float* Ydata) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Ydata[i] = -logf(max(Xdata[i * D + labeldata[i]], log_threshold));
  }
}
__global__ void LabelCrossEntropyGradientKernel(
    const int N, const int D, const float* Xdata, const int* labeldata,
    const float* dYdata, const float log_threshold, float* dXdata) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    int idx = i * D + labeldata[i];
    dXdata[idx] = - dYdata[i] / max(Xdata[idx], log_threshold);
  }
}
}  // namespace

template <>
bool LabelCrossEntropyOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& label = OperatorBase::Input<Tensor<int, CUDAContext> >(1);
  auto* Y = Output(0);
  DCHECK_EQ(X.ndim(), 2);
  int N = X.dim(0);
  int D = X.dim(1);
  DCHECK_EQ(label.ndim(), 1);
  DCHECK_EQ(label.dim(0), N);
  Y->Reshape(std::vector<int>(1, N));
  LabelCrossEntropyKernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS,
                            0, device_context_.cuda_stream()>>>(
      N, D, X.data(), label.data(), kLOG_THRESHOLD(), Y->mutable_data());
  return true;
}

template <>
bool LabelCrossEntropyGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& label = OperatorBase::Input<Tensor<int, CUDAContext> >(1);
  auto& dY = Input(2);
  auto* dX = Output(0);
  DCHECK_EQ(X.ndim(), 2);
  int N = X.dim(0);
  int D = X.dim(1);
  DCHECK_EQ(label.ndim(), 1);
  DCHECK_EQ(label.dim(0), N);
  DCHECK_EQ(dY.ndim(), 1);
  DCHECK_EQ(dY.dim(0), N);
  dX->ReshapeLike(X);
  math::Set<float, CUDAContext>(
      dX->size(), 0.f, dX->mutable_data(), &device_context_);
  LabelCrossEntropyGradientKernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS,
                                    0, device_context_.cuda_stream()>>>(
      N, D, X.data(), label.data(), dY.data(), kLOG_THRESHOLD(),
      dX->mutable_data());
  return true;
}

namespace {
REGISTER_CUDA_OPERATOR(LabelCrossEntropy,
                       LabelCrossEntropyOp<float, CUDAContext>)
REGISTER_CUDA_OPERATOR(LabelCrossEntropyGradient,
                       LabelCrossEntropyGradientOp<float, CUDAContext>)
}  // namespace
}  // namespace caffe2
