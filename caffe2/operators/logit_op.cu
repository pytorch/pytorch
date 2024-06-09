#include "caffe2/operators/logit_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void LogitKernel(const int N, const T* X, const float eps, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = fminf(X[i], (T(1) - eps));
    Y[i] = fmaxf(Y[i], eps);
    Y[i] = logf(Y[i] / (T(1) - Y[i]));
  }
}

template <typename T>
__global__ void LogitGradientKernel(
    const int N,
    const T* X,
    const T* dY,
    const float eps,
    T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dX[i] = (X[i] < eps || X[i] > T(1) - eps) ? T(0)
                                              : (dY[i] / X[i] / (T(1) - X[i]));
  }
}

} // namespace

template <>
template <typename T>
bool LogitFunctor<CUDAContext>::
operator()(const int N, const T* X, T* Y, CUDAContext* context) const {
  LogitKernel<T>
      <<<CAFFE_GET_BLOCKS(N),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(N, X, eps_, Y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
bool LogitGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  int n = X.size();
  LogitGradientKernel<<<
      CAFFE_GET_BLOCKS(n),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      n,
      X.data<float>(),
      dY.data<float>(),
      eps_,
      dX->template mutable_data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(
    Logit,
    UnaryElementwiseWithArgsOp<
        TensorTypes<float>,
        CUDAContext,
        LogitFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(LogitGradient, LogitGradientOp<float, CUDAContext>);

} // namespace caffe2
