#include "caffe2/operators/mish_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void MishCUDAKernel(const int N, const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    Y[i] = __ldg(X + i) * tanh(log(T(1) + exp(__ldg(X + i))));
#else
    Y[i] = X[i] * tanh(log(T(1) + exp(X[i])));
#endif
  }
}

template <typename T>
__global__ void MishGradientCUDAKernel(
    const int N,
    const T* X,
    const T* Y,
    const T* dY,
    T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = __ldg(dY + i) * exp(__ldg(X + i)) *
        (exp(T(3) * __ldg(X + i)) + T(4) * exp(T(2) * __ldg(X + i)) +
         (T(6) + T(4) * __ldg(X + i)) * exp(__ldg(X + i)) +
         T(4) * (T(1) + __ldg(X + i))) /
        (((exp(__ldg(X + i)) + T(1)) * (exp(__ldg(X + i)) + T(1)) + T(1)) *
         ((exp(__ldg(X + i)) + T(1)) * (exp(__ldg(X + i)) + T(1)) + T(1)));
#else
    dX[i] = dY[i] * exp(X[i]) *
        (exp(T(3) * X[i]) + T(4) * exp(T(2) * X[i]) +
         (T(6) + T(4) * X[i]) * exp(X[i]) + T(4) * (T(1) + X[i])) /
        (((exp(X[i]) + T(1)) * (exp(X[i]) + T(1)) + T(1)) *
         ((exp(X[i]) + T(1)) * (exp(X[i]) + T(1)) + T(1)));
#endif
  }
}

} // namespace

template <>
template <typename T>
bool MishFunctor<CUDAContext>::
operator()(const int N, const T* X, T* Y, CUDAContext* context) const {
  MishCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(N),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(N, X, Y);
  return true;
}

template <>
template <typename T>
bool MishGradientOp<CUDAContext>::DoRunWithType() {
  auto& Xin = Input(X);
  auto& Yin = Input(Y);
  auto& DYin = Input(DY);
  auto* DXout = Output(DX);
  CAFFE_ENFORCE_EQ(Xin.size(), Yin.size());
  CAFFE_ENFORCE_EQ(DYin.size(), Yin.size());
  DXout->ResizeLike(Yin);

  const int n = Xin.size();
  const T* x = Xin.template data<T>();
  const T* y = Yin.template data<T>();
  const T* dy = DYin.template data<T>();
  T* dx = DXout->template mutable_data<T>();
  MishGradientCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(n),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(n, x, y, dy, dx);
  return true;
}

template <>
bool MishGradientOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float, double>>::call(this, Input(X));
}

REGISTER_CUDA_OPERATOR(
    Mish,
    UnaryElementwiseOp<
        TensorTypes<float, double>,
        CUDAContext,
        MishFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(MishGradient, MishGradientOp<CUDAContext>);

} // namespace caffe2
