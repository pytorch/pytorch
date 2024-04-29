#include "caffe2/operators/swish_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void SwishCUDAKernel(const int N, const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    Y[i] = __ldg(X + i) / (T(1) + exp(-__ldg(X + i)));
#else
    Y[i] = X[i] / (T(1) + exp(-X[i]));
#endif
  }
}

template <typename T>
__global__ void SwishGradientCUDAKernel(
    const int N,
    const T* X,
    const T* Y,
    const T* dY,
    T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = __ldg(dY + i) *
        (__ldg(Y + i) + (T(1) - __ldg(Y + i)) / (T(1) + exp(-__ldg(X + i))));
#else
    dX[i] = dY[i] * (Y[i] + (T(1) - Y[i]) / (T(1) + exp(-X[i])));
#endif
  }
}

} // namespace

template <>
template <typename T>
bool SwishFunctor<CUDAContext>::
operator()(const int N, const T* X, T* Y, CUDAContext* context) const {
  SwishCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(N),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(N, X, Y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
template <typename T>
bool SwishGradientOp<CUDAContext>::DoRunWithType() {
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
  SwishGradientCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(n),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(n, x, y, dy, dx);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
bool SwishGradientOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float, double>>::call(this, Input(X));
}

REGISTER_CUDA_OPERATOR(
    Swish,
    UnaryElementwiseOp<
        TensorTypes<float, double>,
        CUDAContext,
        SwishFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(SwishGradient, SwishGradientOp<CUDAContext>);

} // namespace caffe2
