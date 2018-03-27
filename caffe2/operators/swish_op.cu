#include <cmath>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/elementwise_op.h"
#include "caffe2/operators/swish_op.h"

namespace caffe2 {

template <typename T>
__global__ void SwishKernel(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = x[i] / (1. + exp(-x[i]));
  }
}

template <typename T>
__global__ void
SwishGradientKernel(const int N, const T* x, const T* y, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = dy[i] * (y[i] + (1. - y[i]) / (1. + exp(-x[i])));
  }
}

struct SwishCUDAFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CUDAContext* device_context) {
    SwishKernel<T>
        <<<CAFFE_GET_BLOCKS(n),
           CAFFE_CUDA_NUM_THREADS,
           0,
           device_context->cuda_stream()>>>(n, x, y);
    return;
  }
};

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
  SwishGradientKernel<T>
      <<<CAFFE_GET_BLOCKS(n),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(n, x, y, dy, dx);
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
        SwishCUDAFunctor>);
REGISTER_CUDA_OPERATOR(SwishGradient, SwishGradientOp<CUDAContext>);
} // namespace caffe2
