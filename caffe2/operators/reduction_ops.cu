#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/reduction_ops.h"
#include "caffe2/utils/conversions.h"

#include <cub/cub.cuh>

namespace caffe2 {

REGISTER_CUDA_OPERATOR(SumElements, SumElementsOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SumElementsInt, SumElementsIntOp<int, CUDAContext>);
REGISTER_CUDA_OPERATOR(SumSqrElements, SumSqrElementsOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(RowwiseMax, MaxReductionOp<float, CUDAContext, true>);
REGISTER_CUDA_OPERATOR(ColwiseMax, MaxReductionOp<float, CUDAContext, false>);
REGISTER_CUDA_OPERATOR(
    RowwiseMaxGradient,
    MaxReductionGradientOp<float, CUDAContext, true>)
REGISTER_CUDA_OPERATOR(
    ColwiseMaxGradient,
    MaxReductionGradientOp<float, CUDAContext, false>)

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

__global__ void rowwise_max_gradient_kernel(
    const int batch_size,
    const int M,
    const int N,
    const float* X,
    const float* Y,
    const float* dY,
    float* dX) {
  const int input_size = M * N;
  CUDA_1D_KERNEL_LOOP(i, batch_size * M * N) {
    const int b_i = i / input_size;
    const int b_n = i / input_size / N;
    const int y_index = b_i * M + b_n;
    if (X[i] == Y[y_index]) {
      dX[i] = dY[y_index];
    } else {
      dX[i] = 0.0;
    }
  }
}

template <>
bool SumSqrElementsOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
}


__global__ void colwise_max_gradient_kernel(
    const int batch_size,
    const int M,
    const int N,
    const float* X,
    const float* Y,
    const float* dY,
    float* dX) {
  const int input_size = M * N;
  CUDA_1D_KERNEL_LOOP(i, batch_size * M * N) {
    const int b_i = i / input_size;
    const int b_n = i % input_size % N;
    const int y_index = b_i * N + b_n;
    if (X[i] == Y[y_index]) {
      dX[i] = dY[y_index];
    } else {
      dX[i] = 0.0;
    }
  }
}

template <>
bool SumElementsGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& dY = Input(1);
  DCHECK_EQ(dY.numel(), 1);

  auto* dX = Output(0, X.sizes(), at::dtype<float>());
  SumElementsGradientKernel<float>
      <<<CAFFE_GET_BLOCKS(X.numel()),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          average_,
          X.numel(),
          dY.data<float>(),
          dX->template mutable_data<float>());
  return true;
}

template <typename T, class Context, bool ROWWISE>
bool MaxReductionGradientOp<T, Context, ROWWISE>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dY = Input(2);

  auto* dX = Output(0, X.sizes(), at::dtype<T>());

  CAFFE_ENFORCE_EQ(X.dim(), 3);

  const int batch_size = X.dim32(0);
  const int M = X.dim32(1);
  const int N = X.dim32(2);

  const T* Xdata = X.template data<T>();
  const T* Ydata = Y.template data<T>();
  const T* dYdata = dY.template data<T>();
  T* dXdata = dX->template mutable_data<T>();

  const int input_size = M * N;
  if (ROWWISE) {
    rowwise_max_gradient_kernel<<<
        CAFFE_GET_BLOCKS(batch_size * input_size),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        batch_size, M, N, Xdata, Ydata, dYdata, dXdata);
  } else {
    colwise_max_gradient_kernel<<<
        CAFFE_GET_BLOCKS(batch_size * input_size),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        batch_size, M, N, Xdata, Ydata, dYdata, dXdata);
  }
  return true;
}

} // namespace caffe2
