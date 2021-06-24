#include "caffe2/operators/half_float_ops.h"

#include "caffe2/core/context_gpu.h"

#ifdef CAFFE_HAS_CUDA_FP16

namespace caffe2 {
namespace {
__global__ void FloatToHalfKernel(const int N, const float* X, half* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = __float2half(X[i]);
  }
}

__global__ void HalfToFloatKernel(const int N, const half* X, float* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = __half2float(X[i]);
  }
}
}

template <>
bool FloatToHalfOp<CUDAContext>::RunOnDevice() {
  auto& X = Input(0);

  auto* Y = Output(0, X.sizes(), at::dtype<at::Half>());
  FloatToHalfKernel<<<
      CAFFE_GET_BLOCKS(X.numel()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.numel(),
      X.data<float>(),
      reinterpret_cast<half*>(Y->template mutable_data<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
bool HalfToFloatOp<CUDAContext>::RunOnDevice() {
  auto& X = Input(0);

  auto* Y = Output(0, X.sizes(), at::dtype<float>());
  HalfToFloatKernel<<<
      CAFFE_GET_BLOCKS(X.numel()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.numel(),
      reinterpret_cast<const half*>(X.data<at::Half>()),
      Y->template mutable_data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
bool Float16UniformFillOp<CUDAContext>::RunOnDevice() {
  auto* output = Output(0, shape_, at::dtype<at::Half>());
  at::Half* out = output->template mutable_data<at::Half>();

  auto leading_dim_sz = output->size(0);
  CAFFE_ENFORCE_GT(leading_dim_sz, 0,
      "The input shape should have the first dimension greater than 0");
  int rowsz = output->numel() / output->size(0);

  ReinitializeTensor(
    &temp_data_buffer_, {rowsz}, at::dtype<float>().device(CUDA));
  float* temp_data = temp_data_buffer_.template mutable_data<float>();

  for (uint64_t i = 0; i < leading_dim_sz; i++) {
    math::RandUniform<float, CUDAContext>(
        rowsz, min_, max_, temp_data, &context_);

    FloatToHalfKernel<<<
      CAFFE_GET_BLOCKS(rowsz),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      rowsz,
      temp_data,
      reinterpret_cast<half*>(out + i * rowsz));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return true;
}

REGISTER_CUDA_OPERATOR(FloatToHalf, FloatToHalfOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(HalfToFloat, HalfToFloatOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Float16UniformFill, Float16UniformFillOp<CUDAContext>);
} // namespace caffe2

#endif // CAFFE_HAS_CUDA_FP16
