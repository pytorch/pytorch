#include <cmath>
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/filler_op.h"

namespace caffe2 {

namespace {
__global__ void FillRangeKernel(const int n, float* data) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    data[index] = index;
  }
}

template <typename T>
__global__ void FillDiagonalKernel(
    const int num_diagonal_elements,
    const TIndex step_size,
    const T value,
    T* data) {
  CUDA_1D_KERNEL_LOOP(index, num_diagonal_elements) {
    data[index * step_size] = value;
  }
}
}

template <>
bool RangeFillOp<float, CUDAContext>::Fill(TensorCUDA* output) {
  int N = output->size();
  FillRangeKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, output->mutable_data<float>());
  return true;
}

template <>
template <typename T>
bool DiagonalFillOp<CUDAContext>::FillWithType(TensorCUDA* output) {
  VerifyOutputShape(output);
  auto* data = output->template mutable_data<T>();
  int size = output->size();
  // first fill everything with 0
  math::Set<T, CUDAContext>(size, T(0), data, &context_);

  T value = OperatorBase::GetSingleArgument<T>("value", 0);
  TIndex step_size = GetStepSize(output);
  int num_diagonal_elements = ceil((float)size / step_size);

  FillDiagonalKernel<<<
      CAFFE_GET_BLOCKS(num_diagonal_elements),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(num_diagonal_elements, step_size, value, data);
  return true;
}

REGISTER_CUDA_OPERATOR(UniformFill, UniformFillOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(UniformIntFill, UniformFillOp<int, CUDAContext>);
REGISTER_CUDA_OPERATOR(ConstantFill, ConstantFillOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(DiagonalFill, DiagonalFillOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(GaussianFill, GaussianFillOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(XavierFill, XavierFillOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(MSRAFill, MSRAFillOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(RangeFill, RangeFillOp<float, CUDAContext>);

} // namespace caffe2
