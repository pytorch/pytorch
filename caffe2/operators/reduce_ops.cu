#include "caffe2/operators/reduce_ops.h"

#include <algorithm>
#include <functional>
#include <vector>

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/fixed_divisor.h"

namespace caffe2 {

namespace {

template <typename T, int D>
__global__ void ComputeReduceMinMaxGradientCUDAKernel(
    const int X_size,
    const SimpleArray<int, D> Y_strides,
    const SimpleArray<FixedDivisor<int>, D> X_dims,
    const T* dY_data,
    const T* X_data,
    const T* Y_data,
    T* dX_data) {
  CUDA_1D_KERNEL_LOOP(X_index, X_size) {
    int Y_index = 0;
    int X_index_val = X_index;
#pragma unroll
    for (int i = D - 1; i >= 0; --i) {
      int d;
      X_dims.data[i].DivMod(X_index_val, &X_index_val, &d);
      Y_index += d * Y_strides.data[i];
    }
#if __CUDA_ARCH__ >= 350
    dX_data[X_index] = __ldg(Y_data + Y_index) == __ldg(X_data + X_index)
        ? __ldg(dY_data + Y_index)
        : T(0);
#else
    dX_data[X_index] =
        Y_data[Y_index] == X_data[X_index] ? dY_data[Y_index] : T(0);
#endif
  }
}

template <typename T, int D>
void ComputeReduceMinMaxGradientCUDAImpl(
    const int* Y_dims,
    const int* X_dims,
    const T* dY_data,
    const T* X_data,
    const T* Y_data,
    T* dX_data,
    CUDAContext* context) {
  SimpleArray<int, D> Y_strides_array;
  SimpleArray<FixedDivisor<int>, D> X_dims_array;
  int cur_stride = 1;
  for (int i = D - 1; i >= 0; --i) {
    if (X_dims[i] == 0) {
      return;
    }
    Y_strides_array.data[i] = Y_dims[i] == 1 ? 0 : cur_stride;
    X_dims_array.data[i] = FixedDivisor<int>(X_dims[i]);
    cur_stride *= Y_dims[i];
  }
  const int X_size =
      std::accumulate(X_dims, X_dims + D, 1, std::multiplies<int>());
  ComputeReduceMinMaxGradientCUDAKernel<T, D>
      <<<CAFFE_GET_BLOCKS(X_size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          X_size,
          Y_strides_array,
          X_dims_array,
          dY_data,
          X_data,
          Y_data,
          dX_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

template <>
template <typename T>
bool MinReducer<CUDAContext>::Backward(
    const std::vector<int>& dY_dims,
    const std::vector<int>& dX_dims,
    const T* dY_data,
    const T* X_data,
    const T* Y_data,
    T* dX_data,
    CUDAContext* context) const {
  const int ndim = dY_dims.size();
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(
      ndim,
      ComputeReduceMinMaxGradientCUDAImpl,
      T,
      dY_dims.data(),
      dX_dims.data(),
      dY_data,
      X_data,
      Y_data,
      dX_data,
      context);
  return true;
}

template <>
template <typename T>
bool MaxReducer<CUDAContext>::Backward(
    const std::vector<int>& dY_dims,
    const std::vector<int>& dX_dims,
    const T* dY_data,
    const T* X_data,
    const T* Y_data,
    T* dX_data,
    CUDAContext* context) const {
  const int ndim = dY_dims.size();
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(
      ndim,
      ComputeReduceMinMaxGradientCUDAImpl,
      T,
      dY_dims.data(),
      dX_dims.data(),
      dY_data,
      X_data,
      Y_data,
      dX_data,
      context);
  return true;
}

REGISTER_CUDA_OPERATOR(
    ReduceMin,
    ReduceOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CUDAContext,
        MinReducer<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    ReduceMinGradient,
    ReduceGradientOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CUDAContext,
        MinReducer<CUDAContext>>);

REGISTER_CUDA_OPERATOR(
    ReduceMax,
    ReduceOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CUDAContext,
        MaxReducer<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    ReduceMaxGradient,
    ReduceGradientOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CUDAContext,
        MaxReducer<CUDAContext>>);

REGISTER_CUDA_OPERATOR(
    ReduceSum,
    ReduceOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CUDAContext,
        SumReducer<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    ReduceSumGradient,
    ReduceGradientOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CUDAContext,
        SumReducer<CUDAContext>>);

REGISTER_CUDA_OPERATOR(
    ReduceMean,
    ReduceOp<TensorTypes<float>, CUDAContext, MeanReducer<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    ReduceMeanGradient,
    ReduceGradientOp<
        TensorTypes<float>,
        CUDAContext,
        MeanReducer<CUDAContext>>);

} // namespace caffe2
