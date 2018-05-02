#include "caffe2/operators/reduce_ops.h"

#include <algorithm>
#include <functional>
#include <vector>

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T, int D>
__global__ void ComputeReduceMinMaxGradientCUDAKernel(
    const int dX_size,
    const SimpleArray<int, D> dY_strides,
    const SimpleArray<int, D> dX_dims,
    const T* dY_data,
    const T* X_data,
    const T* Y_data,
    T* dX_data) {
  CUDA_1D_KERNEL_LOOP(dX_index, dX_size) {
    int dY_index = 0;
    int dX_index_val = dX_index;
#pragma unroll
    for (int i = D - 1; i >= 0; --i) {
      dY_index += dY_strides.data[i] == 0
          ? 0
          : (dX_index_val % dX_dims.data[i]) * dY_strides.data[i];
      dX_index_val /= dX_dims.data[i];
    }
#if __CUDA_ARCH__ >= 350
    dX_data[dX_index] = __ldg(Y_data + dY_index) == __ldg(X_data + dX_index)
        ? __ldg(dY_data + dY_index)
        : T(0);
#else
    dX_data[dX_index] =
        Y_data[dY_index] == X_data[dX_index] ? dY_data[dY_index] : T(0);
#endif
  }
}

template <typename T, int D>
void ComputeReduceMinMaxGradientCUDAImpl(
    const int* dY_dims,
    const int* dX_dims,
    const T* dY_data,
    const T* X_data,
    const T* Y_data,
    T* dX_data,
    CUDAContext* context) {
  SimpleArray<int, D> dY_strides_array;
  SimpleArray<int, D> dX_dims_array;
  int cur_stride = 1;
  for (int i = D - 1; i >= 0; --i) {
    dY_strides_array.data[i] = dY_dims[i] == 1 ? 0 : cur_stride;
    dX_dims_array.data[i] = dX_dims[i];
    cur_stride *= dY_dims[i];
  }
  const int dX_size =
      std::accumulate(dX_dims, dX_dims + D, 1, std::multiplies<int>());
  ComputeReduceMinMaxGradientCUDAKernel<T, D>
      <<<CAFFE_GET_BLOCKS(dX_size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          dX_size,
          dY_strides_array,
          dX_dims_array,
          dY_data,
          X_data,
          Y_data,
          dX_data);
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
