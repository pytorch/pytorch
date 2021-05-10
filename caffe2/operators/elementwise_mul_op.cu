#include "caffe2/operators/elementwise_mul_op.h"

#include <algorithm>
#include <functional>

#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/elementwise_ops_utils.h"
#include "caffe2/utils/fixed_divisor.h"

namespace caffe2 {

namespace {

template <typename T>
using BlockReduce = cub::BlockReduce<T, CAFFE_CUDA_NUM_THREADS>;

template <typename TGrad, typename TIn, int D>
__global__ void ComputeMulGradientCUDAKernel(
    const int outer_size,
    const int inner_size,
    const SimpleArray<FixedDivisor<int>, D> Y_dims,
    const SimpleArray<int, D> Y_strides,
    const SimpleArray<int, D> W_strides,
    const SimpleArray<FixedDivisor<int>, D> X_dims,
    const TGrad* dY,
    const TIn* W,
    TGrad* dX) {
  __shared__ typename BlockReduce<TGrad>::TempStorage temp_storage;
  int valid = min(inner_size, CAFFE_CUDA_NUM_THREADS);
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    TGrad sum = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int X_index = i * inner_size + j;
      int Y_index = 0;
      int X_index_val = X_index;
#pragma unroll
      for (int d = D - 1; d >= 0; --d) {
        int r;
        X_dims.data[d].DivMod(X_index_val, &X_index_val, &r);
        Y_index += r * Y_strides.data[d];
      }
      int W_index = 0;
      int Y_index_val = Y_index;
#pragma unroll
      for (int d = D - 1; d >= 0; --d) {
        int r;
        Y_dims.data[d].DivMod(Y_index_val, &Y_index_val, &r);
        W_index += r * W_strides.data[d];
      }
#if __CUDA_ARCH__ >= 350
      sum += __ldg(dY + Y_index) * __ldg(W + W_index);
#else
      sum += dY[Y_index] * W[W_index];
#endif
    }
    sum = BlockReduce<TGrad>(temp_storage).Sum(sum, valid);
    if (threadIdx.x == 0) {
      dX[i] = sum;
    }
    __syncthreads();
  }
}
template <typename TGrad, typename TIn, int D>
__global__ void ComputeMulGradientOuterCUDAKernel(
    const int outer_size,
    const SimpleArray<FixedDivisor<int>, D> Y_dims,
    const SimpleArray<int, D> Y_strides,
    const SimpleArray<int, D> W_strides,
    const SimpleArray<FixedDivisor<int>, D> X_dims,
    const TGrad* dY,
    const TIn* W,
    TGrad* dX) {
  CUDA_1D_KERNEL_LOOP(i, outer_size) {
    TGrad sum = 0;
    const int X_index = i;
    int Y_index = 0;
    int X_index_val = X_index;
#pragma unroll
    for (int d = D - 1; d >= 0; --d) {
      int r;
      X_dims.data[d].DivMod(X_index_val, &X_index_val, &r);
      Y_index += r * Y_strides.data[d];
    }
    int W_index = 0;
    int Y_index_val = Y_index;
#pragma unroll
    for (int d = D - 1; d >= 0; --d) {
      int r;
      Y_dims.data[d].DivMod(Y_index_val, &Y_index_val, &r);
      W_index += r * W_strides.data[d];
    }
#if __CUDA_ARCH__ >= 350
    sum += __ldg(dY + Y_index) * __ldg(W + W_index);
#else
    sum += dY[Y_index] * W[W_index];
#endif
    dX[i] = sum;
  }
}
template <typename TGrad, typename TIn, int D>
void ComputeMulGradientCUDAImpl(
    const int outer_size,
    const int inner_size,
    const int* Y_dims,
    const int* W_dims,
    const int* X_axes,
    const TGrad* dY,
    const TIn* W,
    TGrad* dX,
    CUDAContext* context) {
  SimpleArray<FixedDivisor<int>, D> Y_dims_arr;
  SimpleArray<int, D> Y_strides_arr;
  SimpleArray<int, D> W_strides_arr;
  SimpleArray<FixedDivisor<int>, D> X_dims_arr;
  for (int i = 0; i < D; ++i) {
    Y_dims_arr.data[i] = FixedDivisor<int>(Y_dims[i]);
    X_dims_arr.data[i] = FixedDivisor<int>(Y_dims[X_axes[i]]);
  }
  math::utils::ComputeTransposedStrides(D, Y_dims, X_axes, Y_strides_arr.data);
  int cur_stride = 1;
  for (int i = D - 1; i >= 0; --i) {
    W_strides_arr.data[i] = W_dims[i] == 1 ? 0 : cur_stride;
    cur_stride *= W_dims[i];
  }
  if (inner_size == 1) {
    ComputeMulGradientOuterCUDAKernel<TGrad, TIn, D>
        <<<CAFFE_MAXIMUM_NUM_BLOCKS,
           CAFFE_CUDA_NUM_THREADS,
           0,
           context->cuda_stream()>>>(
            outer_size,
            Y_dims_arr,
            Y_strides_arr,
            W_strides_arr,
            X_dims_arr,
            dY,
            W,
            dX);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    int threads = std::min(inner_size, CAFFE_CUDA_NUM_THREADS);
    ComputeMulGradientCUDAKernel<TGrad, TIn, D>
        <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
           threads,
           0,
           context->cuda_stream()>>>(
            outer_size,
            inner_size,
            Y_dims_arr,
            Y_strides_arr,
            W_strides_arr,
            X_dims_arr,
            dY,
            W,
            dX);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <typename TGrad, typename TIn>
void ComputeMulGradientCUDA(
    const std::vector<int>& Y_dims,
    const std::vector<int>& W_dims,
    const std::vector<int>& X_axes,
    const TGrad* dY,
    const TIn* W,
    TGrad* dX,
    CUDAContext* context) {
  CAFFE_ENFORCE_EQ(Y_dims.size(), W_dims.size());
  const int ndim = Y_dims.size();
  std::vector<int> X_transpose_axes(ndim);
  math::utils::ComputeTransposeAxesForReduceOp(
      ndim, X_axes.size(), X_axes.data(), X_transpose_axes.data());
  const int pivot = ndim - X_axes.size();
  int outer_size = 1;
  for (int i = 0; i < pivot; ++i) {
    outer_size *= Y_dims[X_transpose_axes[i]];
  }
  int inner_size = 1;
  for (int i = pivot; i < ndim; ++i) {
    inner_size *= Y_dims[X_transpose_axes[i]];
  }
  if (outer_size > 0 && inner_size > 0) {
    DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_2(
        ndim,
        ComputeMulGradientCUDAImpl,
        TGrad,
        TIn,
        outer_size,
        inner_size,
        Y_dims.data(),
        W_dims.data(),
        X_transpose_axes.data(),
        dY,
        W,
        dX,
        context);
  } else if (outer_size > 0) {
    math::Set<TGrad, CUDAContext>(outer_size, TGrad(0), dX, context);
  }
}

} // namespace

template <>
template <typename TGrad, typename TIn, typename TOut>
bool MulFunctor<CUDAContext>::Backward(
    const std::vector<int>& A_dims,
    const std::vector<int>& B_dims,
    const TGrad* dC,
    const TIn* A,
    const TIn* B,
    const TOut* /* C */,
    TGrad* dA,
    TGrad* dB,
    CUDAContext* context) const {
  if (A_dims == B_dims) {
    const int size = std::accumulate(
        A_dims.cbegin(), A_dims.cend(), 1, std::multiplies<int>());
    math::Mul(size, dC, B, dA, context);
    math::Mul(size, dC, A, dB, context);
    return true;
  }
  const int ndim = std::max(A_dims.size(), B_dims.size());
  std::vector<int> A_broadcast_dims(ndim);
  std::vector<int> B_broadcast_dims(ndim);
  std::vector<int> C_broadcast_dims(ndim);
  math::utils::ComputeBroadcastBinaryOpDims(
      A_dims.size(),
      A_dims.data(),
      B_dims.size(),
      B_dims.data(),
      A_broadcast_dims.data(),
      B_broadcast_dims.data(),
      C_broadcast_dims.data());
  std::vector<int> A_axes;
  std::vector<int> B_axes;
  elementwise_ops_utils::ComputeBinaryBroadcastBackwardAxes(
      A_dims, B_dims, &A_axes, &B_axes);
  ComputeMulGradientCUDA<TGrad, TIn>(
      C_broadcast_dims, B_broadcast_dims, A_axes, dC, B, dA, context);
  ComputeMulGradientCUDA<TGrad, TIn>(
      C_broadcast_dims, A_broadcast_dims, B_axes, dC, A, dB, context);
  return true;
}

REGISTER_CUDA_OPERATOR(
    Mul,
    BinaryElementwiseOp<NumericTypes, CUDAContext, MulFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    MulGradient,
    BinaryElementwiseGradientOp<
        NumericTypes,
        CUDAContext,
        MulFunctor<CUDAContext>>);

} // namespace caffe2
