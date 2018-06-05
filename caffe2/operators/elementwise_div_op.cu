#include "caffe2/operators/elementwise_div_op.h"

#include <algorithm>
#include <functional>

#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T>
using BlockReduce = cub::BlockReduce<T, CAFFE_CUDA_NUM_THREADS>;

template <typename TGrad, typename TIn, int D>
__global__ void ComputeDivAGradientCUDAKernel(
    const int outer_size,
    const int inner_size,
    const SimpleArray<int, D> C_dims,
    const SimpleArray<int, D> C_strides,
    const SimpleArray<int, D> B_strides,
    const SimpleArray<int, D> A_dims,
    const TGrad* dC,
    const TIn* B,
    TGrad* dA) {
  __shared__ typename BlockReduce<TGrad>::TempStorage temp_storage;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    TGrad sum = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int A_index = i * inner_size + j;
      int C_index = 0;
      int A_index_val = A_index;
#pragma unroll
      for (int d = D - 1; d >= 0; --d) {
        C_index += (A_index_val % A_dims.data[d]) * C_strides.data[d];
        A_index_val /= A_dims.data[d];
      }
      int B_index = 0;
      int C_index_val = C_index;
#pragma unroll
      for (int d = D - 1; d >= 0; --d) {
        B_index += B_strides.data[d] == 0
            ? 0
            : (C_index_val % C_dims.data[d]) * B_strides.data[d];
        C_index_val /= C_dims.data[d];
      }
#if __CUDA_ARCH__ >= 350
      sum += __ldg(dC + C_index) / __ldg(B + B_index);
#else
      sum += dC[C_index] / B[B_index];
#endif
    }
    sum = BlockReduce<TGrad>(temp_storage).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
      dA[i] = sum;
    }
    __syncthreads();
  }
}

template <typename TGrad, typename TIn, typename TOut>
__global__ void ComputeSimpleDivBGradientCUDAKernel(
    const int size,
    const TGrad* dC,
    const TIn* B,
    const TOut* C,
    TGrad* dB) {
  CUDA_1D_KERNEL_LOOP(i, size) {
#if __CUDA_ARCH__ >= 350
    dB[i] = -__ldg(dC + i) * __ldg(C + i) / __ldg(B + i);
#else
    dB[i] = -dC[i] * C[i] / B[i];
#endif
  }
}

template <typename TGrad, typename TIn, typename TOut, int D>
__global__ void ComputeDivBGradientCUDAKernel(
    const int outer_size,
    const int inner_size,
    const SimpleArray<int, D> C_strides,
    const SimpleArray<int, D> B_dims,
    const TGrad* dC,
    const TIn* B,
    const TOut* C,
    TGrad* dB) {
  __shared__ typename BlockReduce<TGrad>::TempStorage temp_storage;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    TGrad sum = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      int C_index = 0;
      int B_index = i * inner_size + j;
#pragma unroll
      for (int d = D - 1; d >= 0; --d) {
        C_index += (B_index % B_dims.data[d]) * C_strides.data[d];
        B_index /= B_dims.data[d];
      }
#if __CUDA_ARCH__ >= 350
      sum += -__ldg(dC + C_index) * __ldg(C + C_index) / __ldg(B + i);
#else
      sum += -dC[C_index] * C[C_index] / B[i];
#endif
    }
    sum = BlockReduce<TGrad>(temp_storage).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
      dB[i] = sum;
    }
    __syncthreads();
  }
}

template <typename TGrad, typename TIn, int D>
void ComputeDivAGradientCUDAImpl(
    const int outer_size,
    const int inner_size,
    const int* C_dims,
    const int* B_dims,
    const int* A_axes,
    const TGrad* dC,
    const TIn* B,
    TGrad* dA,
    CUDAContext* context) {
  SimpleArray<int, D> C_dims_arr;
  SimpleArray<int, D> C_strides_arr;
  SimpleArray<int, D> B_strides_arr;
  SimpleArray<int, D> A_dims_arr;
  std::copy_n(C_dims, D, C_dims_arr.data);
  math::utils::ComputeTransposedStrides(D, C_dims, A_axes, C_strides_arr.data);
  int cur_stride = 1;
  for (int i = D - 1; i >= 0; --i) {
    B_strides_arr.data[i] = B_dims[i] == 1 ? 0 : cur_stride;
    cur_stride *= B_dims[i];
  }
  for (int i = 0; i < D; ++i) {
    A_dims_arr.data[i] = C_dims[A_axes[i]];
  }
  ComputeDivAGradientCUDAKernel<TGrad, TIn, D>
      <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          outer_size,
          inner_size,
          C_dims_arr,
          C_strides_arr,
          B_strides_arr,
          A_dims_arr,
          dC,
          B,
          dA);
}

template <typename TGrad, typename TIn, typename TOut, int D>
void ComputeDivBGradientCUDAImpl(
    const int outer_size,
    const int inner_size,
    const int* C_dims,
    const int* B_axes,
    const TGrad* dC,
    const TIn* B,
    const TOut* C,
    TGrad* dB,
    CUDAContext* context) {
  SimpleArray<int, D> C_strides_arr;
  SimpleArray<int, D> B_dims_arr;
  math::utils::ComputeTransposedStrides(D, C_dims, B_axes, C_strides_arr.data);
  for (int i = 0; i < D; ++i) {
    B_dims_arr.data[i] = C_dims[B_axes[i]];
  }
  ComputeDivBGradientCUDAKernel<TGrad, TIn, TOut, D>
      <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          outer_size, inner_size, C_strides_arr, B_dims_arr, dC, B, C, dB);
}

template <typename TGrad, typename TIn>
void ComputeDivAGradientCUDA(
    const std::vector<int>& C_dims,
    const std::vector<int>& B_dims,
    const std::vector<int>& A_axes,
    const TGrad* dC,
    const TIn* B,
    TGrad* dA,
    CUDAContext* context) {
  CAFFE_ENFORCE_EQ(C_dims.size(), B_dims.size());
  const int ndim = C_dims.size();
  std::vector<int> A_transpose_axes(ndim);
  math::utils::ComputeTransposeAxesForReduceOp(
      ndim, A_axes.size(), A_axes.data(), A_transpose_axes.data());
  const int pivot = ndim - A_axes.size();
  int outer_size = 1;
  for (int i = 0; i < pivot; ++i) {
    outer_size *= C_dims[A_transpose_axes[i]];
  }
  int inner_size = 1;
  for (int i = pivot; i < ndim; ++i) {
    inner_size *= C_dims[A_transpose_axes[i]];
  }
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_2(
      ndim,
      ComputeDivAGradientCUDAImpl,
      TGrad,
      TIn,
      outer_size,
      inner_size,
      C_dims.data(),
      B_dims.data(),
      A_transpose_axes.data(),
      dC,
      B,
      dA,
      context);
}

template <typename TGrad, typename TIn, typename TOut>
void ComputeDivBGradientCUDA(
    const std::vector<int>& C_dims,
    const std::vector<int>& B_axes,
    const TGrad* dC,
    const TIn* B,
    const TOut* C,
    TGrad* dB,
    CUDAContext* context) {
  const int ndim = C_dims.size();
  std::vector<int> B_transpose_axes(ndim);
  math::utils::ComputeTransposeAxesForReduceOp(
      ndim, B_axes.size(), B_axes.data(), B_transpose_axes.data());
  const int pivot = ndim - B_axes.size();
  int outer_size = 1;
  for (int i = 0; i < pivot; ++i) {
    outer_size *= C_dims[B_transpose_axes[i]];
  }
  int inner_size = 1;
  for (int i = pivot; i < ndim; ++i) {
    inner_size *= C_dims[B_transpose_axes[i]];
  }
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_3(
      ndim,
      ComputeDivBGradientCUDAImpl,
      TGrad,
      TIn,
      TOut,
      outer_size,
      inner_size,
      C_dims.data(),
      B_transpose_axes.data(),
      dC,
      B,
      C,
      dB,
      context);
}

} // namespace

template <>
template <typename TGrad, typename TIn, typename TOut>
bool DivFunctor<CUDAContext>::Backward(
    const std::vector<int>& A_dims,
    const std::vector<int>& B_dims,
    const TGrad* dC,
    const TIn* /* A */,
    const TIn* B,
    const TOut* C,
    TGrad* dA,
    TGrad* dB,
    CUDAContext* context) const {
  if (A_dims == B_dims) {
    const int size = std::accumulate(
        A_dims.cbegin(), A_dims.cend(), 1, std::multiplies<int>());
    math::Div(size, dC, B, dA, context);
    ComputeSimpleDivBGradientCUDAKernel<TGrad, TIn, TOut>
        <<<CAFFE_GET_BLOCKS(size),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context->cuda_stream()>>>(size, dC, B, C, dB);
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
  ComputeBinaryBroadcastBackwardAxes(A_dims, B_dims, &A_axes, &B_axes);
  ComputeDivAGradientCUDA<TGrad, TIn>(
      C_broadcast_dims, B_broadcast_dims, A_axes, dC, B, dA, context);
  ComputeDivBGradientCUDA<TGrad, TIn, TOut>(
      C_broadcast_dims, B_axes, dC, B, C, dB, context);
  return true;
}

REGISTER_CUDA_OPERATOR(
    Div,
    BinaryElementwiseOp<NumericTypes, CUDAContext, DivFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    DivGradient,
    BinaryElementwiseGradientOp<
        NumericTypes,
        CUDAContext,
        DivFunctor<CUDAContext>>);

} // namespace caffe2
