#include "caffe2/operators/pow_op.h"

#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math_gpu_utils.cuh"

namespace caffe2 {

namespace {

template <typename T>
using BlockReduce = cub::BlockReduce<T, CAFFE_CUDA_NUM_THREADS>;

#define CAFFE2_DECLARE_UNARY_POW_GRADIENT_CUDA_KERNEL(Func) \
  template <typename T>                                     \
  __global__ void Func##GradientCUDAKernel(                 \
      const int N, const T* dY, const T* Y, T* dX);
CAFFE2_DECLARE_UNARY_POW_GRADIENT_CUDA_KERNEL(Inv)
CAFFE2_DECLARE_UNARY_POW_GRADIENT_CUDA_KERNEL(Sqrt)
CAFFE2_DECLARE_UNARY_POW_GRADIENT_CUDA_KERNEL(Rsqrt)
#undef CAFFE2_DECLARE_UNARY_POW_GRADIENT_CUDA_KERNEL

#define CAFFE2_SPECIALIZED_UNARY_POW_GRADIENT_CUDA_KERNEL(T, Func, Op, coeff) \
  template <>                                                                 \
  __global__ void Func##GradientCUDAKernel<T>(                                \
      const int N, const T* dY, const T* Y, T* dX) {                          \
    CUDA_1D_KERNEL_LOOP(i, N) {                                               \
      dX[i] = coeff * Op(Y[i]) * dY[i];                                       \
    }                                                                         \
  }
CAFFE2_SPECIALIZED_UNARY_POW_GRADIENT_CUDA_KERNEL(
    float,
    Inv,
    math::gpu_utils::Square<float>,
    -1.0f)
CAFFE2_SPECIALIZED_UNARY_POW_GRADIENT_CUDA_KERNEL(
    float,
    Sqrt,
    math::gpu_utils::Inv<float>,
    0.5f)
CAFFE2_SPECIALIZED_UNARY_POW_GRADIENT_CUDA_KERNEL(
    float,
    Rsqrt,
    math::gpu_utils::Cube<float>,
    -0.5f)
#undef CAFFE2_DECLARE_UNARY_POW_GRADIENT_CUDA_KERNEL

template <typename T, bool kInPlace>
__global__ void SquareGradientCUDAKernel(
    const int N,
    const float* dY,
    const float* X,
    const float* Y,
    float* dX);

template <>
__global__ void SquareGradientCUDAKernel<float, true>(
    const int N,
    const float* dY,
    const float* /* X */,
    const float* Y,
    float* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = 2.0f * sqrtf(__ldg(Y + i)) * __ldg(dY + i);
#else
    dX[i] = 2.0f * sqrtf(Y[i]) * dY[i];
#endif
  }
}

template <>
__global__ void SquareGradientCUDAKernel<float, false>(
    const int N,
    const float* dY,
    const float* X,
    const float* /* Y */,
    float* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = 2.0f * __ldg(X + i) * __ldg(dY + i);
#else
    dX[i] = 2.0f * X[i] * dY[i];
#endif
  }
}

template <typename T, bool kInPlace>
__global__ void UnaryPowGradientCUDAKernel(
    const int N,
    const float exponent,
    const float* dY,
    const float* X,
    const float* Y,
    float* dX);

template <>
__global__ void UnaryPowGradientCUDAKernel<float, true>(
    const int N,
    const float exponent,
    const float* dY,
    const float* /* X */,
    const float* Y,
    float* dX) {
  const float b = (exponent - 1.0f) / exponent;
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = exponent * powf(__ldg(Y + i), b) * __ldg(dY + i);
#else
    dX[i] = exponent * powf(Y[i], b) * dY[i];
#endif
  }
}

template <>
__global__ void UnaryPowGradientCUDAKernel<float, false>(
    const int N,
    const float exponent,
    const float* dY,
    const float* X,
    const float* /* Y */,
    float* dX) {
  const float b = exponent - 1.0f;
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = exponent * powf(__ldg(X + i), b) * __ldg(dY + i);
#else
    dX[i] = exponent * powf(X[i], b) * dY[i];
#endif
  }
}

template <typename T>
__global__ void ComputeSinglePowBGradientCUDAKernel(
    const int N,
    const T* dC,
    const T* A,
    const T* C,
    T* dB);

template <>
__global__ void ComputeSinglePowBGradientCUDAKernel<float>(
    const int N,
    const float* dC,
    const float* A,
    const float* C,
    float* dB) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dB[i] = __ldg(C + i) * logf(__ldg(A + i)) * __ldg(dC + i);
#else
    dB[i] = C[i] * logf(A[i]) * dC[i];
#endif
  }
}

template <typename T>
__global__ void ComputeSimplePowGradientCUDAKernel(
    const int N,
    const T* dC,
    const T* A,
    const T* B,
    const T* C,
    T* dA,
    T* dB);

template <>
__global__ void ComputeSimplePowGradientCUDAKernel<float>(
    const int N,
    const float* dC,
    const float* A,
    const float* B,
    const float* C,
    float* dA,
    float* dB) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dA[i] =
        __ldg(B + i) * powf(__ldg(A + i), __ldg(B + i) - 1.0f) * __ldg(dC + i);
    dB[i] = __ldg(C + i) * logf(__ldg(A + i)) * __ldg(dC + i);
#else
    dA[i] = B[i] * powf(A[i], B[i] - 1.0f) * dC[i];
    dB[i] = C[i] * logf(A[i]) * dC[i];
#endif
  }
}

template <typename T>
struct PowAGradientFunctor {
  inline __device__ T operator()(const T w, const T x, const T y) const;
};

template <>
__device__ float PowAGradientFunctor<float>::
operator()(const float w, const float x, const float /* y */) const {
  return w * powf(x, w - 1.0f);
}

template <typename T>
struct PowBGradientFunctor {
  inline __device__ T operator()(const T w, const T x, const T y) const;
};

template <>
__device__ float PowBGradientFunctor<float>::
operator()(const float w, const float /* x */, const float y) const {
  return y * logf(w);
}

template <typename T, class Op, int D>
__global__ void ComputePowGradientCUDAKernel(
    const int outer_size,
    const int inner_size,
    const SimpleArray<int, D> Y_dims,
    const SimpleArray<int, D> Y_strides,
    const SimpleArray<int, D> W_strides,
    const SimpleArray<int, D> X_dims,
    const Op op,
    const T* dY,
    const T* W,
    const T* X,
    const T* Y,
    T* dX) {
  __shared__ typename BlockReduce<T>::TempStorage temp_storage;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T sum = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int X_index = i * inner_size + j;
      int Y_index = 0;
      int X_index_val = X_index;
#pragma unroll
      for (int d = D - 1; d >= 0; --d) {
        Y_index += (X_index_val % X_dims.data[d]) * Y_strides.data[d];
        X_index_val /= X_dims.data[d];
      }
      int W_index = 0;
      int Y_index_val = Y_index;
#pragma unroll
      for (int d = D - 1; d >= 0; --d) {
        W_index += (Y_index_val % Y_dims.data[d]) * W_strides.data[d];
        Y_index_val /= Y_dims.data[d];
      }
#if __CUDA_ARCH__ >= 350
      sum += op(__ldg(W + W_index), __ldg(X + i), __ldg(Y + Y_index)) *
          __ldg(dY + Y_index);
#else
      sum += op(W[W_index], X[i], Y[Y_index]) * dY[Y_index];
#endif
    }
    sum = BlockReduce<T>(temp_storage).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
      dX[i] = sum;
    }
    __syncthreads();
  }
}

template <typename T, class Op, int D>
void ComputePowGradientCUDAImpl(
    const int outer_size,
    const int inner_size,
    const int* Y_dims,
    const int* W_dims,
    const int* X_axes,
    const Op& op,
    const T* dY,
    const T* W,
    const T* X,
    const T* Y,
    T* dX,
    CUDAContext* context) {
  SimpleArray<int, D> Y_dims_arr;
  SimpleArray<int, D> Y_strides_arr;
  SimpleArray<int, D> W_strides_arr;
  SimpleArray<int, D> X_dims_arr;
  std::copy_n(Y_dims, D, Y_dims_arr.data);
  math::utils::ComputeTransposedStrides(D, Y_dims, X_axes, Y_strides_arr.data);
  int cur_stride = 1;
  for (int i = D - 1; i >= 0; --i) {
    W_strides_arr.data[i] = W_dims[i] == 1 ? 0 : cur_stride;
    cur_stride *= W_dims[i];
  }
  for (int i = 0; i < D; ++i) {
    X_dims_arr.data[i] = Y_dims[X_axes[i]];
  }
  ComputePowGradientCUDAKernel<T, Op, D>
      <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          outer_size,
          inner_size,
          Y_dims_arr,
          Y_strides_arr,
          W_strides_arr,
          X_dims_arr,
          op,
          dY,
          W,
          X,
          Y,
          dX);
}

template <typename T, class Op>
void ComputePowGradientCUDA(
    const std::vector<int>& Y_dims,
    const std::vector<int>& W_dims,
    const std::vector<int>& X_axes,
    const Op& op,
    const T* dY,
    const T* W,
    const T* X,
    const T* Y,
    T* dX,
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
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_2(
      ndim,
      ComputePowGradientCUDAImpl,
      T,
      Op,
      outer_size,
      inner_size,
      Y_dims.data(),
      W_dims.data(),
      X_transpose_axes.data(),
      op,
      dY,
      W,
      X,
      Y,
      dX,
      context);
}

} // namespace

template <>
template <typename T>
bool PowGradientOp<TensorTypes<float>, CUDAContext>::ComputeUnaryPowGradient(
    const int N,
    const T& exponent,
    const T* dY,
    const T* X,
    const T* Y,
    T* dX) {
  if (exponent == T(0)) {
    math::Set<T, CUDAContext>(N, 0, dX, &context_);
  } else if (exponent == T(1)) {
    if (dX != dY) {
      context_.template Copy<T, CUDAContext, CUDAContext>(N, dY, dX);
    }
  } else if (exponent == T(2)) {
    if (X == nullptr) {
      SquareGradientCUDAKernel<T, true>
          <<<CAFFE_GET_BLOCKS(N),
             CAFFE_CUDA_NUM_THREADS,
             0,
             context_.cuda_stream()>>>(N, dY, X, Y, dX);
    } else {
      SquareGradientCUDAKernel<T, false>
          <<<CAFFE_GET_BLOCKS(N),
             CAFFE_CUDA_NUM_THREADS,
             0,
             context_.cuda_stream()>>>(N, dY, X, Y, dX);
    }
  } else if (exponent == T(-1)) {
    InvGradientCUDAKernel<T>
        <<<CAFFE_GET_BLOCKS(N),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(N, dY, Y, dX);
  } else if (exponent == T(0.5)) {
    SqrtGradientCUDAKernel<T>
        <<<CAFFE_GET_BLOCKS(N),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(N, dY, Y, dX);
  } else if (exponent == T(-0.5)) {
    RsqrtGradientCUDAKernel<T>
        <<<CAFFE_GET_BLOCKS(N),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(N, dY, Y, dX);
  } else {
    if (X == nullptr) {
      UnaryPowGradientCUDAKernel<T, true>
          <<<CAFFE_GET_BLOCKS(N),
             CAFFE_CUDA_NUM_THREADS,
             0,
             context_.cuda_stream()>>>(N, exponent, dY, X, Y, dX);
    } else {
      UnaryPowGradientCUDAKernel<T, false>
          <<<CAFFE_GET_BLOCKS(N),
             CAFFE_CUDA_NUM_THREADS,
             0,
             context_.cuda_stream()>>>(N, exponent, dY, X, Y, dX);
    }
  }
  return true;
}

template <>
template <typename T>
bool PowGradientOp<TensorTypes<float>, CUDAContext>::ComputeSinglePowBGradient(
    const int N,
    const T* dC,
    const T* A,
    const T* C,
    T* dB) {
  buff_.Resize(N);
  T* buff_data = buff_.template mutable_data<T>();
  ComputeSinglePowBGradientCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(N),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(N, dC, A, C, buff_data);
  math::Sum<T, CUDAContext>(N, buff_data, dB, &context_, &scratch_);
  return true;
}

template <>
template <typename T>
bool PowGradientOp<TensorTypes<float>, CUDAContext>::ComputeBinaryPowGradient(
    const std::vector<int>& A_dims,
    const std::vector<int>& B_dims,
    const T* dC,
    const T* A,
    const T* B,
    const T* C,
    T* dA,
    T* dB) {
  if (A_dims == B_dims) {
    const int size = std::accumulate(
        A_dims.cbegin(), A_dims.cend(), 1, std::multiplies<int>());
    ComputeSimplePowGradientCUDAKernel<T>
        <<<CAFFE_GET_BLOCKS(size),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(size, dC, A, B, C, dA, dB);
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
  ComputePowGradientCUDA<T, PowAGradientFunctor<T>>(
      C_broadcast_dims,
      B_broadcast_dims,
      A_axes,
      PowAGradientFunctor<T>(),
      dC,
      B,
      A,
      C,
      dA,
      &context_);
  ComputePowGradientCUDA<T, PowBGradientFunctor<T>>(
      C_broadcast_dims,
      A_broadcast_dims,
      B_axes,
      PowBGradientFunctor<T>(),
      dC,
      A,
      B,
      C,
      dB,
      &context_);
  return true;
}

REGISTER_CUDA_OPERATOR(Pow, PowOp<TensorTypes<float>, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    PowGradient,
    PowGradientOp<TensorTypes<float>, CUDAContext>);

} // namespace caffe2
