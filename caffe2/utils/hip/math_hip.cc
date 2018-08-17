#include "hip/hip_runtime.h"
// Implements the math functions for GPU.

#include "caffe2/utils/math.h"

#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>

#include <thrust/functional.h>

#include "caffe2/core/hip/context_hip.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/fixed_divisor.h"
#include "caffe2/utils/math_utils.h"

#if THRUST_VERSION >= 100800
#define THRUST_SUPPORTS_PER_THREAD
#endif // THRUST_VERSION >= 100800

#define ROCBLAS_FP16 0

namespace caffe2 {
namespace math {

namespace {

#define DELEGATE_SIMPLE_HOST_DEVICE_BINARY_FUNCTOR(Func, expr)        \
  template <typename T>                                               \
  struct Func##Functor {                                              \
    inline __host__ __device__ T                                      \
    operator()(const T& lhs, const T& rhs) const {                    \
      return lhs expr rhs;                                            \
    }                                                                 \
  };                                                                  \
  template <>                                                         \
  struct Func##Functor<float16> {                                     \
    inline __host__ __device__ float16                                \
    operator()(const float16& lhs, const float16& rhs) const {        \
      return convert::To<float, float16>(convert::To<float16, float>( \
          lhs) expr convert::To<float16, float>(rhs));                \
    }                                                                 \
  };
DELEGATE_SIMPLE_HOST_DEVICE_BINARY_FUNCTOR(Add, +)
DELEGATE_SIMPLE_HOST_DEVICE_BINARY_FUNCTOR(Sub, -)
DELEGATE_SIMPLE_HOST_DEVICE_BINARY_FUNCTOR(Mul, *)
DELEGATE_SIMPLE_HOST_DEVICE_BINARY_FUNCTOR(Div, /)
#undef DELEGATE_SIMPLE_HOST_DEVICE_BINARY_FUNCTOR

template <typename TIn, typename TOut, class BinaryOperator>
__global__ void SimpleBinaryOpHIPKernel(
    const int N,
    const BinaryOperator op,
    const TIn* A,
    const TIn* B,
    TOut* C) {
  HIP_1D_KERNEL_LOOP(i, N) {
    C[i] = op(A[i], B[i]);
  }
}

template <typename TIn, typename TOut, class BinaryOperator, bool broadcast_1st>
__global__ void RowwiseBinaryOpHIPKernel(
    const int size,
    const int cols,
    const BinaryOperator op,
    const TIn* A,
    const TIn* B,
    TOut* C) {
  HIP_1D_KERNEL_LOOP(C_index, size) {
    const int j = C_index % cols;
    const int A_index = broadcast_1st ? j : C_index;
    const int B_index = broadcast_1st ? C_index : j;
    C[C_index] = op(A[A_index], B[B_index]);
  }
}

template <typename TIn, typename TOut, class BinaryOperator, bool broadcast_1st>
__global__ void ColwiseBinaryOpHIPKernel(
    const int size,
    const int cols,
    const BinaryOperator op,
    const TIn* A,
    const TIn* B,
    TOut* C) {
  HIP_1D_KERNEL_LOOP(C_index, size) {
    const int i = C_index / cols;
    const int A_index = broadcast_1st ? i : C_index;
    const int B_index = broadcast_1st ? C_index : i;
    C[C_index] = op(A[A_index], B[B_index]);
  }
}

template <typename TIn, typename TOut, class BinaryOperator, int D>
__global__ void BroadcastBinaryOpHIPKernel(
    const int size,
    const SimpleArray<int, D> A_strides,
    const SimpleArray<int, D> B_strides,
    const SimpleArray<FixedDivisor<int>, D> C_dims,
    const BinaryOperator op,
    const TIn* A,
    const TIn* B,
    TOut* C) {
  HIP_1D_KERNEL_LOOP(C_index, size) {
    int A_index = 0;
    int B_index = 0;
    int C_index_val = C_index;
#pragma unroll
    for (int i = D - 1; i >= 0; --i) {
      int d;
      C_dims.data[i].DivMod(C_index_val, &C_index_val, &d);
      A_index += d * A_strides.data[i];
      B_index += d * B_strides.data[i];
    }
    C[C_index] = op(A[A_index], B[B_index]);
  }
}

template <typename TIn, typename TOut, class BinaryOperator>
void BinaryOpWith2DBroadcasting(
    const int rows,
    const int cols,
    const bool rowwise_broadcast,
    const bool broadcast_1st,
    const BinaryOperator& op,
    const TIn* A,
    const TIn* B,
    TOut* C,
    HIPContext* context) {
  if (rows == 0 || cols == 0) {
    return;
  }
  const int size = rows * cols;
  if (rowwise_broadcast) {
    if (broadcast_1st) {
      hipLaunchKernelGGL(
          (RowwiseBinaryOpHIPKernel<TIn, TOut, BinaryOperator, true>),
          dim3(CAFFE_GET_BLOCKS(size)),
          dim3(CAFFE_HIP_NUM_THREADS),
          0,
          context->hip_stream(),
          size,
          cols,
          op,
          A,
          B,
          C);
    } else {
      hipLaunchKernelGGL(
          (RowwiseBinaryOpHIPKernel<TIn, TOut, BinaryOperator, false>),
          dim3(CAFFE_GET_BLOCKS(size)),
          dim3(CAFFE_HIP_NUM_THREADS),
          0,
          context->hip_stream(),
          size,
          cols,
          op,
          A,
          B,
          C);
    }
  } else {
    if (broadcast_1st) {
      hipLaunchKernelGGL(
          (ColwiseBinaryOpHIPKernel<TIn, TOut, BinaryOperator, true>),
          dim3(CAFFE_GET_BLOCKS(size)),
          dim3(CAFFE_HIP_NUM_THREADS),
          0,
          context->hip_stream(),
          size,
          cols,
          op,
          A,
          B,
          C);
    } else {
      hipLaunchKernelGGL(
          (ColwiseBinaryOpHIPKernel<TIn, TOut, BinaryOperator, false>),
          dim3(CAFFE_GET_BLOCKS(size)),
          dim3(CAFFE_HIP_NUM_THREADS),
          0,
          context->hip_stream(),
          size,
          cols,
          op,
          A,
          B,
          C);
    }
  }
}

template <typename TIn, typename TOut, class BinaryOperator, int D>
void BroadcastBinaryOpImpl(
    const int* A_dims,
    const int* B_dims,
    const int* C_dims,
    const BinaryOperator& op,
    const TIn* A,
    const TIn* B,
    TOut* C,
    HIPContext* context) {
  SimpleArray<int, D> A_strides_array;
  SimpleArray<int, D> B_strides_array;
  SimpleArray<FixedDivisor<int>, D> C_dims_array;
  int A_stride = 1;
  int B_stride = 1;
  for (int i = D - 1; i >= 0; --i) {
    if (C_dims[i] == 0) {
      return;
    }
    A_strides_array.data[i] = A_dims[i] == 1 ? 0 : A_stride;
    B_strides_array.data[i] = B_dims[i] == 1 ? 0 : B_stride;
    A_stride *= A_dims[i];
    B_stride *= B_dims[i];
    C_dims_array.data[i] = FixedDivisor<int>(C_dims[i]);
  }
  const int size =
      std::accumulate(C_dims, C_dims + D, 1, std::multiplies<int>());
  hipLaunchKernelGGL(
      (BroadcastBinaryOpHIPKernel<TIn, TOut, BinaryOperator, D>),
      dim3(CAFFE_GET_BLOCKS(size)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      size,
      A_strides_array,
      B_strides_array,
      C_dims_array,
      op,
      A,
      B,
      C);
}

template <typename TIn, typename TOut, class BinaryOperator>
void BroadcastBinaryOp(
    const int A_ndim,
    const int* A_dims,
    const int B_ndim,
    const int* B_dims,
    const BinaryOperator& op,
    const TIn* A,
    const TIn* B,
    TOut* C,
    HIPContext* context) {
  const int ndim = std::max(A_ndim, B_ndim);
  std::vector<int> A_dims_array(ndim);
  std::vector<int> B_dims_array(ndim);
  std::vector<int> C_dims_array(ndim);
  utils::ComputeBroadcastBinaryOpDims(
      A_ndim,
      A_dims,
      B_ndim,
      B_dims,
      A_dims_array.data(),
      B_dims_array.data(),
      C_dims_array.data());
  if (A_dims_array == B_dims_array) {
    const int size = std::accumulate(
        C_dims_array.cbegin(), C_dims_array.cend(), 1, std::multiplies<int>());
    hipLaunchKernelGGL(
        (SimpleBinaryOpHIPKernel<TIn, TOut, BinaryOperator>),
        dim3(CAFFE_GET_BLOCKS(size)),
        dim3(CAFFE_HIP_NUM_THREADS),
        0,
        context->hip_stream(),
        size,
        op,
        A,
        B,
        C);
    return;
  }
  int rows;
  int cols;
  bool broadcast_1st;
  if (utils::IsRowwiseBroadcastBinaryOp(
          ndim,
          A_dims_array.data(),
          B_dims_array.data(),
          &rows,
          &cols,
          &broadcast_1st)) {
    BinaryOpWith2DBroadcasting<TIn, TOut, BinaryOperator>(
        rows, cols, true, broadcast_1st, op, A, B, C, context);
    return;
  }
  if (utils::IsColwiseBroadcastBinaryOp(
          ndim,
          A_dims_array.data(),
          B_dims_array.data(),
          &rows,
          &cols,
          &broadcast_1st)) {
    BinaryOpWith2DBroadcasting<TIn, TOut, BinaryOperator>(
        rows, cols, false, broadcast_1st, op, A, B, C, context);
    return;
  }
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_3(
      ndim,
      BroadcastBinaryOpImpl,
      TIn,
      TOut,
      BinaryOperator,
      A_dims_array.data(),
      B_dims_array.data(),
      C_dims_array.data(),
      op,
      A,
      B,
      C,
      context);
}

} // namespace

#define DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(T, Func, op)            \
  __global__ void Func##HIPKernel(const int N, const T* X, T* Y) { \
    HIP_1D_KERNEL_LOOP(i, N) {                                     \
      Y[i] = op(X[i]);                                             \
    }                                                              \
  }                                                                \
  template <>                                                      \
  void Func<T, HIPContext>(                                        \
      const int N, const T* x, T* y, HIPContext* context) {        \
    hipLaunchKernelGGL(                                            \
        (Func##HIPKernel),                                         \
        CAFFE_GET_BLOCKS(N),                                       \
        CAFFE_HIP_NUM_THREADS,                                     \
        0,                                                         \
        context->hip_stream(),                                     \
        N,                                                         \
        x,                                                         \
        y);                                                        \
  }

DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(float, Exp, expf)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(float, Log, logf)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(float, Cos, cosf)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(float, Acos, acosf)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(float, Sin, sinf)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(float, Asin, asinf)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(float, Tan, tanf)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(float, Atan, atanf)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(float, Sinh, sinhf)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(float, Cosh, coshf)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(float, Tanh, tanhf)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(float, Abs, fabsf)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(float, Sqr, utils::Square<float>)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(float, Sqrt, sqrtf)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(float, Rsqrt, rsqrtf)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(float, Cbrt, cbrtf)

DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(float, Cube, utils::Cube<float>)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(double, Cube, utils::Cube<double>)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(
    std::int32_t,
    Cube,
    utils::Cube<std::int32_t>)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(
    std::int64_t,
    Cube,
    utils::Cube<std::int64_t>)

DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(bool, Not, utils::Not)

DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(float, Neg, utils::Negate<float>)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(double, Neg, utils::Negate<double>)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(
    std::int32_t,
    Neg,
    utils::Negate<std::int32_t>)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(
    std::int64_t,
    Neg,
    utils::Negate<std::int64_t>)

DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(float, Sign, utils::Sign<float>)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(double, Sign, utils::Sign<double>)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(
    std::int32_t,
    Sign,
    utils::Sign<std::int32_t>)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(
    std::int64_t,
    Sign,
    utils::Sign<std::int64_t>)

DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(float, Inv, utils::Inv<float>)
DELEGATE_SIMPLE_HIP_UNARY_FUNCTION(double, Inv, utils::Inv<double>)

#undef DELEGATE_SIMPLE_HIP_UNARY_FUNCTION

#define DELEGATE_SINCOS_HIP_FUNCTION(T, fn)                         \
  __global__ void _Kernel_##T##_##SinCos(                           \
      const int N, const T* x, T* ys, T* yc) {                      \
    HIP_1D_KERNEL_LOOP(i, N) {                                      \
      fn(__ldg(x + i), ys + i, yc + i);                             \
    }                                                               \
  }                                                                 \
  template <>                                                       \
  void SinCos<T, HIPContext>(                                       \
      const int N, const T* x, T* ys, T* yc, HIPContext* context) { \
    hipLaunchKernelGGL(                                             \
        (_Kernel_##T##_##SinCos),                                   \
        CAFFE_GET_BLOCKS(N),                                        \
        CAFFE_HIP_NUM_THREADS,                                      \
        0,                                                          \
        context->hip_stream(),                                      \
        N,                                                          \
        x,                                                          \
        ys,                                                         \
        yc);                                                        \
  }

DELEGATE_SINCOS_HIP_FUNCTION(float, sincosf)
DELEGATE_SINCOS_HIP_FUNCTION(double, sincos)

#define DELEGATE_SIMPLE_HIP_BINARY_FUNCTION(TIn, TOut, Func, Op)               \
  template <>                                                                  \
  void Func<TIn, HIPContext>(                                                  \
      const int N, const TIn* A, const TIn* B, TOut* C, HIPContext* context) { \
    hipLaunchKernelGGL(                                                        \
        (SimpleBinaryOpHIPKernel<TIn, TOut, Op<TIn>>),                         \
        CAFFE_GET_BLOCKS(N),                                                   \
        CAFFE_HIP_NUM_THREADS,                                                 \
        0,                                                                     \
        context->hip_stream(),                                                 \
        N,                                                                     \
        Op<TIn>(),                                                             \
        A,                                                                     \
        B,                                                                     \
        C);                                                                    \
  }

#define DEFINE_SIMPLE_HIP_COMPARE_FUNCTION(Func, Op)                \
  DELEGATE_SIMPLE_HIP_BINARY_FUNCTION(std::int32_t, bool, Func, Op) \
  DELEGATE_SIMPLE_HIP_BINARY_FUNCTION(std::int64_t, bool, Func, Op) \
  DELEGATE_SIMPLE_HIP_BINARY_FUNCTION(float, bool, Func, Op)        \
  DELEGATE_SIMPLE_HIP_BINARY_FUNCTION(double, bool, Func, Op)       \
  DELEGATE_SIMPLE_HIP_BINARY_FUNCTION(bool, bool, Func, Op)

DEFINE_SIMPLE_HIP_COMPARE_FUNCTION(EQ, thrust::equal_to)
DEFINE_SIMPLE_HIP_COMPARE_FUNCTION(NE, thrust::not_equal_to)
DEFINE_SIMPLE_HIP_COMPARE_FUNCTION(LT, thrust::less)
DEFINE_SIMPLE_HIP_COMPARE_FUNCTION(LE, thrust::less_equal)
DEFINE_SIMPLE_HIP_COMPARE_FUNCTION(GT, thrust::greater)
DEFINE_SIMPLE_HIP_COMPARE_FUNCTION(GE, thrust::greater_equal)

#undef DEFINE_SIMPLE_HIP_COMPARE_FUNCTION

#define DEFINE_SIMPLE_HIP_BINARY_FUNCTION(Func, Op)                         \
  DELEGATE_SIMPLE_HIP_BINARY_FUNCTION(std::int32_t, std::int32_t, Func, Op) \
  DELEGATE_SIMPLE_HIP_BINARY_FUNCTION(std::int64_t, std::int64_t, Func, Op) \
  DELEGATE_SIMPLE_HIP_BINARY_FUNCTION(float, float, Func, Op)               \
  DELEGATE_SIMPLE_HIP_BINARY_FUNCTION(double, double, Func, Op)             \
  DELEGATE_SIMPLE_HIP_BINARY_FUNCTION(float16, float16, Func, Op)

DEFINE_SIMPLE_HIP_BINARY_FUNCTION(Add, AddFunctor)
DEFINE_SIMPLE_HIP_BINARY_FUNCTION(Sub, SubFunctor)
DEFINE_SIMPLE_HIP_BINARY_FUNCTION(Mul, MulFunctor)
DEFINE_SIMPLE_HIP_BINARY_FUNCTION(Div, DivFunctor)

#undef DEFINE_SIMPLE_HIP_BINARY_FUNCTION

DELEGATE_SIMPLE_HIP_BINARY_FUNCTION(bool, bool, And, thrust::logical_and)
DELEGATE_SIMPLE_HIP_BINARY_FUNCTION(bool, bool, Or, thrust::logical_or)
DELEGATE_SIMPLE_HIP_BINARY_FUNCTION(bool, bool, Xor, thrust::bit_xor)

#define DEFINE_SIMPLE_HIP_BITWISE_BINARY_FUNCTION(Func, Op)                 \
  DELEGATE_SIMPLE_HIP_BINARY_FUNCTION(bool, bool, Func, Op)                 \
  DELEGATE_SIMPLE_HIP_BINARY_FUNCTION(std::int32_t, std::int32_t, Func, Op) \
  DELEGATE_SIMPLE_HIP_BINARY_FUNCTION(std::int64_t, std::int64_t, Func, Op)

DEFINE_SIMPLE_HIP_BITWISE_BINARY_FUNCTION(BitwiseAnd, thrust::bit_and)
DEFINE_SIMPLE_HIP_BITWISE_BINARY_FUNCTION(BitwiseOr, thrust::bit_or)
DEFINE_SIMPLE_HIP_BITWISE_BINARY_FUNCTION(BitwiseXor, thrust::bit_xor)

#undef DEFINE_SIMPLE_HIP_BITWISE_BINARY_FUNCTION

DELEGATE_SIMPLE_HIP_BINARY_FUNCTION(float, float, ElemwiseMax, thrust::maximum);

#undef DELEGATE_SIMPLE_HIP_BINARY_FUNCTION

#define DELEGATE_2D_BROADCAST_HIP_BINARY_FUNCTION(TIn, TOut, Func, Op) \
  template <>                                                          \
  void Rowwise##Func<TIn, HIPContext, true>(                           \
      const int rows,                                                  \
      const int cols,                                                  \
      const TIn* A,                                                    \
      const TIn* B,                                                    \
      TOut* C,                                                         \
      HIPContext* context) {                                           \
    if (rows == 0 || cols == 0) {                                      \
      return;                                                          \
    }                                                                  \
    const int size = rows * cols;                                      \
    hipLaunchKernelGGL(                                                \
        RowwiseBinaryOpHIPKernel<TIn, TOut, Op<TIn>, true>,            \
        CAFFE_GET_BLOCKS(size),                                        \
        CAFFE_HIP_NUM_THREADS,                                         \
        0,                                                             \
        context->hip_stream(),                                         \
        size,                                                          \
        cols,                                                          \
        Op<TIn>(),                                                     \
        A,                                                             \
        B,                                                             \
        C);                                                            \
  }                                                                    \
  template <>                                                          \
  void Rowwise##Func<TIn, HIPContext, false>(                          \
      const int rows,                                                  \
      const int cols,                                                  \
      const TIn* A,                                                    \
      const TIn* B,                                                    \
      TOut* C,                                                         \
      HIPContext* context) {                                           \
    if (rows == 0 || cols == 0) {                                      \
      return;                                                          \
    }                                                                  \
    const int size = rows * cols;                                      \
    hipLaunchKernelGGL(                                                \
        RowwiseBinaryOpHIPKernel<TIn, TOut, Op<TIn>, false>,           \
        CAFFE_GET_BLOCKS(size),                                        \
        CAFFE_HIP_NUM_THREADS,                                         \
        0,                                                             \
        context->hip_stream(),                                         \
        size,                                                          \
        cols,                                                          \
        Op<TIn>(),                                                     \
        A,                                                             \
        B,                                                             \
        C);                                                            \
  }                                                                    \
  template <>                                                          \
  void Colwise##Func<TIn, HIPContext, true>(                           \
      const int rows,                                                  \
      const int cols,                                                  \
      const TIn* A,                                                    \
      const TIn* B,                                                    \
      TOut* C,                                                         \
      HIPContext* context) {                                           \
    if (rows == 0 || cols == 0) {                                      \
      return;                                                          \
    }                                                                  \
    const int size = rows * cols;                                      \
    hipLaunchKernelGGL(                                                \
        ColwiseBinaryOpHIPKernel<TIn, TOut, Op<TIn>, true>,            \
        CAFFE_GET_BLOCKS(size),                                        \
        CAFFE_HIP_NUM_THREADS,                                         \
        0,                                                             \
        context->hip_stream(),                                         \
        size,                                                          \
        cols,                                                          \
        Op<TIn>(),                                                     \
        A,                                                             \
        B,                                                             \
        C);                                                            \
  }                                                                    \
  template <>                                                          \
  void Colwise##Func<TIn, HIPContext, false>(                          \
      const int rows,                                                  \
      const int cols,                                                  \
      const TIn* A,                                                    \
      const TIn* B,                                                    \
      TOut* C,                                                         \
      HIPContext* context) {                                           \
    if (rows == 0 || cols == 0) {                                      \
      return;                                                          \
    }                                                                  \
    const int size = rows * cols;                                      \
    hipLaunchKernelGGL(                                                \
        ColwiseBinaryOpHIPKernel<TIn, TOut, Op<TIn>, false>,           \
        CAFFE_GET_BLOCKS(size),                                        \
        CAFFE_HIP_NUM_THREADS,                                         \
        0,                                                             \
        context->hip_stream(),                                         \
        size,                                                          \
        cols,                                                          \
        Op<TIn>(),                                                     \
        A,                                                             \
        B,                                                             \
        C);                                                            \
  }

#define DEFINE_2D_BROADCAST_HIP_COMPARE_FUNCTION(Func, Op)                \
  DELEGATE_2D_BROADCAST_HIP_BINARY_FUNCTION(std::int32_t, bool, Func, Op) \
  DELEGATE_2D_BROADCAST_HIP_BINARY_FUNCTION(std::int64_t, bool, Func, Op) \
  DELEGATE_2D_BROADCAST_HIP_BINARY_FUNCTION(float, bool, Func, Op)        \
  DELEGATE_2D_BROADCAST_HIP_BINARY_FUNCTION(double, bool, Func, Op)       \
  DELEGATE_2D_BROADCAST_HIP_BINARY_FUNCTION(bool, bool, Func, Op)

DEFINE_2D_BROADCAST_HIP_COMPARE_FUNCTION(EQ, thrust::equal_to)
DEFINE_2D_BROADCAST_HIP_COMPARE_FUNCTION(NE, thrust::not_equal_to)
DEFINE_2D_BROADCAST_HIP_COMPARE_FUNCTION(LT, thrust::less)
DEFINE_2D_BROADCAST_HIP_COMPARE_FUNCTION(LE, thrust::less_equal)
DEFINE_2D_BROADCAST_HIP_COMPARE_FUNCTION(GT, thrust::greater)
DEFINE_2D_BROADCAST_HIP_COMPARE_FUNCTION(GE, thrust::greater_equal)

#undef DEFINE_2D_BROADCAST_HIP_COMPARE_FUNCTION

#define DEFINE_2D_BROADCAST_HIP_BINARY_FUNCTION(Func, Op)             \
  DELEGATE_2D_BROADCAST_HIP_BINARY_FUNCTION(                          \
      std::int32_t, std::int32_t, Func, Op)                           \
  DELEGATE_2D_BROADCAST_HIP_BINARY_FUNCTION(                          \
      std::int64_t, std::int64_t, Func, Op)                           \
  DELEGATE_2D_BROADCAST_HIP_BINARY_FUNCTION(float, float, Func, Op)   \
  DELEGATE_2D_BROADCAST_HIP_BINARY_FUNCTION(double, double, Func, Op) \
  DELEGATE_2D_BROADCAST_HIP_BINARY_FUNCTION(float16, float16, Func, Op)

DEFINE_2D_BROADCAST_HIP_BINARY_FUNCTION(Add, AddFunctor)
DEFINE_2D_BROADCAST_HIP_BINARY_FUNCTION(Sub, SubFunctor)
DEFINE_2D_BROADCAST_HIP_BINARY_FUNCTION(Mul, MulFunctor)
DEFINE_2D_BROADCAST_HIP_BINARY_FUNCTION(Div, DivFunctor)

#undef DEFINE_2D_BROADCAST_HIP_BINARY_FUNCTION

DELEGATE_2D_BROADCAST_HIP_BINARY_FUNCTION(bool, bool, And, thrust::logical_and)
DELEGATE_2D_BROADCAST_HIP_BINARY_FUNCTION(bool, bool, Or, thrust::logical_or)
DELEGATE_2D_BROADCAST_HIP_BINARY_FUNCTION(bool, bool, Xor, thrust::bit_xor)

#define DEFINE_2D_BROADCAST_HIP_BITWISE_BINARY_FUNCTION(Func, Op) \
  DELEGATE_2D_BROADCAST_HIP_BINARY_FUNCTION(bool, bool, Func, Op) \
  DELEGATE_2D_BROADCAST_HIP_BINARY_FUNCTION(                      \
      std::int32_t, std::int32_t, Func, Op)                       \
  DELEGATE_2D_BROADCAST_HIP_BINARY_FUNCTION(                      \
      std::int64_t, std::int64_t, Func, Op)

DEFINE_2D_BROADCAST_HIP_BITWISE_BINARY_FUNCTION(BitwiseAnd, thrust::bit_and)
DEFINE_2D_BROADCAST_HIP_BITWISE_BINARY_FUNCTION(BitwiseOr, thrust::bit_or)
DEFINE_2D_BROADCAST_HIP_BITWISE_BINARY_FUNCTION(BitwiseXor, thrust::bit_xor)

#undef DEFINE_2D_BROADCAST_HIP_BITWISE_BINARY_FUNCTION

#undef DELEGATE_2D_BROADCAST_HIP_BINARY_FUNCTION

#define DELEGATE_BROADCAST_HIP_BINARY_FUNCTION(TIn, TOut, Func, Op)   \
  template <>                                                         \
  void Func<TIn, HIPContext>(                                         \
      const int A_ndim,                                               \
      const int* A_dims,                                              \
      const int B_ndim,                                               \
      const int* B_dims,                                              \
      const TIn* A,                                                   \
      const TIn* B,                                                   \
      TOut* C,                                                        \
      HIPContext* context) {                                          \
    BroadcastBinaryOp<TIn, TOut, Op<TIn>>(                            \
        A_ndim, A_dims, B_ndim, B_dims, Op<TIn>(), A, B, C, context); \
  }

#define DEFINE_BROADCAST_HIP_COMPARE_FUNCTION(Func, Op)                \
  DELEGATE_BROADCAST_HIP_BINARY_FUNCTION(std::int32_t, bool, Func, Op) \
  DELEGATE_BROADCAST_HIP_BINARY_FUNCTION(std::int64_t, bool, Func, Op) \
  DELEGATE_BROADCAST_HIP_BINARY_FUNCTION(float, bool, Func, Op)        \
  DELEGATE_BROADCAST_HIP_BINARY_FUNCTION(double, bool, Func, Op)       \
  DELEGATE_BROADCAST_HIP_BINARY_FUNCTION(bool, bool, Func, Op)

DEFINE_BROADCAST_HIP_COMPARE_FUNCTION(EQ, thrust::equal_to)
DEFINE_BROADCAST_HIP_COMPARE_FUNCTION(NE, thrust::not_equal_to)
DEFINE_BROADCAST_HIP_COMPARE_FUNCTION(LT, thrust::less)
DEFINE_BROADCAST_HIP_COMPARE_FUNCTION(LE, thrust::less_equal)
DEFINE_BROADCAST_HIP_COMPARE_FUNCTION(GT, thrust::greater)
DEFINE_BROADCAST_HIP_COMPARE_FUNCTION(GE, thrust::greater_equal)

#undef DEFINE_BROADCAST_HIP_COMPARE_FUNCTION

#define DEFINE_BROADCAST_HIP_BINARY_FUNCTION(Func, Op)                         \
  DELEGATE_BROADCAST_HIP_BINARY_FUNCTION(std::int32_t, std::int32_t, Func, Op) \
  DELEGATE_BROADCAST_HIP_BINARY_FUNCTION(std::int64_t, std::int64_t, Func, Op) \
  DELEGATE_BROADCAST_HIP_BINARY_FUNCTION(float, float, Func, Op)               \
  DELEGATE_BROADCAST_HIP_BINARY_FUNCTION(double, double, Func, Op)             \
  DELEGATE_BROADCAST_HIP_BINARY_FUNCTION(float16, float16, Func, Op)

DEFINE_BROADCAST_HIP_BINARY_FUNCTION(Add, AddFunctor)
DEFINE_BROADCAST_HIP_BINARY_FUNCTION(Sub, SubFunctor)
DEFINE_BROADCAST_HIP_BINARY_FUNCTION(Mul, MulFunctor)
DEFINE_BROADCAST_HIP_BINARY_FUNCTION(Div, DivFunctor)

#undef DEFINE_BROADCAST_HIP_BINARY_FUNCTION

DELEGATE_BROADCAST_HIP_BINARY_FUNCTION(bool, bool, And, thrust::logical_and)
DELEGATE_BROADCAST_HIP_BINARY_FUNCTION(bool, bool, Or, thrust::logical_or)
DELEGATE_BROADCAST_HIP_BINARY_FUNCTION(bool, bool, Xor, thrust::bit_xor)

#define DEFINE_BROADCAST_HIP_BITWISE_BINARY_FUNCTION(Func, Op)                 \
  DELEGATE_BROADCAST_HIP_BINARY_FUNCTION(bool, bool, Func, Op)                 \
  DELEGATE_BROADCAST_HIP_BINARY_FUNCTION(std::int32_t, std::int32_t, Func, Op) \
  DELEGATE_BROADCAST_HIP_BINARY_FUNCTION(std::int64_t, std::int64_t, Func, Op)

DEFINE_BROADCAST_HIP_BITWISE_BINARY_FUNCTION(BitwiseAnd, thrust::bit_and)
DEFINE_BROADCAST_HIP_BITWISE_BINARY_FUNCTION(BitwiseOr, thrust::bit_or)
DEFINE_BROADCAST_HIP_BITWISE_BINARY_FUNCTION(BitwiseXor, thrust::bit_xor)

#undef DEFINE_BROADCAST_HIP_BITWISE_BINARY_FUNCTION

#undef DELEGATE_BROADCAST_HIP_BINARY_FUNCTION

#define DELEGATE_REDUCTION_FUNCTION(T, Funcname, func)                  \
  template <>                                                           \
  void Funcname<T, HIPContext>(                                         \
      const int N,                                                      \
      const T* src,                                                     \
      T* dst,                                                           \
      Tensor* scratch_ptr,                                              \
      HIPContext* context) {                                            \
    size_t memRequired = 0;                                             \
    cub::DeviceReduce::func(                                            \
        nullptr, memRequired, src, dst, N, context->hip_stream());      \
    auto buffer_size =                                                  \
        static_cast<TIndex>((memRequired + sizeof(T) - 1) / sizeof(T)); \
    scratch_ptr->Resize(std::vector<TIndex>{buffer_size});              \
    cub::DeviceReduce::func(                                            \
        static_cast<void*>(scratch_ptr->mutable_data<T>()),             \
        memRequired,                                                    \
        src,                                                            \
        dst,                                                            \
        N,                                                              \
        context->hip_stream());                                         \
  }

DELEGATE_REDUCTION_FUNCTION(float, ReduceMin, Min)
DELEGATE_REDUCTION_FUNCTION(float, ReduceMax, Max)
DELEGATE_REDUCTION_FUNCTION(int32_t, ReduceMax, Max)
DELEGATE_REDUCTION_FUNCTION(int64_t, ReduceMax, Max)

#undef DELEGATE_REDUCTION_FUNCTION

// Caffe2 gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template <>
void Gemm<float, HIPContext>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const float* B,
    const float beta,
    float* C,
    HIPContext* context,
    TensorProto::DataType math_type) {
  // Note that rocblas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  rocblas_operation cuTransA = (TransA == CblasNoTrans)
      ? rocblas_operation_none
      : rocblas_operation_transpose;
  rocblas_operation cuTransB = (TransB == CblasNoTrans)
      ? rocblas_operation_none
      : rocblas_operation_transpose;
  ROCBLAS_ENFORCE(rocblas_sgemm(
      context->rocblas_handle(),
      cuTransB,
      cuTransA,
      N,
      M,
      K,
      &alpha,
      B,
      ldb,
      A,
      lda,
      &beta,
      C,
      N));
}

template <>
void Gemm<float16, HIPContext>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float16* A,
    const float16* B,
    const float beta,
    float16* C,
    HIPContext* context,
    TensorProto::DataType math_type) {
  CAFFE_THROW("Unsupported math type");
#if ROCBLAS_FP16 // rocblas does not support fp16 yet
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  rocblas_operation cuTransA = (TransA == CblasNoTrans)
      ? rocblas_operation_none
      : rocblas_operation_transpose;
  rocblas_operation cuTransB = (TransB == CblasNoTrans)
      ? rocblas_operation_none
      : rocblas_operation_transpose;
  if (math_type == TensorProto_DataType_FLOAT) {
    ROCBLAS_CHECK(rocblas_sgemmEx(
        context->rocblas_handle(),
        cuTransB,
        cuTransA,
        N,
        M,
        K,
        &alpha,
        B,
        CUDA_R_16F,
        ldb,
        A,
        CUDA_R_16F,
        lda,
        &beta,
        C,
        CUDA_R_16F,
        N));

  } else if (math_type == TensorProto_DataType_FLOAT16) {
    // convert alpha, beta from float -> __half
    /*auto alpha_fp16 = convert::floatToHalf(alpha);
    auto beta_fp16 = convert::floatToHalf(beta);

    // call cublasHgemm
    ROCBLAS_CHECK(cublasHgemm(
        context->rocblas_handle(),
        cuTransB,
        cuTransA,
        N,
        M,
        K,
        &alpha_fp16,
        (const __half*)B,
        ldb,
        (const __half*)A,
        lda,
        &beta_fp16,
        (__half*)C,
        N));*/
  } else {
    // fail
    CAFFE_THROW("Unsupported math type");
  }
#endif
}

template <>
void BiasCHW<float, HIPContext>(
    const float* bias,
    const float* bias_multiplier,
    const int bias_channels,
    const int image_size,
    float* image,
    HIPContext* context) {
  Gemm<float, HIPContext>(
      CblasNoTrans,
      CblasNoTrans,
      bias_channels,
      image_size,
      1,
      1,
      bias,
      bias_multiplier,
      1,
      image,
      context);
}

template <>
void GemmBatched<float, HIPContext>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float** A,
    const float** B,
    const float beta,
    float** C,
    HIPContext* context,
    TensorProto::DataType math_type) {
  // rocblas doesn't support SgemmBatched yet.
  for (int i = 0; i < batch_size; ++i) {
    Gemm<float, HIPContext>(
        TransA,
        TransB,
        M,
        N,
        K,
        alpha,
        A[i],
        B[i],
        beta,
        C[i],
        context,
        math_type);
  }
}

template <>
void GemmStridedBatched<float, HIPContext>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const int A_stride,
    const float* B,
    const int B_stride,
    const float beta,
    float* C,
    const int C_stride,
    HIPContext* context,
    TensorProto::DataType math_type) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransB == CblasNoTrans) ? N : K;
  const rocblas_operation cuTransA = (TransA == CblasNoTrans)
      ? rocblas_operation_none
      : rocblas_operation_transpose;
  const rocblas_operation cuTransB = (TransB == CblasNoTrans)
      ? rocblas_operation_none
      : rocblas_operation_transpose;
  ROCBLAS_ENFORCE(rocblas_sgemm_strided_batched(
      context->rocblas_handle(),
      cuTransB,
      cuTransA,
      N,
      M,
      K,
      &alpha,
      B,
      ldb,
      B_stride,
      A,
      lda,
      A_stride,
      &beta,
      C,
      N,
      C_stride,
      batch_size));
}

template <>
void GemmStridedBatched<float16, HIPContext>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float16* A,
    const int A_stride,
    const float16* B,
    const int B_stride,
    const float beta,
    float16* C,
    const int C_stride,
    HIPContext* context,
    TensorProto::DataType math_type) {
#if ROCBLAS_FP16 // rocblas does not support fp16 yet
  if (math_type == TensorProto_DataType_FLOAT) {
    // loop over matrices in the batch
    for (int i = 0; i < batch_size; ++i) {
      math::Gemm<float16, HIPContext>(
          TransA,
          TransB,
          M,
          N,
          K,
          alpha,
          A + a_stride * i,
          B + b_stride * i,
          beta,
          C + c_stride * i,
          context);
    }
  } else if (math_type == TensorProto_DataType_FLOAT16) {
    // Note that cublas follows fortran order, so the order is different from
    // the cblas convention.
    const int lda = (TransA == CblasNoTrans) ? K : M;
    const int ldb = (TransB == CblasNoTrans) ? N : K;
    const rocblas_operation cuTransA = (TransA == CblasNoTrans)
        ? rocblas_operation_none
        : rocblas_operation_transpose;
    const rocblas_operation cuTransB = (TransB == CblasNoTrans)
        ? rocblas_operation_none
        : rocblas_operation_transpose;

    // convert alpha, beta from float -> __half
    auto alpha_fp16 = convert::floatToHalf(alpha);
    auto beta_fp16 = convert::floatToHalf(beta);
    ROCBLAS_ENFORCE(cublasHgemmStridedBatched(
        context->rocblas_handle(),
        cuTransB,
        cuTransA,
        N,
        M,
        K,
        &alpha_fp16,
        (const __half*)B,
        ldb,
        B_stride,
        (const __half*)A,
        lda,
        A_stride,
        &beta_fp16,
        (__half*)C,
        N,
        C_stride,
        batch_size));
  }
#endif
}

template <>
void GemmEx<float, HIPContext>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const int lda,
    const float* B,
    const int ldb,
    const float beta,
    float* C,
    const int ldc,
    HIPContext* context) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  rocblas_operation cuTransA = (TransA == CblasNoTrans)
      ? rocblas_operation_none
      : rocblas_operation_transpose;
  rocblas_operation cuTransB = (TransB == CblasNoTrans)
      ? rocblas_operation_none
      : rocblas_operation_transpose;
  ROCBLAS_ENFORCE(rocblas_sgemm(
      context->rocblas_handle(),
      cuTransB,
      cuTransA,
      N,
      M,
      K,
      &alpha,
      B,
      ldb,
      A,
      lda,
      &beta,
      C,
      ldc));
}

template <>
void Gemv<float, HIPContext>(
    const CBLAS_TRANSPOSE TransA,
    const int M,
    const int N,
    const float alpha,
    const float* A,
    const float* x,
    const float beta,
    float* y,
    HIPContext* context,
    TensorProto::DataType math_type) {
  rocblas_operation cuTransA = (TransA == CblasNoTrans)
      ? rocblas_operation_transpose
      : rocblas_operation_none;
  ROCBLAS_ENFORCE(rocblas_sgemv(
      context->rocblas_handle(),
      cuTransA,
      N,
      M,
      &alpha,
      A,
      N,
      x,
      1,
      &beta,
      y,
      1));
}

// Batched Add variants
namespace {

template <typename T>
__global__ void AddStripedBatchKernel(
    const int N,
    const T* first,
    T* Y,
    const int stripe,
    const int batch) {
  for (int j = 0; j < batch; j++) {
    const T* x = first + j * stripe;
    HIP_1D_KERNEL_LOOP(i, N) {
      float tmpY = convert::To<T, float>(Y[i]);
      tmpY += convert::To<T, float>(x[i]);
      Y[i] = convert::To<float, T>(tmpY);
    }
  }
}
} // namespace

#define CAFFE2_SPECIALIZED_HIP_ADD_STRIPED_BATCH(T) \
  template <>                                       \
  void AddStripedBatch<T, HIPContext>(              \
      const int N,                                  \
      const T* first,                               \
      T* Y,                                         \
      const int stripe,                             \
      const int batch,                              \
      HIPContext* context) {                        \
    hipLaunchKernelGGL(                             \
        AddStripedBatchKernel<T>,                   \
        CAFFE_GET_BLOCKS(N),                        \
        CAFFE_HIP_NUM_THREADS,                      \
        0,                                          \
        context->hip_stream(),                      \
        N,                                          \
        first,                                      \
        Y,                                          \
        stripe,                                     \
        batch);                                     \
  }

CAFFE2_SPECIALIZED_HIP_ADD_STRIPED_BATCH(float);
CAFFE2_SPECIALIZED_HIP_ADD_STRIPED_BATCH(float16);
#undef CAFFE2_SPECIALIZED_HIP_ADD_STRIPED_BATCH

template <>
void Gemv<float16, HIPContext>(
    const CBLAS_TRANSPOSE TransA,
    const int M,
    const int N,
    const float alpha,
    const float16* A,
    const float16* x,
    const float beta,
    float16* y,
    HIPContext* context,
    TensorProto::DataType math_type) {
  CAFFE_THROW("Unsupported math type");
#if ROCBLAS_FP16 // rocblas does not support fp16 yet
  rocblas_operation cuTransA = (TransA == CblasNoTrans)
      ? rocblas_operation_transpose
      : rocblas_operation_none;

  // sort out what we need to call cublasSgemmEx / cublasHgemm
  int m = (cuTransA == rocblas_operation_none) ? N : M;
  int k = (cuTransA == rocblas_operation_none) ? M : N;
  int LDA = (cuTransA == rocblas_operation_none) ? m : k;
  int LDC = m;

  if (math_type == TensorProto_DataType_FLOAT) {
    ROCBLAS_CHECK(cublasSgemmEx(
        context->rocblas_handle(),
        cuTransA,
        rocblas_operation_none,
        m,
        1,
        k,
        &alpha,
        A,
        CUDA_R_16F,
        LDA,
        x,
        CUDA_R_16F,
        k,
        &beta,
        y,
        CUDA_R_16F,
        LDC));
  } else if (math_type == TensorProto_DataType_FLOAT16) {
    auto alpha_fp16 = convert::floatToHalf(alpha);
    auto beta_fp16 = convert::floatToHalf(beta);

    ROCBLAS_CHECK(cublasHgemm(
        context->rocblas_handle(),
        cuTransA,
        rocblas_operation_none,
        m,
        1,
        k,
        &alpha_fp16,
        (const __half*)A,
        LDA,
        (const __half*)x,
        k,
        &beta_fp16,
        (__half*)y,
        LDC));
  } else {
    // fail
    CAFFE_THROW("Unsupported math type");
  }
#endif
}

namespace {
template <typename T>
__global__ void SetKernel(const int N, const T alpha, T* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] = alpha;
  }
}
} // namespace

#define CAFFE2_SPECIALIZED_HIP_SET(T)                             \
  template <>                                                     \
  void Set<T, HIPContext>(                                        \
      const size_t N, const T alpha, T* Y, HIPContext* context) { \
    if (N == 0) {                                                 \
      return;                                                     \
    }                                                             \
    hipLaunchKernelGGL(                                           \
        (SetKernel),                                              \
        CAFFE_GET_BLOCKS(N),                                      \
        CAFFE_HIP_NUM_THREADS,                                    \
        0,                                                        \
        context->hip_stream(),                                    \
        static_cast<const int>(N),                                \
        alpha,                                                    \
        Y);                                                       \
  }

CAFFE2_SPECIALIZED_HIP_SET(float);
CAFFE2_SPECIALIZED_HIP_SET(double);
CAFFE2_SPECIALIZED_HIP_SET(bool);
CAFFE2_SPECIALIZED_HIP_SET(int8_t);
CAFFE2_SPECIALIZED_HIP_SET(int16_t);
CAFFE2_SPECIALIZED_HIP_SET(float16);
CAFFE2_SPECIALIZED_HIP_SET(int);
CAFFE2_SPECIALIZED_HIP_SET(int64_t);
CAFFE2_SPECIALIZED_HIP_SET(char);
CAFFE2_SPECIALIZED_HIP_SET(uint8_t);
CAFFE2_SPECIALIZED_HIP_SET(uint16_t);
#undef CAFFE2_SPECIALIZED_HIP_SET

namespace {
template <typename T>
__global__ void
UniformShift(const size_t N, const float min, const float max, T* x) {
  float scale = max - min;
  HIP_1D_KERNEL_LOOP(i, N) {
    x[i] = convert::To<float, T>(convert::To<T, float>(x[i]) * scale + min);
  }
}

__global__ void
UniformIntFit(const size_t N, const int min, const int max, unsigned int* x) {
  int* x_int = reinterpret_cast<int*>(x);
  int range = (max - min + 1);
  HIP_1D_KERNEL_LOOP(i, N) {
    x_int[i] = min + static_cast<int>(x[i] % range);
  }
}
} // namespace

template <>
void RandUniform<float, HIPContext>(
    const size_t n,
    const float min,
    const float max,
    float* r,
    HIPContext* context) {
  HIPRAND_ENFORCE(hiprandGenerateUniform(context->hiprand_generator(), r, n));
  hipLaunchKernelGGL(
      (UniformShift<float>),
      dim3(CAFFE_GET_BLOCKS(n)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      n,
      min,
      max,
      r);
}

template <>
void RandUniform<double, HIPContext>(
    const size_t n,
    const double min,
    const double max,
    double* r,
    HIPContext* context) {
  HIPRAND_ENFORCE(
      hiprandGenerateUniformDouble(context->hiprand_generator(), r, n));
  hipLaunchKernelGGL(
      (UniformShift<double>),
      dim3(CAFFE_GET_BLOCKS(n)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      n,
      min,
      max,
      r);
}

template <>
void RandUniform<int, HIPContext>(
    const size_t n,
    const int min,
    const int max,
    int* r,
    HIPContext* context) {
  HIPRAND_ENFORCE(hiprandGenerate(
      context->hiprand_generator(), reinterpret_cast<unsigned int*>(r), n));
  hipLaunchKernelGGL(
      (UniformIntFit),
      dim3(CAFFE_GET_BLOCKS(n)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      n,
      min,
      max,
      reinterpret_cast<unsigned int*>(r));
}

template <typename T>
size_t HandleOddLengthRandGaussian(
    const size_t n,
    const T mean,
    const T std,
    T* r,
    HIPContext* context) {
  if (n % 2 == 1) {
    std::default_random_engine generator;
    std::normal_distribution<T> distribution(mean, std);
    const T random_value = distribution(generator);
    math::Set<T, HIPContext>(1, random_value, r + (n - 1), context);
    return n - 1;
  }
  return n;
}

template <>
void RandGaussian<float, HIPContext>(
    const size_t n,
    const float mean,
    const float std,
    float* r,
    HIPContext* context) {
  // If n is odd, we add a random Gaussian value at the end manually
  // and generate n-1 random values using curandGenerateNormal.
  // curandGenerateNormal requires n to be even.
  const size_t even_n =
      HandleOddLengthRandGaussian<float>(n, mean, std, r, context);
  HIPRAND_ENFORCE(hiprandGenerateNormal(
      context->hiprand_generator(), r, even_n, mean, std));
}

template <>
void RandGaussian<double, HIPContext>(
    const size_t n,
    const double mean,
    const double std,
    double* r,
    HIPContext* context) {
  const size_t even_n =
      HandleOddLengthRandGaussian<double>(n, mean, std, r, context);
  HIPRAND_ENFORCE(hiprandGenerateNormalDouble(
      context->hiprand_generator(), r, even_n, mean, std));
}

template <>
void Dot<float, HIPContext>(
    const int n,
    const float* a,
    const float* b,
    float* y,
    HIPContext* context) {
  float result;
  ROCBLAS_ENFORCE(
      rocblas_sdot(context->rocblas_handle(), n, a, 1, b, 1, &result));
  context->CopyFromCPU<float>(1, &result, y);
}

template <>
void Dot<float16, HIPContext>(
    const int n,
    const float16* a,
    const float16* b,
    float16* y,
    HIPContext* context) {
  CAFFE_THROW("Unsupported math type");
#if ROCBLAS_FP16 // rocblas does not support fp16 yet
  float16 result;
  // execute with 32-bit math
  ROCBLAS_CHECK(cublasDotEx(
      context->rocblas_handle(),
      n,
      a,
      CUDA_R_16F,
      1,
      b,
      CUDA_R_16F,
      1,
      &result,
      CUDA_R_16F,
      CUDA_R_32F));
  context->Copy<float16, CPUContext, HIPContext>(1, &result, y);
#endif
}

// A previous version of caffe2 used Thrust but it turns out that thrust
// reduction has an implicit scratch space allocation and deallocation, which
// may interfere with NCCL and create a deadlock. Hence we are using a custom
// reduction here.
#define SUM_KERNEL_NTHREADS 128
template <typename T>
__global__ void SumKernel(const int N, const T* X, T* Y, bool square) {
  const int idx = threadIdx.x;
  __shared__ float reduction_buffer[SUM_KERNEL_NTHREADS];

  reduction_buffer[idx] = 0;

  // A multilevel reduction.
  // N -> 128
  if (!square) {
    for (int i = idx; i < N; i += SUM_KERNEL_NTHREADS) {
      reduction_buffer[idx] += convert::To<T, float>(X[i]);
    }
  } else {
    for (int i = idx; i < N; i += SUM_KERNEL_NTHREADS) {
      float Xi = convert::To<T, float>(X[i]);
      reduction_buffer[idx] += Xi * Xi;
    }
  }
  __syncthreads();
  // 128 -> 32
  if (idx < 32) {
    reduction_buffer[idx] += reduction_buffer[idx + 32] +
        reduction_buffer[idx + 64] + reduction_buffer[idx + 96];
  }
  __syncthreads();
  // 32 -> 1
  if (idx == 0) {
    float tmp = 0;
    for (int i = 0; i < 32; ++i) {
      tmp += reduction_buffer[i];
    }
    *Y = convert::To<float, T>(tmp);
  }
}

// According to the benchmarks script
// caffe2/caffe2/experiments/python/device_reduce_sum_bench.py,
// device reduce is slower for N <= 10000.
#define DEVICE_REDUCE_SIZE_THRESHOLD 10000

namespace {

template <typename T>
__global__ void SumConvertKernel(float* sum, T* dest) {
  *dest = convert::To<float, T>(*sum);
}

template <typename T, typename IterT>
void SumGenericIter(
    const int N,
    IterT it,
    T*& dest,
    HIPContext* context,
    Tensor* scratch_ptr) {
  size_t memRequired = 0;
  cub::DeviceReduce::Sum(
      nullptr, memRequired, it, dest, N, context->hip_stream());
  auto buffer_size =
      static_cast<TIndex>((memRequired + sizeof(T) - 1) / sizeof(T));
  if (!dest) {
    // allocate one more T at the end of scratch for dest
    scratch_ptr->Resize(std::vector<TIndex>{buffer_size + 1});
    dest = scratch_ptr->template mutable_data<T>() + buffer_size;
  } else {
    scratch_ptr->Resize(std::vector<TIndex>{buffer_size});
  }
  cub::DeviceReduce::Sum(
      static_cast<void*>(scratch_ptr->template mutable_data<T>()),
      memRequired,
      it,
      dest,
      N,
      context->hip_stream());
}
} // namespace

template <>
void Sum<float, HIPContext>(
    const int N,
    const float* x,
    float* y,
    HIPContext* context,
    Tensor* scratch_ptr) {
  if (scratch_ptr && N > DEVICE_REDUCE_SIZE_THRESHOLD) {
    SumGenericIter<float>(N, x, y, context, scratch_ptr);
  } else {
    hipLaunchKernelGGL(
        (SumKernel),
        dim3(1),
        dim3(SUM_KERNEL_NTHREADS),
        0,
        context->hip_stream(),
        N,
        x,
        y,
        false);
  }
}

template <>
void Sum<int32_t, HIPContext>(
    const int N,
    const int32_t* x,
    int32_t* y,
    HIPContext* context,
    Tensor* scratch_ptr) {
  if (scratch_ptr && N > DEVICE_REDUCE_SIZE_THRESHOLD) {
    SumGenericIter<int32_t>(N, x, y, context, scratch_ptr);
  } else {
    hipLaunchKernelGGL(
        (SumKernel),
        dim3(1),
        dim3(SUM_KERNEL_NTHREADS),
        0,
        context->hip_stream(),
        N,
        x,
        y,
        false);
  }
}

namespace {
template <typename T>
struct FloatTransform {
  inline __host__ __device__ float operator()(const T v) const {
    return convert::To<T, float>(v);
  }
};
} // namespace

#define CAFFE2_MATH_SUM_FUNC(T)                                           \
  template <>                                                             \
  void Sum<T, HIPContext>(                                                \
      const int N,                                                        \
      const T* x,                                                         \
      T* y,                                                               \
      HIPContext* context,                                                \
      Tensor* scratch_ptr) {                                              \
    if (scratch_ptr && N > DEVICE_REDUCE_SIZE_THRESHOLD) {                \
      FloatTransform<T> transform;                                        \
      cub::TransformInputIterator<float, FloatTransform<T>, const T*> it( \
          x, transform);                                                  \
      float* sum = nullptr;                                               \
      SumGenericIter<float>(N, it, sum, context, scratch_ptr);            \
      hipLaunchKernelGGL(                                                 \
          (SumConvertKernel),                                             \
          dim3(1),                                                        \
          dim3(1),                                                        \
          0,                                                              \
          context->hip_stream(),                                          \
          sum,                                                            \
          y);                                                             \
    } else {                                                              \
      hipLaunchKernelGGL(                                                 \
          (SumKernel),                                                    \
          dim3(1),                                                        \
          dim3(SUM_KERNEL_NTHREADS),                                      \
          0,                                                              \
          context->hip_stream(),                                          \
          N,                                                              \
          x,                                                              \
          y,                                                              \
          false);                                                         \
    }                                                                     \
  }

CAFFE2_MATH_SUM_FUNC(float16)
#undef CAFFE2_MATH_SUM_FUNC

namespace {
template <typename T>
struct SqrTransform {
  inline __host__ __device__ T operator()(const T v) const {
    return v * v;
  }
};
} //  namespace

template <>
void SumSqr<float, HIPContext>(
    const int N,
    const float* x,
    float* y,
    HIPContext* context,
    Tensor* scratch_ptr) {
  if (scratch_ptr && N > DEVICE_REDUCE_SIZE_THRESHOLD) {
    SqrTransform<float> transform;
    cub::TransformInputIterator<float, SqrTransform<float>, const float*> it(
        x, transform);
    SumGenericIter<float>(N, it, y, context, scratch_ptr);
  } else {
    hipLaunchKernelGGL(
        (SumKernel),
        dim3(1),
        dim3(SUM_KERNEL_NTHREADS),
        0,
        context->hip_stream(),
        N,
        x,
        y,
        true);
  }
}

#define CAFFE2_MATH_SUMSQR_FUNC(T)                                    \
  template <>                                                         \
  void SumSqr<T, HIPContext>(                                         \
      const int N,                                                    \
      const T* x,                                                     \
      T* y,                                                           \
      HIPContext* context,                                            \
      Tensor* scratch_ptr) {                                          \
    if (scratch_ptr && N > DEVICE_REDUCE_SIZE_THRESHOLD) {            \
      FloatTransform<T> float_transform;                              \
      cub::TransformInputIterator<float, FloatTransform<T>, const T*> \
          float_it(x, float_transform);                               \
      SqrTransform<float> sqr_transform;                              \
      cub::TransformInputIterator<                                    \
          float,                                                      \
          SqrTransform<float>,                                        \
          decltype(float_it)>                                         \
          it(float_it, sqr_transform);                                \
      float* sum = nullptr;                                           \
      SumGenericIter<float>(N, it, sum, context, scratch_ptr);        \
      hipLaunchKernelGGL(                                             \
          (SumConvertKernel),                                         \
          dim3(1),                                                    \
          dim3(1),                                                    \
          0,                                                          \
          context->hip_stream(),                                      \
          sum,                                                        \
          y);                                                         \
    } else {                                                          \
      hipLaunchKernelGGL(                                             \
          (SumKernel),                                                \
          dim3(1),                                                    \
          dim3(SUM_KERNEL_NTHREADS),                                  \
          0,                                                          \
          context->hip_stream(),                                      \
          N,                                                          \
          x,                                                          \
          y,                                                          \
          true);                                                      \
    }                                                                 \
  }

CAFFE2_MATH_SUMSQR_FUNC(float16)
#undef CAFFE2_MATH_SUMSQR_FUNC
#undef DEVICE_REDUCE_SIZE_THRESHOLD

namespace {
template <typename T>
__global__ void
SelectKernel(const int N, const int D, const T* x, const int* idx, T* y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    y[i] = x[i * D + idx[i]];
  }
}
} // namespace

template <>
void Select<float, HIPContext>(
    const int N,
    const int D,
    const float* x,
    const int* idx,
    float* y,
    HIPContext* context) {
  hipLaunchKernelGGL(
      (SelectKernel<float>),
      dim3(CAFFE_GET_BLOCKS(N)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      N,
      D,
      x,
      idx,
      y);
}

template <>
void Select<float16, HIPContext>(
    const int N,
    const int D,
    const float16* x,
    const int* idx,
    float16* y,
    HIPContext* context) {
  hipLaunchKernelGGL(
      (SelectKernel<float16>),
      dim3(CAFFE_GET_BLOCKS(N)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      N,
      D,
      x,
      idx,
      y);
}

namespace {

template <typename TAlpha, typename TData>
__global__ void
ScaleKernel(const int n, const TAlpha alpha, const TData* x, TData* y) {
  HIP_1D_KERNEL_LOOP(i, n) {
    y[i] = x[i] * static_cast<TData>(alpha);
  }
}

template <typename TAlpha, typename TData>
__global__ void
ScaleKernel(const int n, const TAlpha* alpha, const TData* x, TData* y) {
  HIP_1D_KERNEL_LOOP(i, n) {
    y[i] = x[i] * static_cast<TData>(*alpha);
  }
}

template <>
__global__ void ScaleKernel<float16, float16>(
    const int n,
    const float16 alpha,
    const float16* x,
    float16* y) {
  HIP_1D_KERNEL_LOOP(i, n) {
    y[i] = convert::To<float, float16>(
        convert::To<float16, float>(x[i]) * convert::To<float16, float>(alpha));
  }
}

template <>
__global__ void ScaleKernel<float16, float16>(
    const int n,
    const float16* alpha,
    const float16* x,
    float16* y) {
  HIP_1D_KERNEL_LOOP(i, n) {
    y[i] = convert::To<float, float16>(
        convert::To<float16, float>(x[i]) *
        convert::To<float16, float>(*alpha));
  }
}

// fp16 specialization
template <>
__global__ void ScaleKernel<float, float16>(
    const int n,
    const float alpha,
    const float16* x,
    float16* y) {
  HIP_1D_KERNEL_LOOP(i, n) {
    y[i] =
        convert::To<float, float16>(convert::To<float16, float>(x[i]) * alpha);
  }
}

template <>
__global__ void ScaleKernel<float, float16>(
    const int n,
    const float* alpha,
    const float16* x,
    float16* y) {
  HIP_1D_KERNEL_LOOP(i, n) {
    y[i] = convert::To<float, float16>(
        convert::To<float16, float>(x[i]) * (*alpha));
  }
}

template <typename T>
__global__ void PowKernel(const int n, const T* x, const T exponent, T* y) {
  HIP_1D_KERNEL_LOOP(i, n) {
    y[i] = powf(x[i], exponent);
  }
}

} // namespace

template <>
void Powx<float, HIPContext>(
    const int N,
    const float* a,
    const float b,
    float* y,
    HIPContext* context) {
  hipLaunchKernelGGL(
      (PowKernel),
      dim3(CAFFE_GET_BLOCKS(N)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      N,
      a,
      b,
      y);
}

#define CAFFE2_SPECIALIZED_HIP_SCALE(TAlpha, TData) \
  template <>                                       \
  void Scale<TAlpha, TData, HIPContext>(            \
      const int n,                                  \
      const TAlpha alpha,                           \
      const TData* x,                               \
      TData* y,                                     \
      HIPContext* context) {                        \
    hipLaunchKernelGGL(                             \
        (ScaleKernel<TAlpha, TData>),               \
        dim3(CAFFE_GET_BLOCKS(n)),                  \
        dim3(CAFFE_HIP_NUM_THREADS),                \
        0,                                          \
        context->hip_stream(),                      \
        n,                                          \
        alpha,                                      \
        x,                                          \
        y);                                         \
  }                                                 \
  template <>                                       \
  void Scale<TAlpha, TData, HIPContext>(            \
      const int n,                                  \
      const TAlpha* alpha,                          \
      const TData* x,                               \
      TData* y,                                     \
      HIPContext* context) {                        \
    hipLaunchKernelGGL(                             \
        (ScaleKernel<TAlpha, TData>),               \
        dim3(CAFFE_GET_BLOCKS(n)),                  \
        dim3(CAFFE_HIP_NUM_THREADS),                \
        0,                                          \
        context->hip_stream(),                      \
        n,                                          \
        alpha,                                      \
        x,                                          \
        y);                                         \
  }
CAFFE2_SPECIALIZED_HIP_SCALE(float, float)
CAFFE2_SPECIALIZED_HIP_SCALE(float16, float16)
CAFFE2_SPECIALIZED_HIP_SCALE(float, float16)
CAFFE2_SPECIALIZED_HIP_SCALE(double, double)
CAFFE2_SPECIALIZED_HIP_SCALE(std::int32_t, std::int32_t)
CAFFE2_SPECIALIZED_HIP_SCALE(std::int64_t, std::int64_t)
#undef CAFFE2_SPECIALIZED_HIP_SCALE

template <>
void Axpy<float, HIPContext>(
    const int N,
    const float alpha,
    const float* X,
    float* Y,
    HIPContext* context) {
  ROCBLAS_ENFORCE(
      rocblas_saxpy(context->rocblas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void Axpy<double, HIPContext>(
    const int N,
    const float alpha,
    const double* X,
    double* Y,
    HIPContext* context) {
  double alpha_d{alpha};
  ROCBLAS_ENFORCE(
      rocblas_daxpy(context->rocblas_handle(), N, &alpha_d, X, 1, Y, 1));
}

template <>
void Axpy<float16, HIPContext>(
    const int N,
    const float alpha,
    const float16* X,
    float16* Y,
    HIPContext* context) {
  CAFFE_THROW("Unsupported math type");
#if ROCBLAS_FP16
  ROCBLAS_CHECK(cublasAxpyEx(
      context->rocblas_handle(),
      N,
      &alpha,
      CUDA_R_16F,
      X,
      CUDA_R_16F,
      1,
      Y,
      CUDA_R_16F,
      1,
      CUDA_R_32F));
#endif
}

namespace {
template <typename T>
__global__ void AxpyKernel(const int n, const float* a, const T* x, T* y) {
  HIP_1D_KERNEL_LOOP(index, n) {
    y[index] = convert::Get<T>(
        convert::Get<float>(x[index]) * (*a) + convert::Get<float>(y[index]));
  }
}
} // namespace

template <>
void Axpy<float, HIPContext>(
    const int n,
    const float* alpha,
    const float* X,
    float* Y,
    HIPContext* context) {
  hipLaunchKernelGGL(
      (AxpyKernel<float>),
      dim3(CAFFE_GET_BLOCKS(n)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      n,
      alpha,
      X,
      Y);
}

template <>
void Axpy<float16, HIPContext>(
    const int n,
    const float* alpha,
    const float16* X,
    float16* Y,
    HIPContext* context) {
  hipLaunchKernelGGL(
      (AxpyKernel<float16>),
      dim3(CAFFE_GET_BLOCKS(n)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      n,
      alpha,
      X,
      Y);
}

namespace {
template <typename T>
__global__ void
AxpbyKernel(const int n, const T a, const T* x, const T b, T* y) {
  HIP_1D_KERNEL_LOOP(index, n) {
    y[index] = x[index] * a + y[index] * b;
  }
}
} // namespace

template <>
void Axpby<float, HIPContext>(
    const int n,
    const float a,
    const float* x,
    const float b,
    float* y,
    HIPContext* context) {
  hipLaunchKernelGGL(
      (AxpbyKernel<float>),
      dim3(CAFFE_GET_BLOCKS(n)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      n,
      a,
      x,
      b,
      y);
}

namespace {

template <typename T>
__global__ void Im2ColNCHWHIPKernel(
    const int n,
    const int input_h,
    const int input_w,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int stride_h,
    const int stride_w,
    const int output_h,
    const int output_w,
    const T* img_data,
    T* col_data) {
  HIP_1D_KERNEL_LOOP(index, n) {
    const int w_out = index % output_w;
    const int h_index = index / output_w;
    const int h_out = h_index % output_h;
    const int channel_in = h_index / output_h;
    const int channel_out = channel_in * kernel_h * kernel_w;
    const int h_in = h_out * stride_h - pad_t;
    const int w_in = w_out * stride_w - pad_l;
    const int output_size = output_h * output_w;
    T* col_data_ptr =
        col_data + (channel_out * output_h + h_out) * output_w + w_out;
    const T* img_data_ptr =
        img_data + (channel_in * input_h + h_in) * input_w + w_in;
    int dh = 0;
    for (int i = 0; i < kernel_h; ++i) {
      int dw = 0;
      for (int j = 0; j < kernel_w; ++j) {
        const int h = h_in + dh;
        const int w = w_in + dw;
        *col_data_ptr = (h >= 0 && w >= 0 && h < input_h && w < input_w)
            ? __ldg(img_data_ptr + dh * input_w + dw)
            : 0;
        col_data_ptr += output_size;
        dw += dilation_w;
      }
      dh += dilation_h;
    }
  }
}

template <typename T>
__global__ void Im2ColNHWCHIPKernel(
    const int n,
    const int input_h,
    const int input_w,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int stride_h,
    const int stride_w,
    const int output_w,
    const int channels,
    const T* img_data,
    T* col_data) {
  HIP_1D_KERNEL_LOOP(index, n) {
    const int channel_in = index % channels;
    const int w_out = index / channels % output_w;
    const int h_out = index / channels / output_w;
    const int h_in = h_out * stride_h - pad_t;
    const int w_in = w_out * stride_w - pad_l;
    T* col_data_ptr = col_data +
        (h_out * output_w + w_out) * channels * kernel_h * kernel_w +
        channel_in;
    int dh = 0;
    for (int i = 0; i < kernel_h; ++i) {
      int dw = 0;
      for (int j = 0; j < kernel_w; ++j) {
        const int h = h_in + dh;
        const int w = w_in + dw;
        *col_data_ptr = (h >= 0 && w >= 0 && h < input_h && w < input_w)
            ? __ldg(img_data + (h * input_w + w) * channels + channel_in)
            : 0;
        col_data_ptr += channels;
        dw += dilation_w;
      }
      dh += dilation_h;
    }
  }
}

template <typename T>
__global__ void Col2ImNCHWHIPKernel(
    const int n,
    const int input_h,
    const int input_w,
    const int patch_h,
    const int patch_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int stride_h,
    const int stride_w,
    const int output_h,
    const int output_w,
    const T* col_data,
    T* img_data) {
  const int dpatch_h = dilation_h * (patch_h - 1) + 1;
  const int dpatch_w = dilation_w * (patch_w - 1) + 1;

  HIP_1D_KERNEL_LOOP(index, n) {
    T val = 0;
    const int w = index % input_w + pad_l;
    const int h = index / input_w % input_h + pad_t;
    const int c = index / (input_h * input_w);

    // compute the start and end of the output
    const int w_col_start = (w < dpatch_w) ? 0 : (w - dpatch_w) / stride_w + 1;
    const int w_col_end = min(w / stride_w + 1, output_w);
    const int h_col_start = (h < dpatch_h) ? 0 : (h - dpatch_h) / stride_h + 1;
    const int h_col_end = min(h / stride_h + 1, output_h);

    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        int h_k = (h - h_col * stride_h);
        int w_k = (w - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          const int col_data_index =
              (((c * patch_h + h_k) * patch_w + w_k) * output_h + h_col) *
                  output_w +
              w_col;
          val += __ldg(col_data + col_data_index);
        }
      }
    }
    img_data[index] = val;
  }
}

template <typename T>
__global__ void Col2ImNHWCHIPKernel(
    const int n,
    const int input_w,
    const int channels,
    const int patch_h,
    const int patch_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int stride_h,
    const int stride_w,
    const int output_h,
    const int output_w,
    const T* col_data,
    T* img_data) {
  const int dpatch_h = dilation_h * (patch_h - 1) + 1;
  const int dpatch_w = dilation_w * (patch_w - 1) + 1;

  HIP_1D_KERNEL_LOOP(index, n) {
    T val = 0;
    const int c = index % channels;
    const int w = index / channels % input_w + pad_l;
    const int h = index / channels / input_w + pad_t;
    // compute the start and end of the output
    const int w_col_start = (w < dpatch_w) ? 0 : (w - dpatch_w) / stride_w + 1;
    const int w_col_end = min(w / stride_w + 1, output_w);
    const int h_col_start = (h < dpatch_h) ? 0 : (h - dpatch_h) / stride_h + 1;
    const int h_col_end = min(h / stride_h + 1, output_h);
    const int channels_col = patch_h * patch_w * channels;

    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        int h_k = h - h_col * stride_h;
        int w_k = w - w_col * stride_w;
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          const int c_col = (h_k * patch_w + w_k) * channels + c;
          val += __ldg(
              col_data + (h_col * output_w + w_col) * channels_col + c_col);
        }
      }
    }
    img_data[index] = val;
  }
}

template <typename T, int N, bool kCol2Im>
__global__ void Im2ColNdNCHWHIPKernel(
    const int outer_size,
    const int inner_size,
    const int kernel_size,
    SimpleArray<int, N + 1> img_shape,
    SimpleArray<int, N + 1> col_shape,
    SimpleArray<int, N> kernel_shape,
    SimpleArray<int, N> stride,
    SimpleArray<int, N> dilation,
    SimpleArray<int, N> pad,
    const T* X_data,
    T* Y_data) {
  int d_offset[N];
  int d_iter[N];
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    int offset_i = i;
#pragma unroll
    for (int d_i = N - 1; d_i >= 0; --d_i) {
      d_offset[d_i] = offset_i % kernel_shape.data[d_i];
      offset_i /= kernel_shape.data[d_i];
    }
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      int offset_j = j;
#pragma unroll
      for (int d_i = N - 1; d_i >= 0; --d_i) {
        d_iter[d_i] = offset_j % col_shape.data[d_i + 1];
        offset_j /= col_shape.data[d_i + 1];
      }
      const int col_index = i * inner_size + j;
      int img_index = i / kernel_size;
      bool is_padding = false;
#pragma unroll
      for (int d_i = 0; d_i < N; ++d_i) {
        const int d_img = d_iter[d_i] * stride.data[d_i] - pad.data[d_i] +
            d_offset[d_i] * dilation.data[d_i];
        is_padding |= d_img < 0 || d_img >= img_shape.data[d_i + 1];
        img_index = img_index * img_shape.data[d_i + 1] + d_img;
      }
      if (!kCol2Im) {
        Y_data[col_index] = is_padding ? 0 : __ldg(X_data + img_index);
      } else if (!is_padding) {
        atomicAdd(Y_data + img_index, __ldg(X_data + col_index));
      }
    }
  }
}

template <typename T, int N>
void Im2ColNdNCHWHIPImpl(
    const int img_size,
    const int col_size,
    const int* img_shape,
    const int* col_shape,
    const int* kernel_shape,
    const int* stride,
    const int* dilation,
    const int* pad,
    const float* img_data,
    float* col_data,
    HIPContext* context) {
  const int outer_size = col_shape[0];
  const int inner_size = col_size / outer_size;
  const int kernel_size = std::accumulate(
      kernel_shape, kernel_shape + N, 1, std::multiplies<int>());
  SimpleArray<int, N + 1> img_shape_array;
  SimpleArray<int, N + 1> col_shape_array;
  SimpleArray<int, N> kernel_shape_array;
  SimpleArray<int, N> stride_array;
  SimpleArray<int, N> dilation_array;
  SimpleArray<int, N> pad_array;
  std::memcpy(img_shape_array.data, img_shape, (N + 1) * sizeof(int));
  std::memcpy(col_shape_array.data, col_shape, (N + 1) * sizeof(int));
  std::memcpy(kernel_shape_array.data, kernel_shape, N * sizeof(int));
  std::memcpy(stride_array.data, stride, N * sizeof(int));
  std::memcpy(dilation_array.data, dilation, N * sizeof(int));
  std::memcpy(pad_array.data, pad, N * sizeof(int));
  hipLaunchKernelGGL(
      (Im2ColNdNCHWHIPKernel<T, N, false>),
      dim3(std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      outer_size,
      inner_size,
      kernel_size,
      img_shape_array,
      col_shape_array,
      kernel_shape_array,
      stride_array,
      dilation_array,
      pad_array,
      img_data,
      col_data);
}

template <typename T, int N>
void Col2ImNdNCHWHIPImpl(
    const int img_size,
    const int col_size,
    const int* img_shape,
    const int* col_shape,
    const int* kernel_shape,
    const int* stride,
    const int* dilation,
    const int* pad,
    const float* col_data,
    float* img_data,
    HIPContext* context) {
  const int outer_size = col_shape[0];
  const int inner_size = col_size / outer_size;
  const int kernel_size = std::accumulate(
      kernel_shape, kernel_shape + N, 1, std::multiplies<int>());
  SimpleArray<int, N + 1> img_shape_array;
  SimpleArray<int, N + 1> col_shape_array;
  SimpleArray<int, N> kernel_shape_array;
  SimpleArray<int, N> stride_array;
  SimpleArray<int, N> dilation_array;
  SimpleArray<int, N> pad_array;
  std::memcpy(img_shape_array.data, img_shape, (N + 1) * sizeof(int));
  std::memcpy(col_shape_array.data, col_shape, (N + 1) * sizeof(int));
  std::memcpy(kernel_shape_array.data, kernel_shape, N * sizeof(int));
  std::memcpy(stride_array.data, stride, N * sizeof(int));
  std::memcpy(dilation_array.data, dilation, N * sizeof(int));
  std::memcpy(pad_array.data, pad, N * sizeof(int));
  Set<T, HIPContext>(img_size, 0, img_data, context);
  hipLaunchKernelGGL(
      (Im2ColNdNCHWHIPKernel<T, N, true>),
      dim3(std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      outer_size,
      inner_size,
      kernel_size,
      img_shape_array,
      col_shape_array,
      kernel_shape_array,
      stride_array,
      dilation_array,
      pad_array,
      col_data,
      img_data);
}

} // namespace

template <>
void Im2Col<float, HIPContext, StorageOrder::NCHW>(
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    const float* img_data,
    float* col_data,
    HIPContext* context,
    const int /* groups */) {
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
  const int output_h = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  const int output_w = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
  const int num_kernels = channels * output_h * output_w;
  hipLaunchKernelGGL(
      (Im2ColNCHWHIPKernel<float>),
      dim3(CAFFE_GET_BLOCKS(num_kernels)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      num_kernels,
      height,
      width,
      kernel_h,
      kernel_w,
      dilation_h,
      dilation_w,
      pad_t,
      pad_l,
      stride_h,
      stride_w,
      output_h,
      output_w,
      img_data,
      col_data);
}

template <>
void Im2Col<float, HIPContext, StorageOrder::NHWC>(
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    const float* img_data,
    float* col_data,
    HIPContext* context,
    const int groups) {
  CAFFE_ENFORCE_EQ(groups, 1, "groups must be 1 for GPU NHWC Im2Col");

  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
  const int output_h = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  const int output_w = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
  const int num_kernels = output_h * output_w * channels;
  hipLaunchKernelGGL(
      (Im2ColNHWCHIPKernel<float>),
      dim3(CAFFE_GET_BLOCKS(num_kernels)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      num_kernels,
      height,
      width,
      kernel_h,
      kernel_w,
      dilation_h,
      dilation_w,
      pad_t,
      pad_l,
      stride_h,
      stride_w,
      output_w,
      channels,
      img_data,
      col_data);
}

template <>
void Col2Im<float, HIPContext, StorageOrder::NCHW>(
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    const float* col_data,
    float* img_data,
    HIPContext* context,
    const int /* groups */) {
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
  const int output_h = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  const int output_w = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
  const int num_kernels = channels * height * width;
  hipLaunchKernelGGL(
      (Col2ImNCHWHIPKernel<float>),
      dim3(CAFFE_GET_BLOCKS(num_kernels)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      num_kernels,
      height,
      width,
      kernel_h,
      kernel_w,
      dilation_h,
      dilation_w,
      pad_t,
      pad_l,
      stride_h,
      stride_w,
      output_h,
      output_w,
      col_data,
      img_data);
}

template <>
void Col2Im<float, HIPContext, StorageOrder::NHWC>(
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    const float* col_data,
    float* img_data,
    HIPContext* context,
    const int groups) {
  CAFFE_ENFORCE_EQ(groups, 1, "groups must be 1 for GPU NHWC Col2Im");

  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
  const int output_h = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  const int output_w = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
  const int num_kernels = height * width * channels;
  hipLaunchKernelGGL(
      (Col2ImNHWCHIPKernel<float>),
      dim3(CAFFE_GET_BLOCKS(num_kernels)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      num_kernels,
      width,
      channels,
      kernel_h,
      kernel_w,
      dilation_h,
      dilation_w,
      pad_t,
      pad_l,
      stride_h,
      stride_w,
      output_h,
      output_w,
      col_data,
      img_data);
}

template <>
void Im2ColNd<float, HIPContext, StorageOrder::NCHW>(
    const int N,
    const int img_size,
    const int col_size,
    const int* img_shape,
    const int* col_shape,
    const int* kernel_shape,
    const int* stride,
    const int* dilation,
    const int* pad,
    const float* img_data,
    float* col_data,
    HIPContext* context) {
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(
      N,
      Im2ColNdNCHWHIPImpl,
      float,
      img_size,
      col_size,
      img_shape,
      col_shape,
      kernel_shape,
      stride,
      dilation,
      pad,
      img_data,
      col_data,
      context);
}

template <>
void Col2ImNd<float, HIPContext, StorageOrder::NCHW>(
    const int N,
    const int img_size,
    const int col_size,
    const int* img_shape,
    const int* col_shape,
    const int* kernel_shape,
    const int* stride,
    const int* dilation,
    const int* pad,
    const float* col_data,
    float* img_data,
    HIPContext* context) {
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(
      N,
      Col2ImNdNCHWHIPImpl,
      float,
      img_size,
      col_size,
      img_shape,
      col_shape,
      kernel_shape,
      stride,
      dilation,
      pad,
      col_data,
      img_data,
      context);
}

template <>
void CopyMatrix<HIPContext>(
    const size_t itemsize,
    const int M,
    const int N,
    const void* A,
    const int lda,
    void* B,
    const int ldb,
    HIPContext* context,
    TypeMeta::TypedCopy copy) {
  CAFFE_ENFORCE(!copy, "Copy constructor is not supported in HIP context");
  hipMemcpy2DAsync(
      B,
      ldb * itemsize,
      A,
      lda * itemsize,
      N * itemsize,
      M,
      hipMemcpyDeviceToDevice,
      context->hip_stream());
}

template <>
void CopyVector<float, HIPContext>(
    const int N,
    const float* src,
    float* dst,
    HIPContext* context) {
  if (src != dst && N > 0) {
    hipMemcpyAsync(
        dst,
        src,
        sizeof(float) * N,
        hipMemcpyDeviceToDevice,
        context->hip_stream());
  }
}

namespace {

template <typename T>
using BlockReduce = cub::BlockReduce<T, CAFFE_HIP_NUM_THREADS>;

template <typename T, class Reducer>
__global__ void RowwiseReduceKernel(
    const int rows,
    const int cols,
    const Reducer reducer,
    const T init,
    const T alpha,
    const T* X,
    T* Y) {
  __shared__ typename BlockReduce<T>::TempStorage temp_storage;
  for (int i = blockIdx.x; i < rows; i += gridDim.x) {
    T val = init;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
      val = reducer(X[i * cols + j], val);
    }
    val = BlockReduce<T>(temp_storage).Reduce(val, reducer);
    if (threadIdx.x == 0) {
      Y[i] = val * alpha;
    }
    __syncthreads();
  }
}

template <typename T, class Reducer>
__global__ void ColwiseReduceKernel(
    const int rows,
    const int cols,
    const Reducer reducer,
    const T init,
    const T alpha,
    const T* X,
    T* Y) {
  __shared__ typename BlockReduce<T>::TempStorage temp_storage;
  for (int i = blockIdx.x; i < cols; i += gridDim.x) {
    T val = init;
    for (int j = threadIdx.x; j < rows; j += blockDim.x) {
      val = reducer(X[j * cols + i], val);
    }
    val = BlockReduce<T>(temp_storage).Reduce(val, reducer);
    if (threadIdx.x == 0) {
      Y[i] = val * alpha;
    }
    __syncthreads();
  }
}

} // namespace

#define CAFFE2_SPECIALIZED_HIP_ROWWISE_MAX(T)                            \
  template <>                                                            \
  void RowwiseMax<T, HIPContext>(                                        \
      const int N, const int D, const T* x, T* y, HIPContext* context) { \
    hipLaunchKernelGGL(                                                  \
        RowwiseReduceKernel,                                             \
        std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),                           \
        CAFFE_HIP_NUM_THREADS,                                           \
        0,                                                               \
        context->hip_stream(),                                           \
        N,                                                               \
        D,                                                               \
        cub::Max(),                                                      \
        std::numeric_limits<T>::lowest(),                                \
        T(1),                                                            \
        x,                                                               \
        y);                                                              \
  }
CAFFE2_SPECIALIZED_HIP_ROWWISE_MAX(float)
#undef CAFFE2_SPECIALIZED_HIP_ROWWISE_MAX

#define CAFFE2_SPECIALIZED_HIP_COLWISE_MAX(T)                            \
  template <>                                                            \
  void ColwiseMax<T, HIPContext>(                                        \
      const int N, const int D, const T* x, T* y, HIPContext* context) { \
    hipLaunchKernelGGL(                                                  \
        ColwiseReduceKernel,                                             \
        std::min(D, CAFFE_MAXIMUM_NUM_BLOCKS),                           \
        CAFFE_HIP_NUM_THREADS,                                           \
        0,                                                               \
        context->hip_stream(),                                           \
        N,                                                               \
        D,                                                               \
        cub::Max(),                                                      \
        std::numeric_limits<T>::lowest(),                                \
        T(1),                                                            \
        x,                                                               \
        y);                                                              \
  }
CAFFE2_SPECIALIZED_HIP_COLWISE_MAX(float)
#undef CAFFE2_SPECIALIZED_HIP_COLWISE_MAX

namespace {
__global__ void
maximum_kernel(const int N, const float alpha, const float* x, float* y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    y[i] = fmaxf(x[i], alpha);
  }
}
} // namespace

template <>
void Maximum(
    const int N,
    const float alpha,
    const float* x,
    float* y,
    HIPContext* context) {
  hipLaunchKernelGGL(
      (maximum_kernel),
      dim3(std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      N,
      alpha,
      x,
      y);
}

namespace {

template <typename T, class Reducer, int D>
__global__ void ReduceTensorHIPKernel(
    const int outer_size,
    const int inner_size,
    SimpleArray<int, D> X_strides,
    SimpleArray<FixedDivisor<int>, D> Y_dims,
    const Reducer reducer,
    const T init,
    const T alpha,
    const T* X,
    T* Y) {
  __shared__ typename BlockReduce<T>::TempStorage temp_storage;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T val = init;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      int X_index = 0;
      int Y_index = i * inner_size + j;
#pragma unroll
      for (int d = D - 1; d >= 0; --d) {
        int r;
        Y_dims.data[d].DivMod(Y_index, &Y_index, &r);
        X_index += r * X_strides.data[d];
      }
#if __HIP_ARCH__ >= 350
      val = reducer(val, __ldg(X + X_index));
#else
      val = reducer(val, X[X_index]);
#endif
    }
    val = BlockReduce<T>(temp_storage).Reduce(val, reducer);
    if (threadIdx.x == 0) {
      Y[i] = val * alpha;
    }
    __syncthreads();
  }
}

template <typename T, class Reducer, int D>
void ReduceTensorHIPImpl(
    const int outer_size,
    const int inner_size,
    const int* dims,
    const int* axes,
    const Reducer& reducer,
    const T init,
    const T alpha,
    const T* X,
    T* Y,
    HIPContext* context) {
  SimpleArray<int, D> X_strides;
  SimpleArray<FixedDivisor<int>, D> Y_dims;
  utils::ComputeTransposedStrides(D, dims, axes, X_strides.data);
  for (int i = 0; i < D; ++i) {
    Y_dims.data[i] = FixedDivisor<int>(dims[axes[i]]);
  }
  hipLaunchKernelGGL(
      (ReduceTensorHIPKernel<T, Reducer, D>),
      dim3(std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      outer_size,
      inner_size,
      X_strides,
      Y_dims,
      reducer,
      init,
      alpha,
      X,
      Y);
}

template <typename T, class Reducer>
void ReduceTensorHIP(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const Reducer& reducer,
    const T init,
    const T alpha,
    const T* X,
    T* Y,
    HIPContext* context) {
  CAFFE_ENFORCE_LE(num_axes, num_dims);
  std::vector<int> transpose_axes(num_dims);
  utils::ComputeTransposeAxesForReduceOp(
      num_dims, num_axes, axes, transpose_axes.data());
  const int pivot = num_dims - num_axes;
  int outer_size = 1;
  for (int i = 0; i < pivot; ++i) {
    outer_size *= dims[transpose_axes[i]];
  }
  int inner_size = 1;
  for (int i = pivot; i < num_dims; ++i) {
    inner_size *= dims[transpose_axes[i]];
  }
  if (outer_size == 0) {
    return;
  }
  if (inner_size == 0) {
    Set<T, HIPContext>(outer_size, alpha * init, Y, context);
    return;
  }
  if (inner_size == 1) {
    Scale<T, T, HIPContext>(outer_size, alpha, X, Y, context);
    return;
  }
  if (transpose_axes[pivot] == pivot) {
    hipLaunchKernelGGL(
        (RowwiseReduceKernel<T>),
        dim3(std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS)),
        dim3(CAFFE_HIP_NUM_THREADS),
        0,
        context->hip_stream(),
        outer_size,
        inner_size,
        reducer,
        init,
        alpha,
        X,
        Y);
    return;
  }
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_2(
      num_dims,
      ReduceTensorHIPImpl,
      T,
      Reducer,
      outer_size,
      inner_size,
      dims,
      transpose_axes.data(),
      reducer,
      init,
      alpha,
      X,
      Y,
      context);
}

template <typename T>
void ReduceMeanHIPImpl(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const T alpha,
    const T* X,
    T* Y,
    HIPContext* context) {
  const int X_size =
      std::accumulate(dims, dims + num_dims, 1, std::multiplies<int>());
  int scale = 1;
  for (int i = 0; i < num_axes; ++i) {
    scale *= dims[axes[i]];
  }
  ReduceTensorHIP(
      num_dims,
      dims,
      num_axes,
      axes,
      cub::Sum(),
      T(0),
      alpha / static_cast<T>(scale),
      X,
      Y,
      context);
}

} // namespace

#define CAFFE2_SPECIALIZED_HIP_REDUCE_MIN(T) \
  template <>                                \
  void ReduceMin<T, HIPContext>(             \
      const int num_dims,                    \
      const int* dims,                       \
      const int num_axes,                    \
      const int* axes,                       \
      const T alpha,                         \
      const T* X,                            \
      T* Y,                                  \
      HIPContext* context) {                 \
    ReduceTensorHIP(                         \
        num_dims,                            \
        dims,                                \
        num_axes,                            \
        axes,                                \
        cub::Min(),                          \
        std::numeric_limits<T>::max(),       \
        alpha,                               \
        X,                                   \
        Y,                                   \
        context);                            \
  }
CAFFE2_SPECIALIZED_HIP_REDUCE_MIN(std::int32_t)
CAFFE2_SPECIALIZED_HIP_REDUCE_MIN(std::int64_t)
CAFFE2_SPECIALIZED_HIP_REDUCE_MIN(float)
CAFFE2_SPECIALIZED_HIP_REDUCE_MIN(double)
#undef CAFFE2_SPECIALIZED_HIP_REDUCE_MIN

#define CAFFE2_SPECIALIZED_HIP_REDUCE_MAX(T) \
  template <>                                \
  void ReduceMax<T, HIPContext>(             \
      const int num_dims,                    \
      const int* dims,                       \
      const int num_axes,                    \
      const int* axes,                       \
      const T alpha,                         \
      const T* X,                            \
      T* Y,                                  \
      HIPContext* context) {                 \
    ReduceTensorHIP(                         \
        num_dims,                            \
        dims,                                \
        num_axes,                            \
        axes,                                \
        cub::Max(),                          \
        std::numeric_limits<T>::lowest(),    \
        alpha,                               \
        X,                                   \
        Y,                                   \
        context);                            \
  }
CAFFE2_SPECIALIZED_HIP_REDUCE_MAX(std::int32_t)
CAFFE2_SPECIALIZED_HIP_REDUCE_MAX(std::int64_t)
CAFFE2_SPECIALIZED_HIP_REDUCE_MAX(float)
CAFFE2_SPECIALIZED_HIP_REDUCE_MAX(double)
#undef CAFFE2_SPECIALIZED_HIP_REDUCE_MAX

#define CAFFE2_SPECIALIZED_HIP_REDUCE_SUM(T) \
  template <>                                \
  void ReduceSum<T, HIPContext>(             \
      const int num_dims,                    \
      const int* dims,                       \
      const int num_axes,                    \
      const int* axes,                       \
      const T alpha,                         \
      const T* X,                            \
      T* Y,                                  \
      HIPContext* context) {                 \
    ReduceTensorHIP(                         \
        num_dims,                            \
        dims,                                \
        num_axes,                            \
        axes,                                \
        cub::Sum(),                          \
        T(0),                                \
        alpha,                               \
        X,                                   \
        Y,                                   \
        context);                            \
  }
CAFFE2_SPECIALIZED_HIP_REDUCE_SUM(std::int32_t)
CAFFE2_SPECIALIZED_HIP_REDUCE_SUM(std::int64_t)
CAFFE2_SPECIALIZED_HIP_REDUCE_SUM(float)
CAFFE2_SPECIALIZED_HIP_REDUCE_SUM(double)
#undef CAFFE2_SPECIALIZED_HIP_REDUCE_SUM

#define CAFFE2_SPECIALIZED_HIP_REDUCE_MEAN(T)                  \
  template <>                                                  \
  void ReduceMean<T, HIPContext>(                              \
      const int num_dims,                                      \
      const int* dims,                                         \
      const int num_axes,                                      \
      const int* axes,                                         \
      const T alpha,                                           \
      const T* X,                                              \
      T* Y,                                                    \
      HIPContext* context) {                                   \
    ReduceMeanHIPImpl<T>(                                      \
        num_dims, dims, num_axes, axes, alpha, X, Y, context); \
  }
CAFFE2_SPECIALIZED_HIP_REDUCE_MEAN(float)
#undef CAFFE2_SPECIALIZED_HIP_REDUCE_MEAN

namespace {

template <typename T, int D>
__global__ void BroadcastHIPKernel(
    const int Y_size,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const T alpha,
    const T* X,
    T* Y) {
  HIP_1D_KERNEL_LOOP(Y_index, Y_size) {
    int X_index = 0;
    int Y_index_val = Y_index;
#pragma unroll
    for (int i = D - 1; i >= 0; --i) {
      X_index += X_strides.data[i] == 0
          ? 0
          : (Y_index_val % Y_dims.data[i]) * X_strides.data[i];
      Y_index_val /= Y_dims.data[i];
    }
    Y[Y_index] = __ldg(X + X_index) * alpha;
  }
}

template <typename T, int D>
void BroadcastHIPImpl(
    const int X_ndim,
    const int* X_dims,
    const int* Y_dims,
    const T alpha,
    const T* X,
    T* Y,
    HIPContext* context) {
  SimpleArray<int, D> X_strides_array;
  SimpleArray<int, D> Y_dims_array;
  const int d = D - X_ndim;
  std::fill(X_strides_array.data, X_strides_array.data + d, 0);
  int cur_stride = 1;
  for (int i = D - 1; i >= d; --i) {
    CAFFE_ENFORCE(X_dims[i - d] == 1 || X_dims[i - d] == Y_dims[i]);
    X_strides_array.data[i] = X_dims[i - d] == 1 ? 0 : cur_stride;
    cur_stride *= X_dims[i - d];
  }
  std::copy_n(Y_dims, D, Y_dims_array.data);
  const int Y_size =
      std::accumulate(Y_dims, Y_dims + D, 1, std::multiplies<int>());
  hipLaunchKernelGGL(
      (BroadcastHIPKernel<T, D>),
      dim3(CAFFE_GET_BLOCKS(Y_size)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      Y_size,
      X_strides_array,
      Y_dims_array,
      alpha,
      X,
      Y);
}

} // namespace

#define CAFFE2_SPECIALIZED_HIP_BROADCAST(T) \
  template <>                               \
  void Broadcast<T, HIPContext>(            \
      const int X_ndim,                     \
      const int* X_dims,                    \
      const int Y_ndim,                     \
      const int* Y_dims,                    \
      const T alpha,                        \
      const T* X,                           \
      T* Y,                                 \
      HIPContext* context) {                \
    CAFFE_ENFORCE_LE(X_ndim, Y_ndim);       \
    DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1( \
        Y_ndim,                             \
        BroadcastHIPImpl,                   \
        T,                                  \
        X_ndim,                             \
        X_dims,                             \
        Y_dims,                             \
        alpha,                              \
        X,                                  \
        Y,                                  \
        context);                           \
  }
CAFFE2_SPECIALIZED_HIP_BROADCAST(std::int32_t)
CAFFE2_SPECIALIZED_HIP_BROADCAST(std::int64_t)
CAFFE2_SPECIALIZED_HIP_BROADCAST(float)
CAFFE2_SPECIALIZED_HIP_BROADCAST(double)
#undef CAFFE2_SPECIALIZED_HIP_BROADCAST

namespace {

template <typename T>
__global__ void RowwiseMomentsHIPKernel(
    const int rows,
    const int cols,
    const T* X,
    T* mean,
    T* variance) {
  __shared__ typename BlockReduce<T>::TempStorage m_storage;
  __shared__ typename BlockReduce<T>::TempStorage v_storage;
  for (int i = blockIdx.x; i < rows; i += gridDim.x) {
    T m_val = 0;
    T v_val = 0;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
      const int X_index = i * cols + j;
      m_val += __ldg(X + X_index);
      v_val += __ldg(X + X_index) * __ldg(X + X_index);
    }
    m_val = BlockReduce<T>(m_storage).Reduce(m_val, cub::Sum());
    v_val = BlockReduce<T>(v_storage).Reduce(v_val, cub::Sum());
    if (threadIdx.x == 0) {
      mean[i] = m_val / static_cast<T>(cols);
      variance[i] = v_val / static_cast<T>(cols) - mean[i] * mean[i];
    }
    __syncthreads();
  }
}

template <typename T, int D>
__global__ void MomentsHIPKernel(
    const int outer_size,
    const int inner_size,
    SimpleArray<int, D> X_strides,
    SimpleArray<FixedDivisor<int>, D> Y_dims,
    const T* X,
    T* mean,
    T* variance) {
  __shared__ typename BlockReduce<T>::TempStorage m_storage;
  __shared__ typename BlockReduce<T>::TempStorage v_storage;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T m_val = 0;
    T v_val = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      int X_index = 0;
      int Y_index = i * inner_size + j;
#pragma unroll
      for (int d = D - 1; d >= 0; --d) {
        int r;
        Y_dims.data[d].DivMod(Y_index, &Y_index, &r);
        X_index += r * X_strides.data[d];
      }
      m_val += __ldg(X + X_index);
      v_val += __ldg(X + X_index) * __ldg(X + X_index);
    }
    m_val = BlockReduce<T>(m_storage).Reduce(m_val, cub::Sum());
    v_val = BlockReduce<T>(v_storage).Reduce(v_val, cub::Sum());
    if (threadIdx.x == 0) {
      mean[i] = m_val / static_cast<T>(inner_size);
      variance[i] = v_val / static_cast<T>(inner_size) - mean[i] * mean[i];
    }
    __syncthreads();
  }
}

template <typename T, int D>
void MomentsHIPImpl(
    const int outer_size,
    const int inner_size,
    const int* dims,
    const int* axes,
    const T* X,
    T* mean,
    T* variance,
    HIPContext* context) {
  SimpleArray<int, D> X_strides;
  SimpleArray<FixedDivisor<int>, D> Y_dims;
  utils::ComputeTransposedStrides(D, dims, axes, X_strides.data);
  for (int i = 0; i < D; ++i) {
    Y_dims.data[i] = FixedDivisor<int>(dims[axes[i]]);
  }
  hipLaunchKernelGGL(
      (MomentsHIPKernel<T, D>),
      dim3(std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      outer_size,
      inner_size,
      X_strides,
      Y_dims,
      X,
      mean,
      variance);
}

template <typename T>
void MomentsHIP(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const T* X,
    T* mean,
    T* variance,
    HIPContext* context) {
  CAFFE_ENFORCE_LE(num_axes, num_dims);
  std::vector<int> transpose_axes(num_dims);
  utils::ComputeTransposeAxesForReduceOp(
      num_dims, num_axes, axes, transpose_axes.data());
  const int pivot = num_dims - num_axes;
  int outer_size = 1;
  for (int i = 0; i < pivot; ++i) {
    outer_size *= dims[transpose_axes[i]];
  }
  int inner_size = 1;
  for (int i = pivot; i < num_dims; ++i) {
    inner_size *= dims[transpose_axes[i]];
  }
  if (outer_size > 0 && inner_size > 0) {
    if (transpose_axes[pivot] == pivot) {
      hipLaunchKernelGGL(
          (RowwiseMomentsHIPKernel<T>),
          dim3(std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS)),
          dim3(CAFFE_HIP_NUM_THREADS),
          0,
          context->hip_stream(),
          outer_size,
          inner_size,
          X,
          mean,
          variance);
      return;
    }
    DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(
        num_dims,
        MomentsHIPImpl,
        T,
        outer_size,
        inner_size,
        dims,
        transpose_axes.data(),
        X,
        mean,
        variance,
        context);
  }
}

} // namespace

#define CAFFE2_SPECIALIZED_HIP_MOMENTS(T)                                      \
  template <>                                                                  \
  void Moments<T, HIPContext>(                                                 \
      const int num_dims,                                                      \
      const int* dims,                                                         \
      const int num_axes,                                                      \
      const int* axes,                                                         \
      const T* X,                                                              \
      T* mean,                                                                 \
      T* variance,                                                             \
      HIPContext* context) {                                                   \
    MomentsHIP<T>(num_dims, dims, num_axes, axes, X, mean, variance, context); \
  }
CAFFE2_SPECIALIZED_HIP_MOMENTS(float)
#undef CAFFE2_SPECIALIZED_HIP_MOMENTS

namespace {

template <typename T, int D>
__global__ void TransposeHIPKernel(
    const int size,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<FixedDivisor<int>, D> Y_dims,
    const T* X,
    T* Y) {
  HIP_1D_KERNEL_LOOP(Y_index, size) {
    int X_index = 0;
    int Y_index_val = Y_index;
#pragma unroll
    for (int i = D - 1; i >= 0; --i) {
      int d;
      Y_dims.data[i].DivMod(Y_index_val, &Y_index_val, &d);
      X_index += d * X_strides.data[i];
    }
    Y[Y_index] = __ldg(X + X_index);
  }
}

template <typename T, int D>
void TransposeHIPImpl(
    const int* dims,
    const int* axes,
    const T* X,
    T* Y,
    HIPContext* context) {
  SimpleArray<int, D> X_strides;
  SimpleArray<FixedDivisor<int>, D> Y_dims;
  utils::ComputeTransposedStrides(D, dims, axes, X_strides.data);
  int size = 1;
  for (int i = 0; i < D; ++i) {
    Y_dims.data[i] = FixedDivisor<int>(dims[axes[i]]);
    size *= dims[i];
  }
  hipLaunchKernelGGL(
      (TransposeHIPKernel<T, D>),
      dim3(CAFFE_GET_BLOCKS(size)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(),
      size,
      X_strides,
      Y_dims,
      X,
      Y);
}

} // namespace

#define CAFFE2_SPECIALIZED_HIP_TRANSPOSE(T)                              \
  template <>                                                            \
  void Transpose<T, HIPContext>(                                         \
      const int ndim,                                                    \
      const int* dims,                                                   \
      const int* axes,                                                   \
      const T* X,                                                        \
      T* Y,                                                              \
      HIPContext* context) {                                             \
    if (utils::IsIdentityPermutation(ndim, axes)) {                      \
      const int size =                                                   \
          std::accumulate(dims, dims + ndim, 1, std::multiplies<int>()); \
      context->template Copy<T, HIPContext, HIPContext>(size, X, Y);     \
      return;                                                            \
    }                                                                    \
    DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(                              \
        ndim, TransposeHIPImpl, T, dims, axes, X, Y, context);           \
  }
CAFFE2_SPECIALIZED_HIP_TRANSPOSE(float)
CAFFE2_SPECIALIZED_HIP_TRANSPOSE(double)
CAFFE2_SPECIALIZED_HIP_TRANSPOSE(int)
CAFFE2_SPECIALIZED_HIP_TRANSPOSE(TIndex)
#undef CAFFE2_SPECIALIZED_HIP_TRANSPOSE
} // namespace math
} // namespace caffe2
