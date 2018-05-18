// Implements the math functions for GPU.

#include "caffe2/utils/math.h"

#include <limits>
#include <numeric>
#include <vector>

#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/conversions.h"

#if THRUST_VERSION >= 100800
#define THRUST_SUPPORTS_PER_THREAD
#endif // THRUST_VERSION >= 100800

namespace caffe2 {
namespace math {

#define DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(T, Funcname, function)          \
  __global__ void _Kernel_##T##_##Funcname(const int N, const T* x, T* y) { \
    CUDA_1D_KERNEL_LOOP(i, N) {                                             \
      y[i] = function(x[i]);                                                \
    }                                                                       \
  }                                                                         \
  template <>                                                               \
  void Funcname<T, CUDAContext>(                                            \
      const int N, const T* x, T* y, CUDAContext* context) {                \
    _Kernel_##T##_##Funcname<<<                                             \
        CAFFE_GET_BLOCKS(N),                                                \
        CAFFE_CUDA_NUM_THREADS,                                             \
        0,                                                                  \
        context->cuda_stream()>>>(N, x, y);                                 \
  }

DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Exp, expf);
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Log, logf);
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Cos, cosf);
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Acos, acosf);
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Sin, sinf);
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Asin, asinf);
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Tan, tanf);
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Atan, atanf);
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Abs, fabsf);
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Sqrt, sqrtf);
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, InvSqrt, rsqrtf);

__device__ float cuda_sqrf(const float x) {
  return x * x;
}

DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Sqr, cuda_sqrf);

#undef DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION

#define DELEGATE_SINCOS_CUDA_FUNCTION(T)                             \
  __global__ void _Kernel_##T##_##SinCos(                            \
      const int N, const T* x, T* ys, T* yc) {                       \
    CUDA_1D_KERNEL_LOOP(i, N) {                                      \
      sincos(x[i], ys + i, yc + i);                                  \
    }                                                                \
  }                                                                  \
  template <>                                                        \
  void SinCos<T, CUDAContext>(                                       \
      const int N, const T* x, T* ys, T* yc, CUDAContext* context) { \
    _Kernel_##T##_##SinCos<<<                                        \
        CAFFE_GET_BLOCKS(N),                                         \
        CAFFE_CUDA_NUM_THREADS,                                      \
        0,                                                           \
        context->cuda_stream()>>>(N, x, ys, yc);                     \
  }

DELEGATE_SINCOS_CUDA_FUNCTION(float)
DELEGATE_SINCOS_CUDA_FUNCTION(double)

#undef DELEGATE_SINCOS_CUDA_FUNCTION

#define DELEGATE_SIMPLE_CUDA_BINARY_INFIX_FUNCTION(T, Funcname, expr)         \
  __global__ void _Kernel_##T##_##Funcname(                                   \
      const int N, const T* a, const T* b, T* y) {                            \
    CUDA_1D_KERNEL_LOOP(i, N) {                                               \
      float r = convert::To<T, float>(a[i]) expr convert::To<T, float>(b[i]); \
      y[i] = convert::To<float, T>(r);                                        \
    }                                                                         \
  }                                                                           \
  template <>                                                                 \
  void Funcname<T, CUDAContext>(                                              \
      const int N, const T* a, const T* b, T* y, CUDAContext* context) {      \
    _Kernel_##T##_##Funcname<<<                                               \
        CAFFE_GET_BLOCKS(N),                                                  \
        CAFFE_CUDA_NUM_THREADS,                                               \
        0,                                                                    \
        context->cuda_stream()>>>(N, a, b, y);                                \
  }

DELEGATE_SIMPLE_CUDA_BINARY_INFIX_FUNCTION(float, Add, +);
DELEGATE_SIMPLE_CUDA_BINARY_INFIX_FUNCTION(int32_t, Add, +);
DELEGATE_SIMPLE_CUDA_BINARY_INFIX_FUNCTION(float, Sub, -);
DELEGATE_SIMPLE_CUDA_BINARY_INFIX_FUNCTION(float, Mul, *);
DELEGATE_SIMPLE_CUDA_BINARY_INFIX_FUNCTION(float, Div, /);

DELEGATE_SIMPLE_CUDA_BINARY_INFIX_FUNCTION(float16, Add, +);
DELEGATE_SIMPLE_CUDA_BINARY_INFIX_FUNCTION(float16, Sub, -);
DELEGATE_SIMPLE_CUDA_BINARY_INFIX_FUNCTION(float16, Mul, *);
DELEGATE_SIMPLE_CUDA_BINARY_INFIX_FUNCTION(float16, Div, /);

#undef DELEGATE_SIMPLE_CUDA_BINARY_INFIX_FUNCTION

#define DELEGATE_SIMPLE_CUDA_BINARY_PREFIX_FUNCTION(T, Funcname, func)    \
  __global__ void _Kernel_##T##_##Funcname(                               \
      const int N, const T* a, const T* b, T* y) {                        \
    CUDA_1D_KERNEL_LOOP(i, N) {                                           \
      float r =                                                           \
          func(convert::To<T, float>(a[i]), convert::To<T, float>(b[i])); \
      y[i] = convert::To<float, T>(r);                                    \
    }                                                                     \
  }                                                                       \
  template <>                                                             \
  void Funcname<T, CUDAContext>(                                          \
      const int N, const T* a, const T* b, T* y, CUDAContext* context) {  \
    _Kernel_##T##_##Funcname<<<                                           \
        CAFFE_GET_BLOCKS(N),                                              \
        CAFFE_CUDA_NUM_THREADS,                                           \
        0,                                                                \
        context->cuda_stream()>>>(N, a, b, y);                            \
  }

DELEGATE_SIMPLE_CUDA_BINARY_PREFIX_FUNCTION(float, ElemwiseMax, fmaxf);

#undef DELEGATE_SIMPLE_CUDA_BINARY_INFIX_FUNCTION

#define DELEGATE_REDUCTION_FUNCTION(T, Funcname, func)                  \
  template <>                                                           \
  void Funcname<T, CUDAContext>(                                        \
      const int N,                                                      \
      const T* src,                                                     \
      T* dst,                                                           \
      Tensor<CUDAContext>* scratch_ptr,                                 \
      CUDAContext* context) {                                           \
    size_t memRequired = 0;                                             \
    cub::DeviceReduce::func(                                            \
        nullptr, memRequired, src, dst, N, context->cuda_stream());     \
    auto buffer_size =                                                  \
        static_cast<TIndex>((memRequired + sizeof(T) - 1) / sizeof(T)); \
    scratch_ptr->Resize(std::vector<TIndex>{buffer_size});              \
    cub::DeviceReduce::func(                                            \
        static_cast<void*>(scratch_ptr->mutable_data<T>()),             \
        memRequired,                                                    \
        src,                                                            \
        dst,                                                            \
        N,                                                              \
        context->cuda_stream());                                        \
  }

DELEGATE_REDUCTION_FUNCTION(float, ReduceMin, Min)
DELEGATE_REDUCTION_FUNCTION(float, ReduceMax, Max)
DELEGATE_REDUCTION_FUNCTION(int32_t, ReduceMax, Max)
DELEGATE_REDUCTION_FUNCTION(int64_t, ReduceMax, Max)

#undef DELEGATE_REDUCTION_FUNCTION

// Caffe2 gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template <>
void Gemm<float, CUDAContext>(
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
    CUDAContext* context,
    TensorProto::DataType math_type) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_ENFORCE(cublasSgemm(
      context->cublas_handle(),
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
void Gemm<float16, CUDAContext>(
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
    CUDAContext* context,
    TensorProto::DataType math_type) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  if (math_type == TensorProto_DataType_FLOAT) {
    CUBLAS_CHECK(cublasSgemmEx(
        context->cublas_handle(),
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
    auto alpha_fp16 = convert::floatToHalf(alpha);
    auto beta_fp16 = convert::floatToHalf(beta);

    // call cublasHgemm
    CUBLAS_CHECK(cublasHgemm(
        context->cublas_handle(),
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
        N));
  } else {
    // fail
    CAFFE_THROW("Unsupported math type");
  }
}

template <>
void GemmBatched<float, CUDAContext>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const float* B,
    const float beta,
    float* C,
    CUDAContext* context,
    Tensor<CUDAContext>* scratch,
    TensorProto::DataType math_type) {
  const int a_stride = M * K;
  const int b_stride = K * N;
  const int c_stride = M * N;
#if __CUDACC_VER_MAJOR__ < 8
  // loop over matrices in the batch
  for (int i = 0; i < batch_size; ++i) {
    math::Gemm<float, CUDAContext>(
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
#else
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_ENFORCE(cublasSgemmStridedBatched(
      context->cublas_handle(),
      cuTransB,
      cuTransA,
      N,
      M,
      K,
      &alpha,
      B,
      ldb,
      b_stride,
      A,
      lda,
      a_stride,
      &beta,
      C,
      N,
      c_stride,
      batch_size));
#endif
}

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

}; // namespace

template <>
void GemmBatched<float16, CUDAContext>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float16* A,
    const float16* B,
    const float beta,
    float16* C,
    CUDAContext* context,
    Tensor<CUDAContext>* scratch,
    TensorProto::DataType math_type) {
  const int a_stride = M * K;
  const int b_stride = K * N;
  const int c_stride = M * N;
#if __CUDACC_VER_MAJOR__ < 8
  // loop over matrices in the batch
  for (int i = 0; i < batch_size; ++i) {
    math::Gemm<float16, CUDAContext>(
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
#else
  // 3 options:
  // 1) scratch != null = cast to fp32, SgemmStridedBatched, cast result to fp16
  // 2) math_type == FLOAT, scratch == nullptr = looped SgemmEx
  // 3) math_type == FLOAT16, scratch == nullptr = batched Hgemm

  if (scratch != nullptr) {
    const int A_size = a_stride * batch_size;
    const int B_size = b_stride * batch_size;
    // cast, cublasSgemmStridedBatched, cast
    size_t in_elems = A_size + B_size;
    size_t out_elems = c_stride * batch_size;

    scratch->Resize(in_elems + out_elems);
    float* scratch_ptr = scratch->mutable_data<float>();

    float* A_fp32 = scratch_ptr;
    float* B_fp32 = scratch_ptr + A_size;
    float* C_fp32 = scratch_ptr + A_size + B_size;

    // cast A, B into fp32
    HalfToFloatKernel<<<
        CAFFE_GET_BLOCKS(A_size),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context->cuda_stream()>>>(A_size, (half*)A, A_fp32);
    HalfToFloatKernel<<<
        CAFFE_GET_BLOCKS(B_size),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context->cuda_stream()>>>(B_size, (half*)B, B_fp32);

    // run fp32 batched Gemm
    GemmBatched<float, CUDAContext>(
        TransA,
        TransB,
        batch_size,
        M,
        N,
        K,
        alpha,
        A_fp32,
        B_fp32,
        beta,
        C_fp32,
        context);

    // cast result back to fp16
    FloatToHalfKernel<<<
        CAFFE_GET_BLOCKS(batch_size * M * N),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context->cuda_stream()>>>(batch_size * M * N, C_fp32, (half*)C);
  } else {
    if (math_type == TensorProto_DataType_FLOAT) {
      // loop over matrices in the batch
      for (int i = 0; i < batch_size; ++i) {
        math::Gemm<float16, CUDAContext>(
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
      cublasOperation_t cuTransA =
          (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
      cublasOperation_t cuTransB =
          (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

      // convert alpha, beta from float -> __half
      auto alpha_fp16 = convert::floatToHalf(alpha);
      auto beta_fp16 = convert::floatToHalf(beta);
      CUBLAS_ENFORCE(cublasHgemmStridedBatched(
          context->cublas_handle(),
          cuTransB,
          cuTransA,
          N,
          M,
          K,
          &alpha_fp16,
          (const __half*)B,
          ldb,
          b_stride,
          (const __half*)A,
          lda,
          a_stride,
          &beta_fp16,
          (__half*)C,
          N,
          c_stride,
          batch_size));
    }
  }
#endif
}

#if CUDA_VERSION >= 9000

// No change, but required. Defer to default CUDA engine
template <>
void Gemm<float, CUDAContext, TensorCoreEngine>(
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
    CUDAContext* context,
    TensorProto::DataType math_type) {
  return Gemm<float, CUDAContext>(
      TransA, TransB, M, N, K, alpha, A, B, beta, C, context, math_type);
}

template <>
void Gemm<float16, CUDAContext, TensorCoreEngine>(
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
    CUDAContext* context,
    TensorProto::DataType math_type) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  // enable TensorCore for this call on this handle
  if (TensorCoreAvailable()) {
    CUBLAS_ENFORCE(
        cublasSetMathMode(context->cublas_handle(), CUBLAS_TENSOR_OP_MATH));
  }

  CUBLAS_CHECK(cublasGemmEx(
      context->cublas_handle(),
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
      N,
      CUDA_R_32F,
      CUBLAS_GEMM_DFALT_TENSOR_OP));

  // Now disable TensorCore math for subsequent calls to this handle
  if (TensorCoreAvailable()) {
    CUBLAS_ENFORCE(
        cublasSetMathMode(context->cublas_handle(), CUBLAS_DEFAULT_MATH));
  }
}

template <>
void GemmBatched<float, CUDAContext, TensorCoreEngine>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const float* B,
    const float beta,
    float* C,
    CUDAContext* context,
    Tensor<CUDAContext>* scratch,
    TensorProto::DataType math_type) {
  return GemmBatched<float, CUDAContext, DefaultEngine>(
      TransA,
      TransB,
      batch_size,
      M,
      N,
      K,
      alpha,
      A,
      B,
      beta,
      C,
      context,
      scratch,
      math_type);
}

template <>
void GemmBatched<float16, CUDAContext, TensorCoreEngine>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float16* A,
    const float16* B,
    const float beta,
    float16* C,
    CUDAContext* context,
    Tensor<CUDAContext>* scratch,
    TensorProto::DataType math_type) {
  return GemmBatched<float16, CUDAContext, DefaultEngine>(
      TransA,
      TransB,
      batch_size,
      M,
      N,
      K,
      alpha,
      A,
      B,
      beta,
      C,
      context,
      scratch,
      math_type);
}

#endif // CUDA_VERSION >= 9000

template <>
void GemmEx<float, CUDAContext>(
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
    CUDAContext* context) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_ENFORCE(cublasSgemm(
      context->cublas_handle(),
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
void Gemv<float, CUDAContext>(
    const CBLAS_TRANSPOSE TransA,
    const int M,
    const int N,
    const float alpha,
    const float* A,
    const float* x,
    const float beta,
    float* y,
    CUDAContext* context,
    TensorProto::DataType math_type) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_ENFORCE(cublasSgemv(
      context->cublas_handle(),
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
    CUDA_1D_KERNEL_LOOP(i, N) {
      float tmpY = convert::To<T, float>(Y[i]);
      tmpY += convert::To<T, float>(x[i]);
      Y[i] = convert::To<float, T>(tmpY);
    }
  }
}
} // namespace

#define CAFFE2_SPECIALIZED_CUDA_ADD_STRIPED_BATCH(T)              \
  template <>                                                     \
  void AddStripedBatch<T, CUDAContext>(                           \
      const int N,                                                \
      const T* first,                                             \
      T* Y,                                                       \
      const int stripe,                                           \
      const int batch,                                            \
      CUDAContext* context) {                                     \
    AddStripedBatchKernel<T>                                      \
        <<<CAFFE_GET_BLOCKS(N),                                   \
           CAFFE_CUDA_NUM_THREADS,                                \
           0,                                                     \
           context->cuda_stream()>>>(N, first, Y, stripe, batch); \
  }

CAFFE2_SPECIALIZED_CUDA_ADD_STRIPED_BATCH(float);
CAFFE2_SPECIALIZED_CUDA_ADD_STRIPED_BATCH(float16);
#undef CAFFE2_SPECIALIZED_CUDA_ADD_STRIPED_BATCH

template <>
void Gemv<float16, CUDAContext>(
    const CBLAS_TRANSPOSE TransA,
    const int M,
    const int N,
    const float alpha,
    const float16* A,
    const float16* x,
    const float beta,
    float16* y,
    CUDAContext* context,
    TensorProto::DataType math_type) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;

  // sort out what we need to call cublasSgemmEx / cublasHgemm
  int m = (cuTransA == CUBLAS_OP_N) ? N : M;
  int k = (cuTransA == CUBLAS_OP_N) ? M : N;
  int LDA = (cuTransA == CUBLAS_OP_N) ? m : k;
  int LDC = m;

  if (math_type == TensorProto_DataType_FLOAT) {
    CUBLAS_CHECK(cublasSgemmEx(
        context->cublas_handle(),
        cuTransA,
        CUBLAS_OP_N,
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

    CUBLAS_CHECK(cublasHgemm(
        context->cublas_handle(),
        cuTransA,
        CUBLAS_OP_N,
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
}

namespace {
template <typename T>
__global__ void SetKernel(const int N, const T alpha, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = alpha;
  }
}
} // namespace

#define CAFFE2_SPECIALIZED_CUDA_SET(T)                             \
  template <>                                                      \
  void Set<T, CUDAContext>(                                        \
      const size_t N, const T alpha, T* Y, CUDAContext* context) { \
    SetKernel<<<                                                   \
        CAFFE_GET_BLOCKS(N),                                       \
        CAFFE_CUDA_NUM_THREADS,                                    \
        0,                                                         \
        context->cuda_stream()>>>(N, alpha, Y);                    \
  }

CAFFE2_SPECIALIZED_CUDA_SET(float);
CAFFE2_SPECIALIZED_CUDA_SET(double);
CAFFE2_SPECIALIZED_CUDA_SET(bool);
CAFFE2_SPECIALIZED_CUDA_SET(int8_t);
CAFFE2_SPECIALIZED_CUDA_SET(int16_t);
CAFFE2_SPECIALIZED_CUDA_SET(float16);
CAFFE2_SPECIALIZED_CUDA_SET(int);
CAFFE2_SPECIALIZED_CUDA_SET(int64_t);
CAFFE2_SPECIALIZED_CUDA_SET(char);
CAFFE2_SPECIALIZED_CUDA_SET(uint8_t);
CAFFE2_SPECIALIZED_CUDA_SET(uint16_t);
#undef CAFFE2_SPECIALIZED_CUDA_SET

namespace {
template <typename T>
__global__ void
UniformShift(const size_t N, const float min, const float max, T* x) {
  float scale = max - min;
  CUDA_1D_KERNEL_LOOP(i, N) {
    x[i] = convert::To<float, T>(convert::To<T, float>(x[i]) * scale + min);
  }
}

__global__ void
UniformIntFit(const size_t N, const int min, const int max, unsigned int* x) {
  int* x_int = reinterpret_cast<int*>(x);
  int range = (max - min + 1);
  CUDA_1D_KERNEL_LOOP(i, N) {
    x_int[i] = min + static_cast<int>(x[i] % range);
  }
}
} // namespace

template <>
void RandUniform<float, CUDAContext>(
    const size_t n,
    const float min,
    const float max,
    float* r,
    CUDAContext* context) {
  CURAND_ENFORCE(curandGenerateUniform(context->curand_generator(), r, n));
  UniformShift<float>
      <<<CAFFE_GET_BLOCKS(n),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(n, min, max, r);
}

template <>
void RandUniform<double, CUDAContext>(
    const size_t n,
    const double min,
    const double max,
    double* r,
    CUDAContext* context) {
  CURAND_ENFORCE(
      curandGenerateUniformDouble(context->curand_generator(), r, n));
  UniformShift<double>
      <<<CAFFE_GET_BLOCKS(n),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(n, min, max, r);
}

template <>
void RandUniform<int, CUDAContext>(
    const size_t n,
    const int min,
    const int max,
    int* r,
    CUDAContext* context) {
  CURAND_ENFORCE(curandGenerate(
      context->curand_generator(), reinterpret_cast<unsigned int*>(r), n));
  UniformIntFit<<<
      CAFFE_GET_BLOCKS(n),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(
      n, min, max, reinterpret_cast<unsigned int*>(r));
}

template <typename T>
size_t HandleOddLengthRandGaussian(
    const size_t n,
    const T mean,
    const T std,
    T* r,
    CUDAContext* context) {
  if (n % 2 == 1) {
    std::default_random_engine generator;
    std::normal_distribution<T> distribution(mean, std);
    const T random_value = distribution(generator);
    math::Set<T, CUDAContext>(1, random_value, r + (n - 1), context);
    return n - 1;
  }
  return n;
}

template <>
void RandGaussian<float, CUDAContext>(
    const size_t n,
    const float mean,
    const float std,
    float* r,
    CUDAContext* context) {
  // If n is odd, we add a random Gaussian value at the end manually
  // and generate n-1 random values using curandGenerateNormal.
  // curandGenerateNormal requires n to be even.
  const size_t even_n =
      HandleOddLengthRandGaussian<float>(n, mean, std, r, context);
  CURAND_ENFORCE(
      curandGenerateNormal(context->curand_generator(), r, even_n, mean, std));
}

template <>
void RandGaussian<double, CUDAContext>(
    const size_t n,
    const double mean,
    const double std,
    double* r,
    CUDAContext* context) {
  const size_t even_n =
      HandleOddLengthRandGaussian<double>(n, mean, std, r, context);
  CURAND_ENFORCE(curandGenerateNormalDouble(
      context->curand_generator(), r, even_n, mean, std));
}

template <>
void Dot<float, CUDAContext>(
    const int n,
    const float* a,
    const float* b,
    float* y,
    CUDAContext* context) {
  float result;
  CUBLAS_ENFORCE(cublasSdot(context->cublas_handle(), n, a, 1, b, 1, &result));
  context->Copy<float, CPUContext, CUDAContext>(1, &result, y);
}

template <>
void Dot<float16, CUDAContext>(
    const int n,
    const float16* a,
    const float16* b,
    float16* y,
    CUDAContext* context) {
  float16 result;
  // execute with 32-bit math
  CUBLAS_CHECK(cublasDotEx(
      context->cublas_handle(),
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
  context->Copy<float16, CPUContext, CUDAContext>(1, &result, y);
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
    CUDAContext* context,
    Tensor<CUDAContext>* scratch_ptr) {
  size_t memRequired = 0;
  cub::DeviceReduce::Sum(
      nullptr, memRequired, it, dest, N, context->cuda_stream());
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
      context->cuda_stream());
}
} // namespace

template <>
void Sum<float, CUDAContext>(
    const int N,
    const float* x,
    float* y,
    CUDAContext* context,
    Tensor<CUDAContext>* scratch_ptr) {
  if (scratch_ptr && N > DEVICE_REDUCE_SIZE_THRESHOLD) {
    SumGenericIter<float>(N, x, y, context, scratch_ptr);
  } else {
    SumKernel<<<1, SUM_KERNEL_NTHREADS, 0, context->cuda_stream()>>>(
        N, x, y, false);
  }
}

template <>
void Sum<int32_t, CUDAContext>(
    const int N,
    const int32_t* x,
    int32_t* y,
    CUDAContext* context,
    Tensor<CUDAContext>* scratch_ptr) {
  if (scratch_ptr && N > DEVICE_REDUCE_SIZE_THRESHOLD) {
    SumGenericIter<int32_t>(N, x, y, context, scratch_ptr);
  } else {
    SumKernel<<<1, SUM_KERNEL_NTHREADS, 0, context->cuda_stream()>>>(
        N, x, y, false);
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
  void Sum<T, CUDAContext>(                                               \
      const int N,                                                        \
      const T* x,                                                         \
      T* y,                                                               \
      CUDAContext* context,                                               \
      Tensor<CUDAContext>* scratch_ptr) {                                 \
    if (scratch_ptr && N > DEVICE_REDUCE_SIZE_THRESHOLD) {                \
      FloatTransform<T> transform;                                        \
      cub::TransformInputIterator<float, FloatTransform<T>, const T*> it( \
          x, transform);                                                  \
      float* sum = nullptr;                                               \
      SumGenericIter<float>(N, it, sum, context, scratch_ptr);            \
      SumConvertKernel<<<1, 1, 0, context->cuda_stream()>>>(sum, y);      \
    } else {                                                              \
      SumKernel<<<1, SUM_KERNEL_NTHREADS, 0, context->cuda_stream()>>>(   \
          N, x, y, false);                                                \
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
void SumSqr<float, CUDAContext>(
    const int N,
    const float* x,
    float* y,
    CUDAContext* context,
    Tensor<CUDAContext>* scratch_ptr) {
  if (scratch_ptr && N > DEVICE_REDUCE_SIZE_THRESHOLD) {
    SqrTransform<float> transform;
    cub::TransformInputIterator<float, SqrTransform<float>, const float*> it(
        x, transform);
    SumGenericIter<float>(N, it, y, context, scratch_ptr);
  } else {
    SumKernel<<<1, SUM_KERNEL_NTHREADS, 0, context->cuda_stream()>>>(
        N, x, y, true);
  }
}

#define CAFFE2_MATH_SUMSQR_FUNC(T)                                      \
  template <>                                                           \
  void SumSqr<T, CUDAContext>(                                          \
      const int N,                                                      \
      const T* x,                                                       \
      T* y,                                                             \
      CUDAContext* context,                                             \
      Tensor<CUDAContext>* scratch_ptr) {                               \
    if (scratch_ptr && N > DEVICE_REDUCE_SIZE_THRESHOLD) {              \
      FloatTransform<T> float_transform;                                \
      cub::TransformInputIterator<float, FloatTransform<T>, const T*>   \
          float_it(x, float_transform);                                 \
      SqrTransform<float> sqr_transform;                                \
      cub::TransformInputIterator<                                      \
          float,                                                        \
          SqrTransform<float>,                                          \
          decltype(float_it)>                                           \
          it(float_it, sqr_transform);                                  \
      float* sum = nullptr;                                             \
      SumGenericIter<float>(N, it, sum, context, scratch_ptr);          \
      SumConvertKernel<<<1, 1, 0, context->cuda_stream()>>>(sum, y);    \
    } else {                                                            \
      SumKernel<<<1, SUM_KERNEL_NTHREADS, 0, context->cuda_stream()>>>( \
          N, x, y, true);                                               \
    }                                                                   \
  }

CAFFE2_MATH_SUMSQR_FUNC(float16)
#undef CAFFE2_MATH_SUMSQR_FUNC
#undef DEVICE_REDUCE_SIZE_THRESHOLD

namespace {
template <typename T>
__global__ void
SelectKernel(const int N, const int D, const T* x, const int* idx, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = x[i * D + idx[i]];
  }
}
} // namespace

template <>
void Select<float, CUDAContext>(
    const int N,
    const int D,
    const float* x,
    const int* idx,
    float* y,
    CUDAContext* context) {
  SelectKernel<float>
      <<<CAFFE_GET_BLOCKS(N),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(N, D, x, idx, y);
}

template <>
void Select<float16, CUDAContext>(
    const int N,
    const int D,
    const float16* x,
    const int* idx,
    float16* y,
    CUDAContext* context) {
  SelectKernel<float16>
      <<<CAFFE_GET_BLOCKS(N),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(N, D, x, idx, y);
}

namespace {
template <typename T>
__global__ void ScaleKernel(const int n, const float alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    // y[i] = convert::To<float,T>(convert::To<T, float>(x[i]) * alpha);
    y[i] = convert::Get<T>(convert::Get<float>(x[i]) * alpha);
  }
}

template <typename T>
__global__ void
ScaleKernelDeviceAlpha(const int n, const float* alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = x[i] * (*alpha);
  }
}

template <typename T>
__global__ void PowKernel(const int n, const T* x, const T exponent, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = powf(x[i], exponent);
  }
}

// fp16 specialization
template <>
__global__ void ScaleKernelDeviceAlpha(
    const int n,
    const float* alpha,
    const float16* x,
    float16* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = convert::To<float, float16>(
        convert::To<float16, float>(x[i]) * (*alpha));
  }
}

} // namespace

template <>
void Powx<float, CUDAContext>(
    const int N,
    const float* a,
    const float b,
    float* y,
    CUDAContext* context) {
  PowKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(N, a, b, y);
}

template <>
void Scale<float, CUDAContext>(
    const int n,
    const float alpha,
    const float* x,
    float* y,
    CUDAContext* context) {
  ScaleKernel<float>
      <<<CAFFE_GET_BLOCKS(n),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(n, alpha, x, y);
}

template <>
void Scale<float16, CUDAContext>(
    const int n,
    const float alpha,
    const float16* x,
    float16* y,
    CUDAContext* context) {
  ScaleKernel<float16>
      <<<CAFFE_GET_BLOCKS(n),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(n, alpha, x, y);
}

template <>
void Scale<float, CUDAContext>(
    const int n,
    const float* alpha,
    const float* x,
    float* y,
    CUDAContext* context) {
  ScaleKernelDeviceAlpha<float>
      <<<CAFFE_GET_BLOCKS(n),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(n, alpha, x, y);
}

template <>
void Scale<float16, CUDAContext>(
    const int n,
    const float* alpha,
    const float16* x,
    float16* y,
    CUDAContext* context) {
  ScaleKernelDeviceAlpha<float16>
      <<<CAFFE_GET_BLOCKS(n),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(n, alpha, x, y);
}

template <>
void Axpy<float, CUDAContext>(
    const int N,
    const float alpha,
    const float* X,
    float* Y,
    CUDAContext* context) {
  CUBLAS_ENFORCE(cublasSaxpy(context->cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void Axpy<double, CUDAContext>(
    const int N,
    const float alpha,
    const double* X,
    double* Y,
    CUDAContext* context) {
  double alpha_d{alpha};
  CUBLAS_ENFORCE(
      cublasDaxpy(context->cublas_handle(), N, &alpha_d, X, 1, Y, 1));
}

template <>
void Axpy<float16, CUDAContext>(
    const int N,
    const float alpha,
    const float16* X,
    float16* Y,
    CUDAContext* context) {
  CUBLAS_CHECK(cublasAxpyEx(
      context->cublas_handle(),
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
}

namespace {
template <typename T>
__global__ void AxpyKernel(const int n, const float* a, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    y[index] = convert::Get<T>(
        convert::Get<float>(x[index]) * (*a) + convert::Get<float>(y[index]));
  }
}
} // namespace

template <>
void Axpy<float, CUDAContext>(
    const int n,
    const float* alpha,
    const float* X,
    float* Y,
    CUDAContext* context) {
  AxpyKernel<float>
      <<<CAFFE_GET_BLOCKS(n),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(n, alpha, X, Y);
}

template <>
void Axpy<float16, CUDAContext>(
    const int n,
    const float* alpha,
    const float16* X,
    float16* Y,
    CUDAContext* context) {
  AxpyKernel<float16>
      <<<CAFFE_GET_BLOCKS(n),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(n, alpha, X, Y);
}

namespace {
template <typename T>
__global__ void
AxpbyKernel(const int n, const T a, const T* x, const T b, T* y) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    y[index] = x[index] * a + y[index] * b;
  }
}
} // namespace

template <>
void Axpby<float, CUDAContext>(
    const int n,
    const float a,
    const float* x,
    const float b,
    float* y,
    CUDAContext* context) {
  AxpbyKernel<float>
      <<<CAFFE_GET_BLOCKS(n),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(n, a, x, b, y);
}

namespace {

template <typename T>
__global__ void Im2ColNCHWCUDAKernel(
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
  CUDA_1D_KERNEL_LOOP(index, n) {
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
#if __CUDA_ARCH__ >= 350
        *col_data_ptr = (h >= 0 && w >= 0 && h < input_h && w < input_w)
            ? __ldg(img_data_ptr + dh * input_w + dw)
            : 0;
#else
        *col_data_ptr = (h >= 0 && w >= 0 && h < input_h && w < input_w)
            ? img_data_ptr[dh * input_w + dw]
            : 0;
#endif
        col_data_ptr += output_size;
        dw += dilation_w;
      }
      dh += dilation_h;
    }
  }
}

template <typename T>
__global__ void Im2ColNHWCCUDAKernel(
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
  CUDA_1D_KERNEL_LOOP(index, n) {
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
#if __CUDA_ARCH__ >= 350
        *col_data_ptr = (h >= 0 && w >= 0 && h < input_h && w < input_w)
            ? __ldg(img_data + (h * input_w + w) * channels + channel_in)
            : 0;
#else
        *col_data_ptr = (h >= 0 && w >= 0 && h < input_h && w < input_w)
            ? img_data[(h * input_w + w) * channels + channel_in]
            : 0;
#endif
        col_data_ptr += channels;
        dw += dilation_w;
      }
      dh += dilation_h;
    }
  }
}

template <typename T>
__global__ void Col2ImNCHWCUDAKernel(
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

  CUDA_1D_KERNEL_LOOP(index, n) {
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
#if __CUDA_ARCH__ >= 350
          val += __ldg(col_data + col_data_index);
#else
          val += col_data[col_data_index];
#endif
        }
      }
    }
    img_data[index] = val;
  }
}

template <typename T>
__global__ void Col2ImNHWCCUDAKernel(
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

  CUDA_1D_KERNEL_LOOP(index, n) {
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
#if __CUDA_ARCH__ >= 350
          val += __ldg(
              col_data + (h_col * output_w + w_col) * channels_col + c_col);
#else
          val += col_data[(h_col * output_w + w_col) * channels_col + c_col];
#endif
        }
      }
    }
    img_data[index] = val;
  }
}

template <typename T, int N, bool kCol2Im>
__global__ void Im2ColNdNCHWCUDAKernel(
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
#if __CUDA_ARCH__ >= 350
      if (!kCol2Im) {
        Y_data[col_index] = is_padding ? 0 : __ldg(X_data + img_index);
      } else if (!is_padding) {
        atomicAdd(Y_data + img_index, __ldg(X_data + col_index));
      }
#else
      if (!kCol2Im) {
        Y_data[col_index] = is_padding ? 0 : X_data[img_index];
      } else if (!is_padding) {
        atomicAdd(Y_data + img_index, X_data[col_index]);
      }
#endif
    }
  }
}

template <typename T, int N>
void Im2ColNdNCHWCUDAImpl(
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
    CUDAContext* context) {
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
  Im2ColNdNCHWCUDAKernel<T, N, false>
      <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
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
void Col2ImNdNCHWCUDAImpl(
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
    CUDAContext* context) {
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
  Set<T, CUDAContext>(img_size, 0, img_data, context);
  Im2ColNdNCHWCUDAKernel<T, N, true>
      <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
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
void Im2Col<float, CUDAContext, StorageOrder::NCHW>(
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
    CUDAContext* context) {
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
  const int output_h = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  const int output_w = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
  const int num_kernels = channels * output_h * output_w;
  Im2ColNCHWCUDAKernel<float>
      <<<CAFFE_GET_BLOCKS(num_kernels),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
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
void Im2Col<float, CUDAContext, StorageOrder::NHWC>(
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
    CUDAContext* context) {
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
  const int output_h = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  const int output_w = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
  const int num_kernels = output_h * output_w * channels;
  Im2ColNHWCCUDAKernel<float>
      <<<CAFFE_GET_BLOCKS(num_kernels),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
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
void Col2Im<float, CUDAContext, StorageOrder::NCHW>(
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
    CUDAContext* context) {
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
  const int output_h = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  const int output_w = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
  const int num_kernels = channels * height * width;
  Col2ImNCHWCUDAKernel<float>
      <<<CAFFE_GET_BLOCKS(num_kernels),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
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
void Col2Im<float, CUDAContext, StorageOrder::NHWC>(
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
    CUDAContext* context) {
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
  const int output_h = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  const int output_w = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
  const int num_kernels = height * width * channels;
  Col2ImNHWCCUDAKernel<float>
      <<<CAFFE_GET_BLOCKS(num_kernels),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
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
void Im2ColNd<float, CUDAContext, StorageOrder::NCHW>(
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
    CUDAContext* context) {
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(
      N,
      Im2ColNdNCHWCUDAImpl,
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
void Col2ImNd<float, CUDAContext, StorageOrder::NCHW>(
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
    CUDAContext* context) {
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(
      N,
      Col2ImNdNCHWCUDAImpl,
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
void CopyMatrix<CUDAContext>(
    const size_t itemsize,
    const int M,
    const int N,
    const void* A,
    const int lda,
    void* B,
    const int ldb,
    CUDAContext* context,
    TypeMeta::TypedCopy copy) {
  CAFFE_ENFORCE(!copy, "Copy constructor is not supported in CUDA context");
  cudaMemcpy2DAsync(
      B,
      ldb * itemsize,
      A,
      lda * itemsize,
      N * itemsize,
      M,
      cudaMemcpyDeviceToDevice,
      context->cuda_stream());
}

template <>
void CopyVector<float, CUDAContext>(
    const int N,
    const float* src,
    float* dst,
    CUDAContext* context) {
  if (src != dst && N > 0) {
    cudaMemcpyAsync(
        dst,
        src,
        sizeof(float) * N,
        cudaMemcpyDeviceToDevice,
        context->cuda_stream());
  }
}

namespace {

template <typename T>
using BlockReduce = cub::BlockReduce<T, CAFFE_CUDA_NUM_THREADS>;

template <typename T, class Reducer>
__global__ void RowwiseReduceKernel(
    const int rows,
    const int cols,
    const Reducer reducer,
    const T init,
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
      Y[i] = val;
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
      Y[i] = val;
    }
    __syncthreads();
  }
}

} // namespace

#define CAFFE2_SPECIALIZED_CUDA_ROWWISE_MAX(T)                            \
  template <>                                                             \
  void RowwiseMax<T, CUDAContext>(                                        \
      const int N, const int D, const T* x, T* y, CUDAContext* context) { \
    RowwiseReduceKernel<<<                                                \
        std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),                            \
        CAFFE_CUDA_NUM_THREADS,                                           \
        0,                                                                \
        context->cuda_stream()>>>(                                        \
        N, D, cub::Max(), std::numeric_limits<T>::lowest(), x, y);        \
  }
CAFFE2_SPECIALIZED_CUDA_ROWWISE_MAX(float)
#undef CAFFE2_SPECIALIZED_CUDA_ROWWISE_MAX

#define CAFFE2_SPECIALIZED_CUDA_COLWISE_MAX(T)                            \
  template <>                                                             \
  void ColwiseMax<T, CUDAContext>(                                        \
      const int N, const int D, const T* x, T* y, CUDAContext* context) { \
    ColwiseReduceKernel<<<                                                \
        std::min(D, CAFFE_MAXIMUM_NUM_BLOCKS),                            \
        CAFFE_CUDA_NUM_THREADS,                                           \
        0,                                                                \
        context->cuda_stream()>>>(                                        \
        N, D, cub::Max(), std::numeric_limits<T>::lowest(), x, y);        \
  }
CAFFE2_SPECIALIZED_CUDA_COLWISE_MAX(float)
#undef CAFFE2_SPECIALIZED_CUDA_COLWISE_MAX

namespace {
__global__ void
maximum_kernel(const int N, const float alpha, const float* x, float* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
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
    CUDAContext* context) {
  maximum_kernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(N, alpha, x, y);
}

namespace {

std::vector<int> MakeTransposeAxes(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes) {
  std::vector<int> transpose_axes(num_dims);
  const int d = num_dims - num_axes;
  std::copy_n(axes, num_axes, transpose_axes.begin() + d);
  std::sort(transpose_axes.begin() + d, transpose_axes.end());
  int p = 0;
  int q = d;
  for (int i = 0; i < num_dims; ++i) {
    if (q < num_dims && i == transpose_axes[q]) {
      ++q;
    } else {
      transpose_axes[p++] = i;
    }
  }
  return transpose_axes;
}

template <int D>
void ComputeTransposedStrides(
    const int* X_dims,
    const int* axes,
    int* X_strides) {
  int buff[D];
  int cur_stride = 1;
  for (int i = D - 1; i >= 0; --i) {
    buff[i] = cur_stride;
    cur_stride *= X_dims[i];
  }
  for (int i = 0; i < D; ++i) {
    X_strides[i] = buff[axes[i]];
  }
}

template <typename T, class Reducer, int D>
__global__ void ReduceTensorCUDAKernel(
    const int outer_size,
    const int inner_size,
    SimpleArray<int, D> X_strides,
    SimpleArray<int, D> Y_dims,
    const Reducer reducer,
    const T init,
    const T* X,
    T* Y) {
  __shared__ typename BlockReduce<T>::TempStorage temp_storage;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T val = init;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      int X_index = 0;
      int Y_index = i * inner_size + j;
#pragma unroll
      for (int i = D - 1; i >= 0; --i) {
        X_index += (Y_index % Y_dims.data[i]) * X_strides.data[i];
        Y_index /= Y_dims.data[i];
      }
#if __CUDA_ARCH__ >= 350
      val = reducer(val, __ldg(X + X_index));
#else
      val = reducer(val, X[X_index]);
#endif
    }
    val = BlockReduce<T>(temp_storage).Reduce(val, reducer);
    if (threadIdx.x == 0) {
      Y[i] = val;
    }
    __syncthreads();
  }
}

template <typename T, class Reducer, int D>
void ReduceTensorCUDAImpl(
    const int outer_size,
    const int inner_size,
    const int* dims,
    const int* axes,
    const Reducer& reducer,
    const T& init,
    const T* X,
    T* Y,
    CUDAContext* context) {
  SimpleArray<int, D> X_strides;
  SimpleArray<int, D> Y_dims;
  ComputeTransposedStrides<D>(dims, axes, X_strides.data);
  for (int i = 0; i < D; ++i) {
    Y_dims.data[i] = dims[axes[i]];
  }
  ReduceTensorCUDAKernel<T, Reducer, D>
      <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          outer_size, inner_size, X_strides, Y_dims, reducer, init, X, Y);
}

template <typename T, class Reducer>
void ReduceTensorCUDA(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const Reducer& reducer,
    const T& init,
    const T* X,
    T* Y,
    CUDAContext* context) {
  CAFFE_ENFORCE_LE(num_axes, num_dims);
  const std::vector<int> transpose_axes =
      MakeTransposeAxes(num_dims, dims, num_axes, axes);
  const int pivot = num_dims - num_axes;
  int outer_size = 1;
  for (int i = 0; i < pivot; ++i) {
    outer_size *= dims[transpose_axes[i]];
  }
  int inner_size = 1;
  for (int i = pivot; i < num_dims; ++i) {
    inner_size *= dims[transpose_axes[i]];
  }
  if (transpose_axes[pivot] == pivot) {
    RowwiseReduceKernel<T>
        <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context->cuda_stream()>>>(
            outer_size, inner_size, reducer, init, X, Y);
    return;
  }
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_2(
      num_dims,
      ReduceTensorCUDAImpl,
      T,
      Reducer,
      outer_size,
      inner_size,
      dims,
      transpose_axes.data(),
      reducer,
      init,
      X,
      Y,
      context);
}

template <typename T>
void ReduceMeanCUDAImpl(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const T* X,
    T* Y,
    CUDAContext* context) {
  ReduceTensorCUDA(
      num_dims, dims, num_axes, axes, cub::Sum(), T(0), X, Y, context);
  const int X_size =
      std::accumulate(dims, dims + num_dims, 1, std::multiplies<int>());
  int scale = 1;
  for (int i = 0; i < num_axes; ++i) {
    scale *= dims[axes[i]];
  }
  const int Y_size = X_size / scale;
  Scale<T, CUDAContext>(
      Y_size, 1.0f / static_cast<float>(scale), Y, Y, context);
}

} // namespace

#define CAFFE2_SPECIALIZED_CUDA_REDUCE_MIN(T) \
  template <>                                 \
  void ReduceMin<T, CUDAContext>(             \
      const int num_dims,                     \
      const int* dims,                        \
      const int num_axes,                     \
      const int* axes,                        \
      const T* X,                             \
      T* Y,                                   \
      CUDAContext* context) {                 \
    ReduceTensorCUDA(                         \
        num_dims,                             \
        dims,                                 \
        num_axes,                             \
        axes,                                 \
        cub::Min(),                           \
        std::numeric_limits<T>::max(),        \
        X,                                    \
        Y,                                    \
        context);                             \
  }
CAFFE2_SPECIALIZED_CUDA_REDUCE_MIN(std::int32_t)
CAFFE2_SPECIALIZED_CUDA_REDUCE_MIN(std::int64_t)
CAFFE2_SPECIALIZED_CUDA_REDUCE_MIN(float)
CAFFE2_SPECIALIZED_CUDA_REDUCE_MIN(double)
#undef CAFFE2_SPECIALIZED_CUDA_REDUCE_MIN

#define CAFFE2_SPECIALIZED_CUDA_REDUCE_MAX(T) \
  template <>                                 \
  void ReduceMax<T, CUDAContext>(             \
      const int num_dims,                     \
      const int* dims,                        \
      const int num_axes,                     \
      const int* axes,                        \
      const T* X,                             \
      T* Y,                                   \
      CUDAContext* context) {                 \
    ReduceTensorCUDA(                         \
        num_dims,                             \
        dims,                                 \
        num_axes,                             \
        axes,                                 \
        cub::Max(),                           \
        std::numeric_limits<T>::lowest(),     \
        X,                                    \
        Y,                                    \
        context);                             \
  }
CAFFE2_SPECIALIZED_CUDA_REDUCE_MAX(std::int32_t)
CAFFE2_SPECIALIZED_CUDA_REDUCE_MAX(std::int64_t)
CAFFE2_SPECIALIZED_CUDA_REDUCE_MAX(float)
CAFFE2_SPECIALIZED_CUDA_REDUCE_MAX(double)
#undef CAFFE2_SPECIALIZED_CUDA_REDUCE_MAX

#define CAFFE2_SPECIALIZED_CUDA_REDUCE_SUM(T)                             \
  template <>                                                             \
  void ReduceSum<T, CUDAContext>(                                         \
      const int num_dims,                                                 \
      const int* dims,                                                    \
      const int num_axes,                                                 \
      const int* axes,                                                    \
      const T* X,                                                         \
      T* Y,                                                               \
      CUDAContext* context) {                                             \
    ReduceTensorCUDA(                                                     \
        num_dims, dims, num_axes, axes, cub::Sum(), T(0), X, Y, context); \
  }
CAFFE2_SPECIALIZED_CUDA_REDUCE_SUM(std::int32_t)
CAFFE2_SPECIALIZED_CUDA_REDUCE_SUM(std::int64_t)
CAFFE2_SPECIALIZED_CUDA_REDUCE_SUM(float)
CAFFE2_SPECIALIZED_CUDA_REDUCE_SUM(double)
#undef CAFFE2_SPECIALIZED_CUDA_REDUCE_SUM

#define CAFFE2_SPECIALIZED_CUDA_REDUCE_MEAN(T)                            \
  template <>                                                             \
  void ReduceMean<T, CUDAContext>(                                        \
      const int num_dims,                                                 \
      const int* dims,                                                    \
      const int num_axes,                                                 \
      const int* axes,                                                    \
      const T* X,                                                         \
      T* Y,                                                               \
      CUDAContext* context) {                                             \
    ReduceMeanCUDAImpl<T>(num_dims, dims, num_axes, axes, X, Y, context); \
  }
CAFFE2_SPECIALIZED_CUDA_REDUCE_MEAN(float)
#undef CAFFE2_SPECIALIZED_CUDA_REDUCE_MEAN

namespace {

template <typename T, int D>
__global__ void BroadcastCUDAKernel(
    const int Y_size,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const T* X,
    T* Y) {
  CUDA_1D_KERNEL_LOOP(Y_index, Y_size) {
    int X_index = 0;
    int Y_index_val = Y_index;
#pragma unroll
    for (int i = D - 1; i >= 0; --i) {
      X_index += X_strides.data[i] == 0
          ? 0
          : (Y_index_val % Y_dims.data[i]) * X_strides.data[i];
      Y_index_val /= Y_dims.data[i];
    }
#if __CUDA_ARCH__ >= 350
    Y[Y_index] = __ldg(X + X_index);
#else
    Y[Y_index] = X[X_index];
#endif
  }
}

template <typename T, int D>
void BroadcastCUDAImpl(
    const int X_ndim,
    const int* X_dims,
    const int* Y_dims,
    const T* X,
    T* Y,
    CUDAContext* context) {
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
  BroadcastCUDAKernel<T, D>
      <<<CAFFE_GET_BLOCKS(Y_size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(Y_size, X_strides_array, Y_dims_array, X, Y);
}

} // namespace

#define CAFFE2_SPECIALIZED_CUDA_BROADCAST(T)                                  \
  template <>                                                                 \
  void Broadcast<T, CUDAContext>(                                             \
      const int X_ndim,                                                       \
      const int* X_dims,                                                      \
      const int Y_ndim,                                                       \
      const int* Y_dims,                                                      \
      const T* X,                                                             \
      T* Y,                                                                   \
      CUDAContext* context) {                                                 \
    CAFFE_ENFORCE_LE(X_ndim, Y_ndim);                                         \
    DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(                                   \
        Y_ndim, BroadcastCUDAImpl, T, X_ndim, X_dims, Y_dims, X, Y, context); \
  }
CAFFE2_SPECIALIZED_CUDA_BROADCAST(std::int32_t)
CAFFE2_SPECIALIZED_CUDA_BROADCAST(std::int64_t)
CAFFE2_SPECIALIZED_CUDA_BROADCAST(float)
CAFFE2_SPECIALIZED_CUDA_BROADCAST(double)
#undef CAFFE2_SPECIALIZED_CUDA_BROADCAST

namespace {

template <typename T>
__global__ void RowwiseMomentsCUDAKernel(
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
#if __CUDA_ARCH__ >= 350
      m_val += __ldg(X + X_index);
      v_val += __ldg(X + X_index) * __ldg(X + X_index);
#else
      m_val += X[X_index];
      v_val += X[X_index] * X[X_index];
#endif
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
__global__ void MomentsCUDAKernel(
    const int outer_size,
    const int inner_size,
    SimpleArray<int, D> X_strides,
    SimpleArray<int, D> Y_dims,
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
      for (int i = D - 1; i >= 0; --i) {
        X_index += (Y_index % Y_dims.data[i]) * X_strides.data[i];
        Y_index /= Y_dims.data[i];
      }
#if __CUDA_ARCH__ >= 350
      m_val += __ldg(X + X_index);
      v_val += __ldg(X + X_index) * __ldg(X + X_index);
#else
      m_val += X[X_index];
      v_val += X[X_index] * X[X_index];
#endif
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
void MomentsCUDAImpl(
    const int outer_size,
    const int inner_size,
    const int* dims,
    const int* axes,
    const T* X,
    T* mean,
    T* variance,
    CUDAContext* context) {
  SimpleArray<int, D> X_strides;
  SimpleArray<int, D> Y_dims;
  ComputeTransposedStrides<D>(dims, axes, X_strides.data);
  for (int i = 0; i < D; ++i) {
    Y_dims.data[i] = dims[axes[i]];
  }
  MomentsCUDAKernel<T, D>
      <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          outer_size, inner_size, X_strides, Y_dims, X, mean, variance);
}

template <typename T>
void MomentsCUDA(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const T* X,
    T* mean,
    T* variance,
    CUDAContext* context) {
  CAFFE_ENFORCE_LE(num_axes, num_dims);
  const std::vector<int> transpose_axes =
      MakeTransposeAxes(num_dims, dims, num_axes, axes);
  const int pivot = num_dims - num_axes;
  int outer_size = 1;
  for (int i = 0; i < pivot; ++i) {
    outer_size *= dims[transpose_axes[i]];
  }
  int inner_size = 1;
  for (int i = pivot; i < num_dims; ++i) {
    inner_size *= dims[transpose_axes[i]];
  }
  if (transpose_axes[pivot] == pivot) {
    RowwiseMomentsCUDAKernel<T>
        <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context->cuda_stream()>>>(outer_size, inner_size, X, mean, variance);
    return;
  }
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(
      num_dims,
      MomentsCUDAImpl,
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

} // namespace

#define CAFFE2_SPECIALIZED_CUDA_MOMENTS(T)                           \
  template <>                                                        \
  void Moments<T, CUDAContext>(                                      \
      const int num_dims,                                            \
      const int* dims,                                               \
      const int num_axes,                                            \
      const int* axes,                                               \
      const T* X,                                                    \
      T* mean,                                                       \
      T* variance,                                                   \
      CUDAContext* context) {                                        \
    MomentsCUDA<T>(                                                  \
        num_dims, dims, num_axes, axes, X, mean, variance, context); \
  }
CAFFE2_SPECIALIZED_CUDA_MOMENTS(float)
#undef CAFFE2_SPECIALIZED_CUDA_MOMENTS

namespace {

template <typename T, int D>
__global__ void TransposeCUDAKernel(
    const int size,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const T* X,
    T* Y) {
  CUDA_1D_KERNEL_LOOP(Y_index, size) {
    int X_index = 0;
    int Y_index_val = Y_index;
#pragma unroll
    for (int i = D - 1; i >= 0; --i) {
      X_index += (Y_index_val % Y_dims.data[i]) * X_strides.data[i];
      Y_index_val /= Y_dims.data[i];
    }
#if __CUDA_ARCH__ >= 350
    Y[Y_index] = __ldg(X + X_index);
#else
    Y[Y_index] = X[X_index];
#endif
  }
}

template <typename T, int D>
void TransposeCUDAImpl(
    const int* dims,
    const int* axes,
    const T* X,
    T* Y,
    CUDAContext* context) {
  SimpleArray<int, D> X_strides;
  SimpleArray<int, D> Y_dims;
  ComputeTransposedStrides<D>(dims, axes, X_strides.data);
  int size = 1;
  for (int i = 0; i < D; ++i) {
    Y_dims.data[i] = dims[axes[i]];
    size *= dims[i];
  }
  TransposeCUDAKernel<T, D>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(size, X_strides, Y_dims, X, Y);
}

} // namespace

#define CAFFE2_SPECIALIZED_CUDA_TRANSPOSE(T)                    \
  template <>                                                   \
  void Transpose<T, CUDAContext>(                               \
      const int ndim,                                           \
      const int* dims,                                          \
      const int* axes,                                          \
      const T* X,                                               \
      T* Y,                                                     \
      CUDAContext* context) {                                   \
    DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(                     \
        ndim, TransposeCUDAImpl, T, dims, axes, X, Y, context); \
  }
CAFFE2_SPECIALIZED_CUDA_TRANSPOSE(float)
CAFFE2_SPECIALIZED_CUDA_TRANSPOSE(double)
CAFFE2_SPECIALIZED_CUDA_TRANSPOSE(int)
CAFFE2_SPECIALIZED_CUDA_TRANSPOSE(TIndex)
#undef CAFFE2_SPECIALIZED_CUDA_TRANSPOSE

} // namespace math
} // namespace caffe2
