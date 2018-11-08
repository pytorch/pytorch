#include "common_fpga.h"
#include "context_intel_fpga.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace math {

namespace {}

// The gemm call implements the following operation:
//
//                  C = alpha * op(A) * op(B) + beta * C
//
// where op(A) has size M x K, op(B) has size K x N, and C has size M x N. Each
// of A, B, and C are matrices and alpha and beta are scalars. Note that the
// most common use case of gemm will involve setting alpha to 1 and beta to 0.
//
// op(A) and op(B) represent the transformations that are done to A and B before
// the matrix multiply; depending on the flags set, op(A) is equal to A or A^T
// (transpose) if the argument TransA or TransB is set to CblasNoTrans or
// CblasTrans, respectively, for each of A and B.

// OPENCL kernels provided by Intel, please double check if considering open
// sourcing
template <>
void Gemm<float, OpenCLContext, FPGAEngine>(
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
    OpenCLContext* context,
    TensorProto::DataType math_type) {
  FPGAContextSingleton ctx = *static_cast<FPGAContextSingleton*>(
      context->GetSingleton(FPGAEngine::name));

  VLOG(1) << "RUNNING Gemm " << M << " " << K << " " << N << " "
          << " ab: " << alpha << " " << beta;
  VLOG(1) << "A " << (TransA != CblasNoTrans);
  // ctx.printBuffer((cl::Buffer*)A, M, K, M, K);
  VLOG(1) << "B " << (TransB != CblasNoTrans);
  // ctx.printBuffer((cl::Buffer*)B, K, N, K, N);

  CAFFE_ENFORCE(alpha == 1.0 || alpha == 0.0);
  CAFFE_ENFORCE(beta == 1.0 || beta == 0.0);

  // Invoke C = alpha * A * B + beta * C
  // Normal mode
  // C(M, N) =  A(M, K) * B(K, N)
  // B_T   C(M, N) = A(M, K) * B(N, K)_T
  // A_T   C(M, N) = A(K, M)_T * B(K, N)
  // A_T, B_T C(M, N) = A(K, M)_T * B(N, K)_T
  assert(alpha == 1.0);
  if (beta == 0.0) {
    if (TransA == CblasNoTrans) {
      // A*B or A*B_T
      ctx.MatMul(
          false,
          TransB != CblasNoTrans,
          A,
          nullptr,
          B,
          C,
          M,
          K,
          N,
          false,
          false);
    } else {
      // A_T * B or A_T * B_T
      auto A_T = ctx.transposeBuffer((cl::Buffer*)A, K, M);
      ctx.MatMul(
          false,
          TransB != CblasNoTrans,
          (const float*)A_T,
          nullptr,
          B,
          C,
          M,
          K,
          N,
          false,
          false);
      delete A_T;
    }
  } else if (beta == 1.0) {
    CAFFE_ENFORCE_EQ(TransA, CblasNoTrans);
    ctx.MatMulAccum(false, TransB != CblasNoTrans, A, B, C, M, K, N);
  } else {
    CAFFE_ENFORCE(false, "not supported");
  }
}

template <>
void Gemm<float, OpenCLContext>(
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
    OpenCLContext* context,
    TensorProto::DataType math_type) {
  Gemm<float, OpenCLContext, FPGAEngine>(
      TransA, TransB, M, N, K, alpha, A, B, beta, C, context, math_type);
}

template <>
void Gemv<float, OpenCLContext, FPGAEngine>(
    const CBLAS_TRANSPOSE TransA,
    const int M,
    const int N,
    const float alpha,
    const float* A,
    const float* x,
    const float beta,
    float* y,
    OpenCLContext* context,
    TensorProto::DataType math_type) {
  FPGAContextSingleton ctx = *static_cast<FPGAContextSingleton*>(
      context->GetSingleton(FPGAEngine::name));

  VLOG(1) << "RUNNING Gemv " << M << " " << N;
  CAFFE_ENFORCE_EQ(alpha, 1.0);
  CAFFE_ENFORCE_EQ(beta, 0.0);

  ctx.MatVecMul(TransA != CblasNoTrans, A, x, y, M, N);
}

template <>
void Gemv<float, OpenCLContext>(
    const CBLAS_TRANSPOSE TransA,
    const int M,
    const int N,
    const float alpha,
    const float* A,
    const float* x,
    const float beta,
    float* y,
    OpenCLContext* context,
    TensorProto::DataType math_type) {
  Gemv<float, OpenCLContext, FPGAEngine>(
      TransA, M, N, alpha, A, x, beta, y, context, math_type);
}

// TODO: Set for all types
//#define CAFFE2_SPECIALIZED_OPENCL_SET(T)
template <>
void Set<float, OpenCLContext>(
    const size_t N,
    const float alpha,
    float* Y,
    OpenCLContext* context) {
  if (!Y) {
    VLOG(1) << "Allocating memory for new set of size " << N;
    Y = (float*)(OpenCLContext::New(N * sizeof(bfloat16)).get());
  }
  assert(Y);

  // Actually create a bfloat16 set of 1s
  bfloat16* tmp = new bfloat16[N];
  union bfp_converter x;
  for (int i = 0; i < N; i++) {
    x.fp32 = alpha;
    tmp[i] = x.bfp[1];
  }
  auto& ctx = *(FPGAContextSingleton*)(context->GetSingleton("FPGA"));
  ctx.writeBuffer(tmp, N, 1, (cl::Buffer*)Y);

  delete[] tmp;
}

// CAFFE2_SPECIALIZED_OPENCL_SET(float);
// CAFFE2_SPECIALIZED_OPENCL_SET(double);
// CAFFE2_SPECIALIZED_OPENCL_SET(bool);
// CAFFE2_SPECIALIZED_OPENCL_SET(int8_t);
// CAFFE2_SPECIALIZED_OPENCL_SET(int16_t);
// CAFFE2_SPECIALIZED_OPENCL_SET(at::Half);
// CAFFE2_SPECIALIZED_OPENCL_SET(int);
// CAFFE2_SPECIALIZED_OPENCL_SET(int64_t);
// CAFFE2_SPECIALIZED_OPENCL_SET(char);
// CAFFE2_SPECIALIZED_OPENCL_SET(uint8_t);
// CAFFE2_SPECIALIZED_OPENCL_SET(uint16_t);
#undef CAFFE2_SPECIALIZED_OPENCL_SET

} // namespace math
} // namespace caffe2
