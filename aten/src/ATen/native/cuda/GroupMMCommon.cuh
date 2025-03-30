#pragma once
#include <cutlass/util/packed_stride.hpp>

namespace at::cuda::detail {

using Strides = std::array<int64_t, 3>;

template <
    typename DtypeA,
    typename DtypeB,
    typename DtypeOutput,
    typename DtypeScale,
    typename ProblemShape,
    typename StrideA,
    typename StrideB,
    typename StrideOutput>
__global__ void prepare_grouped_gemm_data(
    DtypeA* A,
    DtypeB* B,
    DtypeOutput* output,
    DtypeScale* scale_A,
    DtypeScale* scale_B,
    DtypeA** A_ptrs,
    DtypeB** B_ptrs,
    DtypeOutput** output_ptrs,
    DtypeScale** inputA_scale_ptrs,
    DtypeScale** inputB_scale_ptrs,
    ProblemShape* problem_sizes,
    // Strides for cutlass, cute::Stride
    StrideA* stride_A,
    StrideB* stride_B,
    StrideOutput* stride_output,
    const int32_t* offs,
    int32_t M,
    int32_t N,
    int32_t K,
    // Original strides of the input tensors
    Strides tensor_StrideA,
    Strides tensor_StrideB,
    Strides tensor_StrideOutput,
    int64_t a_scale_stride,
    int64_t b_scale_stride) {
  int32_t tid = threadIdx.x;
  int32_t delta = 0;
  if (offs != nullptr) {
    int32_t start = tid == 0 ? 0 : offs[tid - 1];
    delta = offs[tid] - start;
    CUDA_KERNEL_ASSERT(
        delta % 16 == 0 && "expected dynamic dimension to be multiple of 16\n");
  }
  int64_t lda, ldb, ldoutput;
  if (M < 0) {
    // A and output is 2d
    M = delta;
    lda = tensor_StrideA[0];
    ldb = tensor_StrideB[2]; // B is transposed
    ldoutput = tensor_StrideOutput[0];
    A_ptrs[tid] = tid == 0 ? A : A + offs[tid - 1] * lda;
    if (scale_A != nullptr) {
      inputA_scale_ptrs[tid] = tid == 0 ? scale_A : scale_A + offs[tid - 1];
      inputB_scale_ptrs[tid] = scale_B + tid * b_scale_stride;
    }
    output_ptrs[tid] = tid == 0 ? output : output + offs[tid - 1] * ldoutput;
    B_ptrs[tid] = B + tid * tensor_StrideB[0];
  } else if (N < 0) {
    N = delta;
    lda = tensor_StrideA[1];
    ldb = tensor_StrideB[1]; // B is transposed
    ldoutput = tensor_StrideOutput[0];
    A_ptrs[tid] = A + tid * tensor_StrideA[0];
    output_ptrs[tid] = tid == 0 ? output : output + offs[tid - 1];
    B_ptrs[tid] = tid == 0 ? B : B + offs[tid - 1] * ldb;
    if (scale_A != nullptr) {
      inputA_scale_ptrs[tid] = scale_A + tid * a_scale_stride;
      inputB_scale_ptrs[tid] = tid == 0 ? scale_B : scale_B + offs[tid - 1];
    }
  } else if (K < 0) {
    // A, B is 2d, output is 3d
    K = delta;
    lda = tensor_StrideA[0];
    ldb = tensor_StrideB[1]; // B is transposed
    ldoutput = tensor_StrideOutput[1];
    A_ptrs[tid] = tid == 0 ? A : A + offs[tid - 1];
    B_ptrs[tid] = tid == 0 ? B : B + offs[tid - 1];
    output_ptrs[tid] = output + tid * tensor_StrideOutput[0];
    if (scale_A != nullptr) {
      inputA_scale_ptrs[tid] = scale_A + tid * M;
      inputB_scale_ptrs[tid] = scale_B + tid * N;
    }
  } else {
    // A, B, output are 3D
    lda = tensor_StrideA[1];
    ldb = tensor_StrideB[2];
    ldoutput = tensor_StrideOutput[1];
    A_ptrs[tid] = A + tid * tensor_StrideA[0];
    B_ptrs[tid] = B + tid * tensor_StrideB[0];
    output_ptrs[tid] = output + tid * tensor_StrideOutput[0];
    if (scale_A != nullptr) {
      inputA_scale_ptrs[tid] = scale_A + tid * a_scale_stride;
      inputB_scale_ptrs[tid] = scale_B + tid * b_scale_stride;
    }
  }
  problem_sizes[tid] = ProblemShape(M, N, K);

  stride_A[tid] = cutlass::make_cute_packed_stride(StrideA{}, {M, lda, 1});
  stride_B[tid] = cutlass::make_cute_packed_stride(StrideB{}, {N, ldb, 1});
  stride_output[tid] =
      cutlass::make_cute_packed_stride(StrideOutput{}, {M, ldoutput, 1});
}
} // namespace at::cuda::detail
