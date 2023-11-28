// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/tunable/TunableOp.h>
#include <ATen/cuda/tunable/GemmCommon.h>
#include <c10/util/StringUtil.h>

#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>

#define TORCH_ROCBLAS_CHECK(EXPR)                 \
  do {                                            \
    rocblas_status __err = EXPR;                  \
    TORCH_CHECK(__err == rocblas_status_success,  \
                "rocblas error: ",                \
                rocblas_status_to_string(__err),  \
                " when calling `" #EXPR "`");     \
  } while (0)

namespace at::cuda::tunable {

template <typename T>
constexpr rocblas_datatype RocBlasDataTypeFor();

template <>
constexpr rocblas_datatype RocBlasDataTypeFor<float>() {
  return rocblas_datatype_f32_r;
}

template <>
constexpr rocblas_datatype RocBlasDataTypeFor<double>() {
  return rocblas_datatype_f64_r;
}

template <>
constexpr rocblas_datatype RocBlasDataTypeFor<Half>() {
  return rocblas_datatype_f16_r;
}

template <>
constexpr rocblas_datatype RocBlasDataTypeFor<BFloat16>() {
  return rocblas_datatype_bf16_r;
}

template <>
constexpr rocblas_datatype RocBlasDataTypeFor<c10::complex<float>>() {
  return rocblas_datatype_f32_c;
}

template <>
constexpr rocblas_datatype RocBlasDataTypeFor<c10::complex<double>>() {
  return rocblas_datatype_f64_c;
}

template <typename T>
constexpr rocblas_datatype RocBlasComputeTypeFor();

template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<float>() {
  return rocblas_datatype_f32_r;
}

template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<double>() {
  return rocblas_datatype_f64_r;
}

template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<Half>() {
  // Note that we're returning the _compute_ type for a given datatype.
  // As of 12/2022, using compute type FP16 for 16-bit floats was much
  // slower than using compute type FP32. So we use FP32 compute even for
  // FP16 datatypes. This is how GEMM is implemented even in the function
  // rocblasGemmHelper (see fpgeneric.h)
  return rocblas_datatype_f32_r;
}

template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<BFloat16>() {
  // Note that we're returning the _compute_ type for a given datatype.
  // As of 12/2022, using compute type FP16 for 16-bit floats was much
  // slower than using compute type FP32. So we use FP32 compute even for
  // BF16 datatypes. This is how GEMM is implemented even in the function
  // rocblasGemmHelper (see fpgeneric.h)
  return rocblas_datatype_f32_r;
}

template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<c10::complex<float>>() {
  return rocblas_datatype_f32_c;
}

template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<c10::complex<double>>() {
  return rocblas_datatype_f64_c;
}

template <typename T>
auto DoCastForHalfOrBfloat16(const T fp) {
  return fp;
}

template <>
inline auto DoCastForHalfOrBfloat16<Half>(const Half fp) {
  // alpha and beta should be the same as compute_type, in Half case it is float.
  float h = fp;
  return h;
}

template <>
inline auto DoCastForHalfOrBfloat16<BFloat16>(const BFloat16 fp) {
  // alpha and beta should be the same as compute_type, in bfloat16 case it is float.
  float h = fp;
  return h;
}

static rocblas_operation _rocblasOpFromChar(char op) {
  switch (op) {
    case 'n':
    case 'N':
      return rocblas_operation_none;
    case 't':
    case 'T':
      return rocblas_operation_transpose;
    case 'c':
    case 'C':
      return rocblas_operation_conjugate_transpose;
  }
  AT_ERROR(
      "_rocblasOpFromChar input should be 't', 'n' or 'c' but got `", op, "`");
}

template <typename T>
auto GetRocBlasGemmTypeStringAndOps() {
  rocblas_handle handle = (rocblas_handle)at::cuda::getCurrentCUDABlasHandle();

  int solution_size;
  auto input_output_type = RocBlasDataTypeFor<T>();
  auto compute_type = RocBlasComputeTypeFor<T>();

  // Get the number of available solutions
  TORCH_ROCBLAS_CHECK(rocblas_gemm_ex_get_solutions_by_type(handle,
                                                            input_output_type,
                                                            input_output_type,
                                                            compute_type,
                                                            rocblas_gemm_flags_none,
                                                            nullptr,
                                                            &solution_size));

  std::vector<int> solutions(solution_size);

  // Get the list of available solutions
  TORCH_ROCBLAS_CHECK(rocblas_gemm_ex_get_solutions_by_type(handle,
                                                            input_output_type,
                                                            input_output_type,
                                                            compute_type,
                                                            rocblas_gemm_flags_none,
                                                            solutions.data(),
                                                            &solution_size));


  // Sort the solutions in ascending order to make the solution vector deterministic across runs
  std::sort(solutions.begin(), solutions.end());

  std::vector<std::pair<std::string, Callable<GemmParams<T>>>> ret;
  for (size_t i = 0; i < solutions.size(); ++i) {
    auto solution = solutions[i];
    auto rocblas_gemm_op = [=](const GemmParams<T>* params) -> TuningStatus {
      auto h_a = DoCastForHalfOrBfloat16(params->alpha);
      auto h_b = DoCastForHalfOrBfloat16(params->beta);
      auto status = rocblas_gemm_ex(
          (rocblas_handle)at::cuda::getCurrentCUDABlasHandle(),
          _rocblasOpFromChar(params->transa),
          _rocblasOpFromChar(params->transb),
          params->m, params->n, params->k,
          &h_a,
          params->a, input_output_type, params->lda,
          params->b, input_output_type, params->ldb,
          &h_b,
          params->c, input_output_type, params->ldc,
          params->c, input_output_type, params->ldc,
          compute_type,
          rocblas_gemm_algo_solution_index,
          solution,
          rocblas_gemm_flags_none);

      if (status != rocblas_status_success) {
        //std::cerr << c10::str("[rocBLAS] Solution #", i, " (original ", solution, ") failed: ", rocblas_status_to_string(status)) << std::endl;
        return FAIL;
      }

      return OK;
    };
    ret.emplace_back(std::make_pair(
        c10::str("RocBlasGemm_", i, "_sol_", solution), std::move(rocblas_gemm_op)));
  }
  return ret;
}

// TODO batched and strided batched
#if 0

template <typename T>
auto GetRocBlasBatchedGemmTypeStringAndOps() {
  rocblas_handle handle = (rocblas_handle)at::cuda::getCurrentCUDABlasHandle();

  int solution_size;
  auto input_output_type = RocBlasDataTypeFor<T>();
  auto compute_type = RocBlasComputeTypeFor<T>();

  // Get the number of available solutions
  TORCH_ROCBLAS_CHECK(rocblas_gemm_batched_ex_get_solutions_by_type(handle,
                                                                   input_output_type,
                                                                   input_output_type,
                                                                   compute_type,
                                                                   rocblas_gemm_flags_none,
                                                                   nullptr,
                                                                   &solution_size));

  std::vector<int> solutions(solution_size);

  // Get the list of available solutions
  TORCH_ROCBLAS_CHECK(rocblas_gemm_batched_ex_get_solutions_by_type(handle,
                                                                   input_output_type,
                                                                   input_output_type,
                                                                   compute_type,
                                                                   rocblas_gemm_flags_none,
                                                                   solutions.data(),
                                                                   &solution_size));

  // Sort the solutions in ascending order to make the solution vector deterministic across runs
  std::sort(solutions.begin(), solutions.end());

  std::vector<std::pair<std::string, Callable<BatchedGemmParams<T>>>> ret;
  for (size_t i = 0; i < solutions.size(); ++i) {
    auto solution = solutions[i];
    auto rocblas_gemm_op = [=](const BatchedGemmParams<T>* params) -> TuningStatus {
      auto h_a = DoCastForHalfOrBfloat16(params->alpha);
      auto h_b = DoCastForHalfOrBfloat16(params->beta);
      auto status = rocblas_gemm_batched_ex(
          (rocblas_handle)at::cuda::getCurrentCUDABlasHandle(),
          params->opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
          params->opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
          params->n, params->m, params->k,
          &h_a,
          params->bs, input_output_type, params->ldb,
          params->as, input_output_type, params->lda,
          &h_b,
          params->cs, input_output_type, params->ldc,
          params->cs, input_output_type, params->ldc,
          params->batch,
          compute_type,
          rocblas_gemm_algo_solution_index,
          solution,
          rocblas_gemm_flags_none);

      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          status != rocblas_status_success,
          "[rocBLAS] Solution #", i, " (original ", solution, ") failed: ", rocblas_status_to_string(status));

      return OK;
    };
    ret.emplace_back(std::make_pair(
        c10::str("RocBlasBatchedGemm_", i, "_sol_", solution), std::move(rocblas_gemm_op)));
  }
  return ret;
}

template <typename T>
auto GetRocBlasStridedBatchedGemmTypeStringAndOps() {
  rocblas_handle handle = (rocblas_handle)at::cuda::getCurrentCUDABlasHandle();

  int solution_size;
  auto input_output_type = RocBlasDataTypeFor<T>();
  auto compute_type = RocBlasComputeTypeFor<T>();

  // Get the number of available solutions
  TORCH_ROCBLAS_CHECK(rocblas_gemm_ex_get_solutions_by_type(handle,
                                                           input_output_type,
                                                           input_output_type,
                                                           compute_type,
                                                           rocblas_gemm_flags_none,
                                                           nullptr,
                                                           &solution_size));

  std::vector<int> solutions(solution_size);

  // Get the list of available solutions
  TORCH_ROCBLAS_CHECK(rocblas_gemm_ex_get_solutions_by_type(handle,
                                                           input_output_type,
                                                           input_output_type,
                                                           compute_type,
                                                           rocblas_gemm_flags_none,
                                                           solutions.data(),
                                                           &solution_size));

  // Sort the solutions in ascending order to make the solution vector deterministic across runs
  std::sort(solutions.begin(), solutions.end());

  std::vector<std::pair<std::string, Callable<StridedBatchedGemmParams<T>>>> ret;
  for (size_t i = 0; i < solutions.size(); ++i) {
    auto solution = solutions[i];
    auto rocblas_gemm_op = [=](const StridedBatchedGemmParams<T>* params) -> TuningStatus {
      auto h_a = DoCastForHalfOrBfloat16(params->alpha);
      auto h_b = DoCastForHalfOrBfloat16(params->beta);
      auto status = rocblas_gemm_strided_batched_ex(
          (rocblas_handle)at::cuda::getCurrentCUDABlasHandle(),
          params->opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
          params->opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
          params->n, params->m, params->k,
          &h_a,
          params->b, input_output_type, params->ldb, params->stride_b,
          params->a, input_output_type, params->lda, params->stride_a,
          &h_b,
          params->c, input_output_type, params->ldc, params->stride_c,
          params->c, input_output_type, params->ldc, params->stride_c,
          params->batch,
          compute_type,
          rocblas_gemm_algo_solution_index,
          solution,
          rocblas_gemm_flags_none);

      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          status != rocblas_status_success,
          "[rocBLAS] Solution #", i, " (original ", solution, ") failed: ", rocblas_status_to_string(status));

      return OK;
    };
    ret.emplace_back(std::make_pair(
        c10::str("RocBlasStridedBatchedGemm_", i, "_sol_", solution), std::move(rocblas_gemm_op)));
  }
  return ret;
}

#endif

}  // namespace at::cuda::tunable
