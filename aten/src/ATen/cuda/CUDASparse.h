#pragma once

#include <ATen/cuda/CUDAContext.h>
#if defined(USE_ROCM)
#include <hipsparse/hipsparse-version.h>
#define HIPSPARSE_VERSION ((hipsparseVersionMajor*100000) + (hipsparseVersionMinor*100) + hipsparseVersionPatch)
#endif

#include <optional>


// cuSparse Generic API spsv function was added in CUDA 11.3.0
// hipSparse supports SpSV as well
#if (defined(CUDART_VERSION) && defined(CUSPARSE_VERSION)) || defined(USE_ROCM)
#define AT_USE_CUSPARSE_GENERIC_SPSV() 1
#else
#define AT_USE_CUSPARSE_GENERIC_SPSV() 0
#endif

// cuSparse Generic API spsm function was added in CUDA 11.3.1
// hipSparse supports SpSM as well
#if (defined(CUDART_VERSION) && defined(CUSPARSE_VERSION)) || defined(USE_ROCM)
#define AT_USE_CUSPARSE_GENERIC_SPSM() 1
#else
#define AT_USE_CUSPARSE_GENERIC_SPSM() 0
#endif

#if !defined(USE_ROCM)
namespace at::cuda::sparse {

// See Note [Enabling Deterministic Operations]
// Returns the cusparseSpMM algorithm that gives bit-wise reproducible results
// for a CSR matrix A, or nullopt when cuSPARSE has none for this call.
// cuSPARSE documents deterministic SpMM results only for
// CUSPARSE_SPMM_CSR_ALG3 and CUSPARSE_SPMM_COO_ALG2 (and BSR_ALG1), and only
// for opA == CUSPARSE_OPERATION_NON_TRANSPOSE. CSR_ALG3 keeps A in CSR
// format, but it does not support batched matrices,
// opB == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE or complex half values, and
// it is not deterministic before cuSPARSE 12.7 (CUDA 13.1; known issue in the
// CUDA 13.0 release notes, fixed as CUSPARSE-2612). When COO_ALG2 is returned
// the caller has to describe A in COO format.
inline std::optional<cusparseSpMMAlg_t> deterministic_spmm_alg(
    cusparseOperation_t opA,
    cusparseOperation_t opB,
    cudaDataType value_type,
    bool batched) {
  if (opA != CUSPARSE_OPERATION_NON_TRANSPOSE) {
    return std::nullopt;
  }
#if CUSPARSE_VERSION >= 12700
  if (!batched && opB != CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE &&
      value_type != CUDA_C_16F && value_type != CUDA_C_16BF) {
    return CUSPARSE_SPMM_CSR_ALG3;
  }
#endif
  return CUSPARSE_SPMM_COO_ALG2;
}

} // namespace at::cuda::sparse
#endif
