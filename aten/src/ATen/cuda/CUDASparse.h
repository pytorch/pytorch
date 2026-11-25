#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#if defined(USE_ROCM)
#include <hipsparse/hipsparse-version.h>
#define HIPSPARSE_VERSION ((hipsparseVersionMajor*100000) + (hipsparseVersionMinor*100) + hipsparseVersionPatch)
#endif


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

// Version of the loaded cuSPARSE library, encoded as CUSPARSE_VERSION. It can
// differ from the version of the headers PyTorch was built with.
inline int cusparse_runtime_version() {
  static const int version = [] {
    int major = 0, minor = 0, patch = 0;
    TORCH_CUDASPARSE_CHECK(cusparseGetProperty(MAJOR_VERSION, &major));
    TORCH_CUDASPARSE_CHECK(cusparseGetProperty(MINOR_VERSION, &minor));
    TORCH_CUDASPARSE_CHECK(cusparseGetProperty(PATCH_LEVEL, &patch));
    return major * 1000 + minor * 100 + patch;
  }();
  return version;
}

// See Note [Enabling Deterministic Operations]
// cuSPARSE documents bit-wise reproducible cusparseSpMM results only for
// CUSPARSE_SPMM_CSR_ALG3, CUSPARSE_SPMM_COO_ALG2 and CUSPARSE_SPMM_BSR_ALG1,
// with opA == CUSPARSE_OPERATION_NON_TRANSPOSE. CSR_ALG3 does not support
// batched matrices, opB == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE or complex
// half values, and it is only deterministic from cuSPARSE 12.7 (CUDA 13.1,
// CUSPARSE-2612). That depends on the loaded library, not on the headers, so
// the version is checked at runtime. COO_ALG2 needs A described in COO format.
inline cusparseSpMMAlg_t deterministic_spmm_alg(
    cusparseOperation_t opA,
    cusparseOperation_t opB,
    cudaDataType value_type,
    bool batched) {
  TORCH_INTERNAL_ASSERT(opA == CUSPARSE_OPERATION_NON_TRANSPOSE);
  if (!batched && opB != CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE &&
      value_type != CUDA_C_16F && value_type != CUDA_C_16BF &&
      cusparse_runtime_version() >= 12700) {
    return CUSPARSE_SPMM_CSR_ALG3;
  }
  return CUSPARSE_SPMM_COO_ALG2;
}

} // namespace at::cuda::sparse
#endif
