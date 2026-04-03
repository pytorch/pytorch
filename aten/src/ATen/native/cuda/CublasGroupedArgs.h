#pragma once

#include <cuda.h>

#include <ATen/BlasBackend.h>
#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>

namespace at::native {

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13020
struct cublasGroupedArgs {
  cublasGroupedArgs(
      const Tensor& mat1,
      const Tensor& mat2,
      const std::optional<Tensor>& offs,
      Tensor& c);

  char transa, transb;
  int64_t avgM, avgN, avgK;
  ScalarType A_dtype, B_dtype, result_dtype;
  int batchCount;

  // All arrays live in a single device allocation
  Tensor buf;

  int32_t* mArray;
  int32_t* nArray;
  int32_t* kArray;
  int32_t* ldaArray;
  int32_t* ldbArray;
  int32_t* lddArray;
  int64_t* APtrArray;
  int64_t* BPtrArray;
  int64_t* DPtrArray;
  int64_t* alphaPtrArray;
  int64_t* betaPtrArray;
};
#endif // !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13020

} // namespace at::native
