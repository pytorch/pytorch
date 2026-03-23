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

  void* mArray;
  void* nArray;
  void* kArray;
  void* ldaArray;
  void* ldbArray;
  void* lddArray;
  void* APtrArray;
  void* BPtrArray;
  void* DPtrArray;
  void* alphaPtrArray;
  void* betaPtrArray;
};
#endif // !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13020

} // namespace at::native
