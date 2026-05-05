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
      Tensor& c,
      int batchCount,
      bool needs_int64);

  char transa, transb;
  int64_t avgM, avgN, avgK;
  ScalarType A_dtype, B_dtype, result_dtype;
  int batchCount;
  bool use_int64;

  // All arrays live in a single device allocation
  Tensor buf;

  // Type-erased pointers into buf (int32_t* or int64_t* depending on use_int64)
  void* mArray;
  void* nArray;
  void* kArray;
  void* ldaArray;
  void* ldbArray;
  void* lddArray;
  int64_t* APtrArray;
  int64_t* BPtrArray;
  int64_t* DPtrArray;
  int64_t* alphaPtrArray;
  int64_t* betaPtrArray;
  float* alphaScalar;
  float* betaScalar;
};
#endif // !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13020

} // namespace at::native
