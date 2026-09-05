#pragma once

#include <ATen/BlasBackend.h>
#include <ATen/core/Tensor.h>
#include <optional>

#if !defined(USE_ROCM)
#include <cuda.h>
#endif

namespace at::native {

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13030
struct cublasGroupedArgs {
  cublasGroupedArgs(
      const Tensor& mat1,
      const Tensor& mat2,
      const std::optional<Tensor>& offs,
      Tensor& c,
      int batchCount,
      bool needs_int64);

  // In grouped GEMM, m/n/k are the cuBLASLt heuristic averages. The actual
  // per-group dimensions live in mArray, nArray, and kArray.
  char transa;
  char transb;
  int64_t m;
  int64_t n;
  int64_t k;
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
#endif // !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13030

} // namespace at::native
