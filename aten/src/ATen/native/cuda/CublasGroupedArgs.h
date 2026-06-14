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
      const std::optional<Tensor>& scale_a = std::nullopt,
      const std::optional<Tensor>& scale_b = std::nullopt,
      const std::optional<Tensor>& scale_result = std::nullopt,
      const std::optional<at::blas::ScalingType>& scaling_choice_a = std::nullopt,
      const std::optional<at::blas::ScalingType>& scaling_choice_b = std::nullopt);

  char transa, transb;
  int64_t avgM, avgN, avgK;
  ScalarType A_dtype, B_dtype, result_dtype;
  int batchCount;

  // All arrays live in a single device allocation
  Tensor buf;
  // BlockWise1x32 scale pointer arrays need a separate allocation
  Tensor scale_ptr_buf;
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
  float* alphaScalar;
  float* betaScalar;

  void* scale_mata_ptr = nullptr;
  void* scale_matb_ptr = nullptr;
  void* scale_result_ptr = nullptr;
  at::blas::ScalingType scale_mata_scaling_type = at::blas::ScalingType::TensorWise;
  at::blas::ScalingType scale_matb_scaling_type = at::blas::ScalingType::TensorWise;
  c10::ScalarType scale_mata_dtype = c10::ScalarType::Float;
  c10::ScalarType scale_matb_dtype = c10::ScalarType::Float;
};
#endif // !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13020

} // namespace at::native
