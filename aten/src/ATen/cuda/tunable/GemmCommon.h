// Original TunableOp is from onnxruntime.
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/tunable.h
// https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/rocm/tunable
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Adapting TunableOp into PyTorch
// Copyright (c) Advanced Micro Devices, Inc.
//
#pragma once

#include <string>

#include <ATen/cuda/tunable/TunableOp.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/util/StringUtil.h>

namespace at::cuda::tunable {

enum class BlasOp {
  N = 0,
  T = 1
};

inline std::string BlasOpToString(BlasOp op) {
  switch (op) {
    case BlasOp::N:
      return "N";
    case BlasOp::T:
      return "T";
  }
  TORCH_CHECK(false, "unrecognized BlasOp");
  return "N";
}

template <typename T>
struct GemmParams : OpParams {
  std::string Signature() const override {
    return c10::str(transa, transb, "_", m, "_", n, "_", k);
  }

  GemmParams* DeepCopy() const {
    GemmParams* copy = new GemmParams;
    *copy = *this;
    c10::DeviceIndex device = 0;
    AT_CUDA_CHECK(c10::cuda::GetDevice(&device));
    size_t c_size = m * n * sizeof(T);
    copy->c = static_cast<T*>(c10::cuda::CUDACachingAllocator::raw_alloc(c_size));
    AT_CUDA_CHECK(c10::cuda::CUDACachingAllocator::memcpyAsync(
        copy->c, device, c, device, c_size, getCurrentCUDAStream(device), true));
    return copy;
  }

  // only call on object returned by DeepCopy
  void Delete() {
    c10::cuda::CUDACachingAllocator::raw_delete(c);
  }

  TuningStatus NumericalCheck(GemmParams<T> *other) {
    auto options = at::TensorOptions().dtype(c10::CppTypeToScalarType<T>::value).device(at::kCUDA);
    // comparison done as 1D tensor
    at::Tensor ref = at::from_blob(c,        {m*n}, options);
    at::Tensor oth = at::from_blob(other->c, {m*n}, options);
    at::Tensor ref_float = ref.to(at::kFloat);
    at::Tensor oth_float = oth.to(at::kFloat);
    std::vector<double> atols{1e-1, 1e-2, 1e-3, 1e-4, 1e-5};
    std::vector<double> rtols{1e-1, 1e-2, 1e-3, 1e-4, 1e-5};
    double last_succeed_atol = 1;
    double last_succeed_rtol = 1;
    for (auto& atol : atols) {
      for (auto& rtol : rtols) {
        if (at::allclose(ref_float, oth_float, rtol, atol)) {
          last_succeed_atol = atol;
          last_succeed_rtol = rtol;
        }
      }
    }
    if (last_succeed_atol == 1) {
      return FAIL;
    }
    else {
      TUNABLE_LOG("├──verify numerics: atol=", last_succeed_atol, ", rtol=", last_succeed_rtol);
    }

    return OK;
  }

  char transa;
  char transb;
  int64_t m;
  int64_t n;
  int64_t k;
  at::opmath_type<T> alpha;
  const T* a;
  int64_t lda;
  const T* b;
  int64_t ldb;
  at::opmath_type<T> beta;
  T* c;
  int64_t ldc;
};

template <typename T>
struct GemmStridedBatchedParams : OpParams {
  std::string Signature() const override {
    return c10::str(transa, transb, "_", m, "_", n, "_", k, "_B_", batch);
  }

  GemmStridedBatchedParams* DeepCopy() const {
    GemmStridedBatchedParams* copy = new GemmStridedBatchedParams;
    *copy = *this;
    c10::DeviceIndex device = 0;
    AT_CUDA_CHECK(c10::cuda::GetDevice(&device));
    size_t c_size = batch * stride_c * sizeof(T);
    copy->c = static_cast<T*>(c10::cuda::CUDACachingAllocator::raw_alloc(c_size));
    AT_CUDA_CHECK(c10::cuda::CUDACachingAllocator::memcpyAsync(
        copy->c, device, c, device, c_size, getCurrentCUDAStream(device), true));
    return copy;
  }

  // only call on object returned by DeepCopy
  void Delete() {
    c10::cuda::CUDACachingAllocator::raw_delete(c);
  }

  TuningStatus NumericalCheck(GemmStridedBatchedParams<T> *other) {
    auto options = at::TensorOptions().dtype(c10::CppTypeToScalarType<T>::value).device(at::kCUDA);
    // comparison done as 1D tensor
    at::Tensor ref = at::from_blob(c,        {batch*stride_c}, options);
    at::Tensor oth = at::from_blob(other->c, {batch*stride_c}, options);
    at::Tensor ref_float = ref.to(at::kFloat);
    at::Tensor oth_float = oth.to(at::kFloat);
    std::vector<double> atols{1e-1, 1e-2, 1e-3, 1e-4, 1e-5};
    std::vector<double> rtols{1e-1, 1e-2, 1e-3, 1e-4, 1e-5};
    double last_succeed_atol = 1;
    double last_succeed_rtol = 1;
    for (auto& atol : atols) {
      for (auto& rtol : rtols) {
        if (at::allclose(ref_float, oth_float, rtol, atol)) {
          last_succeed_atol = atol;
          last_succeed_rtol = rtol;
        }
      }
    }
    if (last_succeed_atol == 1) {
      return FAIL;
    }
    else {
      TUNABLE_LOG("├──verify numerics: atol=", last_succeed_atol, ", rtol=", last_succeed_rtol);
    }

    return OK;
  }

  char transa;
  char transb;
  int64_t m;
  int64_t n;
  int64_t k;
  at::opmath_type<T> alpha;
  const T* a;
  int64_t lda;
  int64_t stride_a;
  const T* b;
  int64_t ldb;
  int64_t stride_b;
  at::opmath_type<T> beta;
  T* c;
  int64_t ldc;
  int64_t stride_c;
  int64_t batch;
};

} // namespace at::cuda::tunable
