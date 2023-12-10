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
    size_t size = m * n * sizeof(T);
    int device;
    AT_CUDA_CHECK(c10::cuda::GetDevice(&device));
    copy->c = static_cast<T*>(c10::cuda::CUDACachingAllocator::raw_alloc(size));
    AT_CUDA_CHECK(c10::cuda::CUDACachingAllocator::memcpyAsync(
        copy->c, device, c, device, size, getCurrentCUDAStream(device), true));
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
    std::vector<double> atols{1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5};
    double last_succeed = 1;
    for (auto& atol : atols) {
      if (at::allclose(ref_float, oth_float, 1e-05, atol)) {
        last_succeed = atol;
      }
      else {
        break;
      }
    }
    if (last_succeed == 1) {
      return FAIL;
    }
    else {
      TUNABLE_LOG("├──verify numerics: atol=", last_succeed, ", rtol=1e-5");
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

} // namespace at::cuda::tunable
