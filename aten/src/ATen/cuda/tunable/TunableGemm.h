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

#include <ATen/cuda/tunable/GemmCommon.h>
#ifdef USE_ROCM
#if ROCM_VERSION >= 50700
#include <ATen/cuda/tunable/GemmHipblaslt.h>
#endif
#include <ATen/cuda/tunable/GemmRocblas.h>
#endif
#include <ATen/cuda/tunable/StreamTimer.h>
#include <ATen/cuda/tunable/TunableOp.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/StringUtil.h>

namespace at::cuda::tunable {

template <typename T>
TuningStatus DefaultGemmOp(const GemmParams<T>* params) {
  at::cuda::blas::gemm_internal<T>(
      params->transa, params->transb,
      params->m, params->n, params->k,
      params->alpha,
      params->a, params->lda,
      params->b, params->ldb,
      params->beta,
      params->c, params->ldc);
  return OK;
}

template <typename T>
bool IsZero(T v) {
  return v == 0.0f;
}

template <>
bool IsZero(BFloat16 v) {
  return v.x == 0;
}

template <>
bool IsZero(Half v) {
  return float(v) == 0.0f;
}

template <>
bool IsZero(c10::complex<double> v) {
  return v == 0.0;
}

template <>
bool IsZero(c10::complex<float> v) {
  return v == 0.0f;
}

template <typename T>
std::string TypeName(T v) {
  return "unknown";
}

template <>
std::string TypeName(float v) {
  return "float";
}

template <>
std::string TypeName(double v) {
  return "double";
}

template <>
std::string TypeName(BFloat16 v) {
  return "BFloat16";
}

template <>
std::string TypeName(Half v) {
  return "Half";
}

template <>
std::string TypeName(c10::complex<double> v) {
  return "c10::complex<double>";
}

template <>
std::string TypeName(c10::complex<float> v) {
  return "c10::complex<float>";
}


template <typename T, BlasOp ALayout, BlasOp BLayout>
class GemmTunableOp : public TunableOp<GemmParams<T>, StreamTimer> {
 public:
  GemmTunableOp() {
    this->RegisterOp(std::string("Default"), DefaultGemmOp<T>);

#ifdef USE_ROCM
    for (auto&& [name, op] : GetRocBlasGemmTypeStringAndOps<T>()) {
      this->RegisterOp(std::move(name), std::move(op));
    }
#endif

#if defined(USE_ROCM) && ROCM_VERSION >= 50700
    // disallow tuning of hipblaslt with c10::complex
    if constexpr (
        !std::is_same_v<T, c10::complex<float>> &&
        !std::is_same_v<T, c10::complex<double>>) {
      for (auto&& [name, op] : GetHipBlasLtGemmTypeStringAndOps<T, ALayout, BLayout>()) {
        this->RegisterOp(std::move(name), std::move(op));
      }
    }
#endif

//#ifdef USE_COMPOSABLE_KERNEL
//    for (auto&& [name, op] : GetCKGemmTypeStringAndOps<T, ALayout, BLayout>()) {
//      this->RegisterOp(std::move(name), std::move(op));
//    }
//
//    for (auto&& [name, op] : GetCKStreamKGemmTypeStringAndOps<T, ALayout, BLayout>()) {
//      this->RegisterOp(std::move(name), std::move(op));
//    }
//    for (auto&& [name, op] : GetCKSplitKGemmTypeStringAndOps<T, ALayout, BLayout>()) {
//      this->RegisterOp(std::move(name), std::move(op));
//    }
//#endif
  }

  const GemmParams<T>* PreTuning(const GemmParams<T>* params) override {
    if (!IsZero(params->beta)) {
      // When beta != 0, C buffer is used as an input as well as an output. We need to create a proxy params for the
      // tuning process. Otherwise, tuning will cause the C buffer been updated accumulatedly, say, we tune it for n
      // iterations, then during tuning C^(1) = alpha A B + beta C^(0), ..., C^(n) = alpha A B + beta C^(n-1). And for
      // the actual run after tuning, the result will be C^(n+1), whereas what we want is C^(1). This only happens if
      // the tuning's FindFastest is invoked.
      //
      // Note, C^(i) is the C at i-th iteration.
      GemmParams<T>* proxy = new GemmParams<T>();
      *proxy = *params;
      proxy->c = static_cast<T*>(c10::cuda::CUDACachingAllocator::raw_alloc(proxy->m * proxy->ldc * sizeof(T)));
      return proxy;
    }

    return params;
  }

  void PostTuning(const GemmParams<T>* params) override {
    if (!IsZero(params->beta)) {
      c10::hip::HIPCachingAllocator::raw_delete(params->c);
      delete params;
    }
  }

  std::string Signature() override {
    return c10::str("GemmTunableOp_", TypeName<T>(T{}), "_", BlasOpToString(ALayout), BlasOpToString(BLayout));
  }
};

} // namespace at::cuda::tunable
