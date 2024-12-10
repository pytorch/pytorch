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
#include <ATen/cuda/tunable/GemmHipblaslt.h>
#include <ATen/cuda/tunable/GemmRocblas.h>
#endif
#include <ATen/cuda/tunable/StreamTimer.h>
#include <ATen/cuda/tunable/TunableOp.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/StringUtil.h>
#include <fmt/printf.h>

namespace at::cuda::tunable {

template <typename T>
class DefaultGemmOp : public Callable<GemmParams<T>> {
  public:
    TuningStatus Call(const GemmParams<T>* params) override {
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
};

static bool _transposeBoolFromChar(char op) {
  return op == 't' || op == 'T';
}

template <typename T>
class DefaultGemmAndBiasOp : public Callable<GemmAndBiasParams<T>> {
  public:
    TuningStatus Call(const GemmAndBiasParams<T>* params) override {
      at::cuda::blas::gemm_and_bias<T>(
          _transposeBoolFromChar(params->transa),
          _transposeBoolFromChar(params->transb),
          params->m, params->n, params->k,
          params->alpha,
          params->a, params->lda,
          params->b, params->ldb,
          params->bias,
          params->c, params->ldc,
          params->activation);
      return OK;
    }
};

template <typename T>
class DefaultGemmStridedBatchedOp : public Callable<GemmStridedBatchedParams<T>> {
  public:
    TuningStatus Call(const GemmStridedBatchedParams<T>* params) override {
      at::cuda::blas::bgemm_internal<T>(
          params->transa, params->transb,
          params->m, params->n, params->k,
          params->alpha,
          params->a, params->lda, params->stride_a,
          params->b, params->ldb, params->stride_b,
          params->beta,
          params->c, params->ldc, params->stride_c,
          params->batch);
      return OK;
    }
};

template <typename T>
class DefaultScaledGemmOp : public Callable<ScaledGemmParams<T>> {
  public:
    TuningStatus Call(const ScaledGemmParams<T>* params) override {
      at::cuda::blas::scaled_gemm(
          params->transa,
          params->transb,
          params->m,
          params->n,
          params->k,
          params->a,
          params->a_scale_ptr,
          params->lda,
          params->a_dtype,
          params->b,
          params->b_scale_ptr,
          params->ldb,
          params->b_dtype,
          params->bias_ptr,
          params->bias_dtype,
          params->c,
          params->c_scale_ptr,
          params->ldc,
          params->c_dtype,
          params->use_fast_accum);
      return OK;
    }
};

template <typename T>
inline bool IsZero(T v) {
  return v == 0.0f;
}

template <>
inline bool IsZero(BFloat16 v) {
  return v.x == 0;
}

template <>
inline bool IsZero(Half v) {
  return float(v) == 0.0f;
}

template <>
inline bool IsZero(c10::complex<double> v) {
  return v == 0.0;
}

template <>
inline bool IsZero(c10::complex<float> v) {
  return v == 0.0f;
}

template <typename T>
inline const char* TypeName(T v) {
  return "unknown";
}

template <>
inline const char* TypeName(float v) {
  return "float";
}

template <>
inline const char* TypeName(double v) {
  return "double";
}

template <>
inline const char* TypeName(BFloat16 v) {
  return "BFloat16";
}

template <>
inline const char* TypeName(Half v) {
  return "Half";
}

template <>
inline const char* TypeName(Float8_e4m3fn v) {
  return "Float8_e4m3fn";
}

template <>
inline const char* TypeName(Float8_e5m2 v) {
  return "Float8_e5m2";
}

template <>
inline const char* TypeName(Float8_e4m3fnuz v) {
  return "Float8_e4m3fnuz";
}

template <>
inline const char* TypeName(Float8_e5m2fnuz v) {
  return "Float8_e5m2fnuz";
}

template <>
inline const char* TypeName(c10::complex<double> v) {
  return "c10::complex<double>";
}

template <>
inline const char* TypeName(c10::complex<float> v) {
  return "c10::complex<float>";
}

template <typename T, BlasOp ALayout, BlasOp BLayout>
class GemmTunableOp : public TunableOp<GemmParams<T>, StreamTimer> {
 public:
  GemmTunableOp() {
    this->RegisterOp(std::string("Default"), std::make_unique<DefaultGemmOp<T>>());

#ifdef USE_ROCM
    static const char *env_rocblas = std::getenv("PYTORCH_TUNABLEOP_ROCBLAS_ENABLED");
    if (env_rocblas == nullptr || strcmp(env_rocblas, "1") == 0) {
      for (auto&& [name, op] : GetRocBlasGemmTypeStringAndOps<T>()) {
        this->RegisterOp(std::move(name), std::move(op));
      }
    }

    static const char *env_hipblaslt = std::getenv("PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED");
    if (env_hipblaslt == nullptr || strcmp(env_hipblaslt, "1") == 0) {
      // disallow tuning of hipblaslt with c10::complex
      if constexpr (
          !std::is_same_v<T, c10::complex<float>> &&
          !std::is_same_v<T, c10::complex<double>>) {
        for (auto&& [name, op] : GetHipBlasLtGemmTypeStringAndOps<T, ALayout, BLayout>()) {
          this->RegisterOp(std::move(name), std::move(op));
        }
      }
    }
#endif
  }

  std::string Signature() override {
    return fmt::sprintf("GemmTunableOp_%s_%c%c", TypeName<T>(T{}), BlasOpToString(ALayout), BlasOpToString(BLayout));
  }
};

template <typename T, BlasOp ALayout, BlasOp BLayout>
class GemmAndBiasTunableOp : public TunableOp<GemmAndBiasParams<T>, StreamTimer> {
 public:
  GemmAndBiasTunableOp() {
    this->RegisterOp(std::string("Default"), std::make_unique<DefaultGemmAndBiasOp<T>>());

#ifdef USE_ROCM
    static const char *env_hipblaslt = std::getenv("PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED");
    if (env_hipblaslt == nullptr || strcmp(env_hipblaslt, "1") == 0) {
      // disallow tuning of hipblaslt with c10::complex
      if constexpr (
          !std::is_same_v<T, c10::complex<float>> &&
          !std::is_same_v<T, c10::complex<double>>) {
        for (auto&& [name, op] : GetHipBlasLtGemmAndBiasTypeStringAndOps<T, ALayout, BLayout>()) {
          this->RegisterOp(std::move(name), std::move(op));
        }
      }
    }
#endif
  }

  std::string Signature() override {
    return fmt::sprintf("GemmAndBiasTunableOp_%s_%c%c", TypeName<T>(T{}), BlasOpToString(ALayout), BlasOpToString(BLayout));
  }
};

template <typename T, BlasOp ALayout, BlasOp BLayout>
class GemmStridedBatchedTunableOp : public TunableOp<GemmStridedBatchedParams<T>, StreamTimer> {
 public:
  GemmStridedBatchedTunableOp() {
    this->RegisterOp(std::string("Default"), std::make_unique<DefaultGemmStridedBatchedOp<T>>());

#ifdef USE_ROCM
    static const char *env_rocblas = std::getenv("PYTORCH_TUNABLEOP_ROCBLAS_ENABLED");
    if (env_rocblas == nullptr || strcmp(env_rocblas, "1") == 0) {
      for (auto&& [name, op] : GetRocBlasGemmStridedBatchedTypeStringAndOps<T>()) {
        this->RegisterOp(std::move(name), std::move(op));
      }
    }

    static const char *env_hipblaslt = std::getenv("PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED");
    if (env_hipblaslt == nullptr || strcmp(env_hipblaslt, "1") == 0) {
      // disallow tuning of hipblaslt with c10::complex
      if constexpr (
          !std::is_same_v<T, c10::complex<float>> &&
          !std::is_same_v<T, c10::complex<double>>) {
        for (auto&& [name, op] : GetHipBlasLtGemmStridedBatchedTypeStringAndOps<T, ALayout, BLayout>()) {
          this->RegisterOp(std::move(name), std::move(op));
        }
      }
    }
#endif
  }

  std::string Signature() override {
    return fmt::sprintf("GemmStridedBatchedTunableOp_%s_%c%c", TypeName<T>(T{}), BlasOpToString(ALayout), BlasOpToString(BLayout));
  }
};

template <typename AT, typename BT, typename CT, BlasOp ALayout, BlasOp BLayout>
class ScaledGemmTunableOp : public TunableOp<ScaledGemmParams<CT>, StreamTimer> {
 public:
  ScaledGemmTunableOp() {
    this->RegisterOp(std::string("Default"), std::make_unique<DefaultScaledGemmOp<CT>>());

#ifdef USE_ROCM
    for (auto&& [name, op] : GetHipBlasLtScaledGemmTypeStringAndOps<AT, BT, CT, ALayout, BLayout>()) {
      this->RegisterOp(std::move(name), std::move(op));
    }
#endif
  }

  std::string Signature() override {
    return fmt::sprintf("ScaledGemmTunableOp_%s_%s_%s_%c%c",
      TypeName<AT>(AT{}),
      TypeName<BT>(BT{}),
      TypeName<CT>(CT{}),
      BlasOpToString(ALayout), BlasOpToString(BLayout));
  }
};

} // namespace at::cuda::tunable
