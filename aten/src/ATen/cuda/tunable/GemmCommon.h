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

#include <ATen/cuda/tunable/TunableOp.h>
#include <c10/util/StringUtil.h>

namespace at::cuda::tunable {

template <typename T>
struct GemmParams : OpParams {
  std::string Signature() const override {
    return c10::str(transa, transb, "_", m, "_", n, "_", k);
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
