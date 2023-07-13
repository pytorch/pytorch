/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ATen/core/TensorBody.h>
#include <c10/util/Exception.h>

#include <cutlass/bfloat16.h>
#include <cutlass/half.h>

/**
 * kernels expect 4D bias/bias.grad with shape
 * (batch_sz, n_heads, n_queries, n_keys). common bias shapes users may pass
 * are:
 * - (n_queries, n_keys)
 * - (batch_sz * n_heads, n_queries, n_keys)
 * - (batch_sz, n_heads, n_queries, n_keys)
 *
 * expand the bias as needed - be careful to only create a view with different
 * shape/strides, no copies allowed.
 */
inline at::Tensor get_bias_4d_view(
    const at::Tensor& bias,
    int batch_sz,
    int n_heads,
    int n_queries,
    int n_keys) {
  TORCH_CHECK(
      bias.size(-2) == n_queries,
      "bias.size(-2) != n_queries: ",
      bias.size(-2),
      " != ",
      n_queries);
  TORCH_CHECK(
      bias.size(-1) == n_keys,
      "bias.size(-1) != n_keys: ",
      bias.size(-1),
      " != ",
      n_keys);
  switch (bias.dim()) {
    case 2: // (n_queries, n_keys) - broadcast across all batches and heads
      return bias.unsqueeze(0).unsqueeze(0).expand(
          {batch_sz, n_heads, n_queries, n_keys});
    case 3: // (batch_sz * n_heads, n_queries, n_keys) - just reshape
      TORCH_CHECK(bias.size(0) == batch_sz * n_heads);
      return bias.view({batch_sz, n_heads, n_queries, n_keys});
    case 4: // (batch_sz, n_heads, n_queries, n_keys) - do nothing
      TORCH_CHECK(bias.size(0) == batch_sz);
      TORCH_CHECK(bias.size(1) == n_heads)
      return bias;
    default:
      TORCH_CHECK(false, "bias can only have ndims in {2, 3, 4}");
  }
}

template <typename scalar_t>
struct CutlassToAtenDtype;

template <>
struct CutlassToAtenDtype<cutlass::half_t> {
  using scalar_t = cutlass::half_t;

  static constexpr __host__ at::ScalarType atScalarType() {
    return at::ScalarType::Half;
  }
};

template <>
struct CutlassToAtenDtype<cutlass::bfloat16_t> {
  using scalar_t = cutlass::bfloat16_t;

  static constexpr __host__ at::ScalarType atScalarType() {
    return at::ScalarType::BFloat16;
  }
};

template <>
struct CutlassToAtenDtype<float> {
  using scalar_t = float;

  static constexpr __host__ at::ScalarType atScalarType() {
    return at::ScalarType::Float;
  }
};
