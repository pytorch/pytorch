#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for ELU functional and module.
struct ELUOptions {
  ELUOptions() {}

  /// The alpha value for the ELU formulation. Default: 1.0
  TORCH_ARG(double, alpha) = 1.0;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

// ============================================================================

/// Options for Hardshrink functional and module.
struct TORCH_API HardshrinkOptions {
  /* implicit */ HardshrinkOptions(double lambda = 0.5);

  /// the lambda value for the Hardshrink formulation. Default: 0.5
  TORCH_ARG(double, lambda);
};

// ============================================================================

/// Options for Hardtanh functional and module.
struct HardtanhOptions {
  HardtanhOptions() {}

  /// minimum value of the linear region range. Default: -1
  TORCH_ARG(double, min_val) = -1.0;

  /// maximum value of the linear region range. Default: 1
  TORCH_ARG(double, max_val) = 1.0;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

// ============================================================================

/// Options for LeakyReLU functional and module.
struct LeakyReLUOptions {
  LeakyReLUOptions() {}

  /// Controls the angle of the negative slope. Default: 1e-2
  TORCH_ARG(double, negative_slope) = 1e-2;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

// ============================================================================

/// Options for MultiheadAttention functional and module.
struct MultiheadAttentionOptions {
  MultiheadAttentionOptions(int64_t embed_dim, int64_t num_heads)
    : embed_dim_(embed_dim), num_heads_(num_heads) {
    if (!kdim_) {
      kdim_ = embed_dim;
    }
    if (!vdim_) {
      vdim_ = embed_dim;
    }
  }

  /// total dimension of the model.
  TORCH_ARG(int64_t, embed_dim);

  /// parallel attention heads.
  TORCH_ARG(int64_t, num_heads);

  /// a Dropout layer on attn_output_weights. Default: 0.0.
  TORCH_ARG(double, dropout) = 0.0;

  /// add bias as module parameter. Default: true.
  TORCH_ARG(bool, bias) = true;

  /// add bias to the key and value sequences at dim=0.
  TORCH_ARG(bool, add_bias_kv) = false;

  /// add a new batch of zeros to the key and value sequences at dim=1.
  TORCH_ARG(bool, add_zero_attn) = false;

  /// total number of features in key. Default: c10::nullopt.
  TORCH_ARG(c10::optional<int64_t>, kdim) = c10::nullopt;

  /// total number of features in key. Default: c10::nullopt.
  TORCH_ARG(c10::optional<int64_t>, vdim) = c10::nullopt;
};

} // namespace nn
} // namespace torch
