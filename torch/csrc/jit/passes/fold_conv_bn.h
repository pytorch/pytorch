#pragma once

#include <torch/csrc/jit/api/module.h>

namespace torch {
namespace jit {

/** \brief Fold Conv2d-BatchNorm2d into Conv2d in forward method of this module
 * and all its submodules.
 *
 * The weight and bias of the Conv2d are correspondingly updated. Should only be
 * used on modules in eval mode.
 */
TORCH_API Module FoldConvBatchNorm(const Module& module);

struct TORCH_API ConvBNParameters {
  at::Tensor conv_w;
  at::Tensor conv_b;
  at::Tensor bn_rm;
  at::Tensor bn_rv;
  double bn_eps = 0.0;
  at::Tensor bn_w;
  at::Tensor bn_b;
};

/**
 * Given the current weight and bias tensors of a Conv module and parameters
 * of the BatchNorm module we're folding with, compute the updated values
 * for the weight and bias.
 *
 * The function is basically copied from torch/nn/utils/fusion.py
 */
TORCH_API std::tuple<at::Tensor, at::Tensor> computeUpdatedConvWeightAndBias(
    const ConvBNParameters& p);

} // namespace jit
} // namespace torch
