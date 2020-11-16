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

} // namespace jit
} // namespace torch
