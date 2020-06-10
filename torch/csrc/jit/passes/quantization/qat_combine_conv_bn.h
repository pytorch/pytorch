#pragma once

#include <torch/csrc/jit/api/module.h>

namespace torch {
namespace jit {

/** \brief Combine Conv2d-Batchnorm2d into an IR pattern which scales
 * the Conv's weights using BatchNorm2d's running statistics, for use in QAT.
 * The BN will be folded into the conv after training in a later pass.
 */
TORCH_API Module QATCombineConvBatchNorm2d(const Module& module);

} // namespace jit
} // namespace torch
