/** This file defines pretransposing weight for addmm if it is constant by
 *  calling contiguous. This is meant to be used after model freezing.
 */
#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/module.h>

/** Pre-transpose addmm weight, which can be faster on SKL machines
 *  Currently assumes that module is frozen
 */

namespace torch {
namespace jit {

TORCH_API void PretransposeWeights(std::shared_ptr<Graph>& g);

} // namespace jit
} // namespace torch
