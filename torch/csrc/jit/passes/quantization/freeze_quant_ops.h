#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

using FreezingOpsFilterFn = std::function<bool(Node*)>;

/** \brief Freeze Quantization Attributes
 *
 * Freezing is a functionality that allows JIT to internalize immutable
 * attributes. In this method we freeze all attributes related to quantizing
 * modules like weight, bias and qparams. This method is combined with inlining
 * and produces a cloned module with only the quantization attributes frozen. If
 * you wish to freeze all attributes in the graph then refer to the
 * freeze_module function instead.
 */

TORCH_API Module FreezeAndFoldQuantOps(
    script::Module& input_module,
    const FreezingOpsFilterFn& is_freezable_op);

} // namespace jit
} // namespace torch
