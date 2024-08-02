#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Undefinedness makes argument matching fail for regular tensor operations
// if 1+ arguments are undefined or possibly undefined tensors.
// Technically, undefined tensors are **not** tensors as the regular tensor
// operations do not know how to handle them.
// However, in practice, there are guards and conversion operators that
// **always** gate regular operations if undefined tensors may be present
// Eventually, we would love to move to the world where we use optionals
// in lieu of undefined tensors.
// When this happens, this pass will be removed
TORCH_API void ClearUndefinedness(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
