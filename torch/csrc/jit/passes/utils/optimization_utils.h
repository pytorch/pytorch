
#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Checks if the parameters, not including the
// first param are all constants.
bool nonConstantParameters(Node* n);

} // namespace jit
} // namespace torch
