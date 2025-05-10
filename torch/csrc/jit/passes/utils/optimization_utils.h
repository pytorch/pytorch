
#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

// Checks if the parameters, not including the
// first param are all constants.
bool nonConstantParameters(Node* n);

} // namespace torch::jit
