#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

// Decompose addmm nodes to add + mm, so expands can be inserted and
// gradients accumulated on the backward pass
//
// In the future, if we need more passes like this, we should convert this
// into a generic canonicalization pass.
void DecomposeAddmm(const std::shared_ptr<Graph>& graph);

}}
