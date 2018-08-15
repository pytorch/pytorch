#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

TORCH_API void eraseIndexWithLists(Graph* graph);


}}  // namespace torch::jit
