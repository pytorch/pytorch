#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

void UnrollLoops(std::shared_ptr<Graph>& graph);

}} // namespace torch::jit
