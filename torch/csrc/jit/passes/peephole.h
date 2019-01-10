#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

void PeepholeOptimize(std::shared_ptr<Graph>& graph);

}}
