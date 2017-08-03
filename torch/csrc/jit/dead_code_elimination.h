#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

void EliminateDeadCode(std::unique_ptr<Graph>& graph);

}}

