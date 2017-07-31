#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

void MatchJITOps(std::unique_ptr<Graph>& graph);

}}
