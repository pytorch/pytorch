#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/tracer_state.h"

namespace torch { namespace jit {

std::shared_ptr<Graph> ToONNX(std::shared_ptr<Graph>& state, bool aten);

}}
