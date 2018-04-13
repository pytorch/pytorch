#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/tracer_state.h"

namespace torch { namespace jit {

void ToONNX(std::shared_ptr<tracer::TracingState>& state, bool aten);
std::shared_ptr<Graph> ToONNXFromGraph(std::shared_ptr<Graph>& state, bool aten);

}}
