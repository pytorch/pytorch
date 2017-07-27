#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

std::unique_ptr<Graph> InitializePyGraph(std::unique_ptr<Graph> graph);

}}
