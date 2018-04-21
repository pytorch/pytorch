#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

std::shared_ptr<Graph> Canonicalize(const std::shared_ptr<Graph>& graph);

}}
