#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

TORCH_API std::shared_ptr<Graph> ImportIRGraph(const std::string& serialized_graph, std::vector<at::Tensor> & initializers);

}}
