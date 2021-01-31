#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void NeZha_TryUpdateGraph(
    std::shared_ptr<Graph>& dst_graph,
    std::shared_ptr<Graph>& src_graph);

} // namespace jit

} // namespace torch
