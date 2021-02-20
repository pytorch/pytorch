#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void NeZha_TryUpdateGraph(
    std::shared_ptr<Graph>& dst_graph,
    std::shared_ptr<Graph>& src_graph);

void NeZha_TryUpdateModule(
    Module& dst_module,
    std::shared_ptr<Graph>& src_graph);

void NeZha_TrySplitModule(
    Module& moudle_1st,
    Module& moudle_2nd);

} // namespace jit

} // namespace torch
