#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void NeZha_TrySplitModule(
    Module& moudle_1st,
    Module& moudle_2nd);

std::vector<Module> NeZha_GetSplitModules(
    Module& module);

void NeZha_TryUpdateModule(
    Module& dst_module,
    std::shared_ptr<Graph>& src_graph);

void NeZha_TryUpdateModule(
    Module& dst_module,
    std::shared_ptr<Graph>& src_graph);

void NeZha_TryUpdateModule(
    Module& dst_module,
    std::shared_ptr<Graph>& src_graph);

void NeZha_TrySplitModule(
    Module& moudle_1st,
    Module& moudle_2nd);

} // namespace jit

} // namespace torch
