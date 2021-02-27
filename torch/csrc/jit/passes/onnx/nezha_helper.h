#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void NeZha_TrySplitModule(
    Module& moudle_1st,
    Module& moudle_2nd);

std::vector<Module> NeZha_GetSplitModules(
    Module& module);

} // namespace jit

} // namespace torch
