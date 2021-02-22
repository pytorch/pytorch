#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void NeZha_TrySplitModule(
    Module& moudle_1st,
    Module& moudle_2nd);

} // namespace jit

} // namespace torch
