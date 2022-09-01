#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>

namespace torch {
namespace jit {

bool add_accumu_on_right(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

bool add_accumu_on_left(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

} // namespace jit
} // namespace torch
