#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace graph_rewrite_helper {

std::string getFuncName(Value* func_value);
Value* getValue(
    const std::string& name,
    const std::unordered_map<const Value*, Value*>& match_vmap,
    const std::unordered_map<std::string, Value*>& vmap);
c10::optional<IValue> getIValue(
    const std::string& name,
    const std::unordered_map<const Value*, Value*>& match_vmap,
    const std::unordered_map<std::string, Value*>& vmap);
void replaceConvolutionWithConv2d(std::shared_ptr<Graph>& graph);

} // namespace graph_rewrite_helper
} // namespace jit
} // namespace torch
