#pragma once

#include <torch/csrc/jit/ir/ir.h>

#include <vector>

namespace torch {
namespace jit {

TypePtr getTensorType(const at::Tensor& t, bool complete);

TypePtr inferShapeAndTypeForInput(
    TypePtr input_type,
    Stack::const_iterator& s_iter,
    const Stack::const_iterator& s_iter_end,
    bool complete);

void setInputTensorTypes(
    Graph& g,
    const Stack& stack,
    bool complete,
    const std::vector<int>& param_count_list = {});

} // namespace jit
} // namespace torch
