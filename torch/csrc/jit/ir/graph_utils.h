#pragma once

#include <torch/csrc/jit/ir/ir.h>

#include <vector>

namespace torch {
namespace jit {

TORCH_API TypePtr getTensorType(const at::Tensor& t, bool complete);

TORCH_API TypePtr inferShapeAndTypeForInput(
    TypePtr input_type,
    Stack::const_iterator& s_iter,
    const Stack::const_iterator& s_iter_end,
    bool complete);

TORCH_API void setInputTensorTypes(
    Graph& g,
    const Stack& stack,
    bool complete,
    const std::vector<int>& param_count_list = {});

} // namespace jit
} // namespace torch
