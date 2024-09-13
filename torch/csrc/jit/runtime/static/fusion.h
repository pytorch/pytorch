#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

TORCH_API void fuseStaticSubgraphs(
    std::shared_ptr<Graph> graph,
    size_t min_size);

TORCH_API void performTensorExprFusion(
    std::shared_ptr<Graph> graph,
    std::vector<IValue> sample_inputs);

} // namespace torch::jit
