#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

TORCH_API void ScalarTypeAnalysisForONNX(
    const std::shared_ptr<Graph>& graph,
    bool lowprecision_cast,
    int opset_version);
void ScalarTypeAnalysisNodeForONNX(Node* n);

} // namespace torch::jit
