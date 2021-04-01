#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API void ScalarTypeAnalysisForONNX(const std::shared_ptr<Graph>& graph);
void ScalarTypeAnalysisNodeForONNX(Node* n);

TORCH_API void ScalarTypeAnalysisForONNXWithoutLowPrecision(
    const std::shared_ptr<Graph>& graph);
void ScalarTypeAnalysisNodeForONNXWithoutLowPrecision(Node* n);
} // namespace jit
} // namespace torch
