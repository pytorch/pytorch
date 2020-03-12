#pragma once
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

TORCH_API bool isNodeParsible(const Node* const node);

TORCH_API void parseJitIR(std::shared_ptr<Graph>& graph, Fusion& fusion);

}}}} // namespace torch::jit::fuser::cuda
