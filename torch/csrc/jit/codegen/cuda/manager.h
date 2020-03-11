#pragma once
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

TORCH_API void compileCudaFusionGroup(Node* fusion_node);

TORCH_API void runCudaFusionGroup(const Node* const fusion_node, Stack& stack);

}}}} // namespace torch::jit::fuser::cuda
