#pragma once
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
#include <torch/csrc/jit/fuser/common/fusion.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

TORCH_API void parseJitIR(std::shared_ptr<Graph>& graph, Fusion& fusion);

}}}} // namespace torch::jit::fuser::cuda
