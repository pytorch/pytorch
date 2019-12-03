#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

TORCH_API int tryCreateFusion(const Node* const node);

}}}} // namespace torch::jit::fuser::cuda
