#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/WindowsTorchApiMacro.h> // TORCH_API
#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

TORCH_API bool isFusibleOnCPU(const Node* const node);

TORCH_API int fuseOnCPU(const Node* const node);

TORCH_API void compileFusionOnCPU(Node* fusion);

TORCH_API void callFusionOnCPU(
  const Node* const fusion
, std::vector<at::Tensor>& outputs
, at::ArrayRef<IValue> inputs);

}}}} // namespace torch::jit::fuser::cpu
