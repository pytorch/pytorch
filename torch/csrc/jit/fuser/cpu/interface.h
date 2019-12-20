#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/WindowsTorchApiMacro.h> // TORCH_API
#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

class CPUFusionBackend : public FusionBackend {
public:
  virtual bool isFusible(const Node* const node) override;
  virtual int fuse(const Node* const node) override;
  virtual void compileFusion(Node* fusion) override;
  virtual void callFusion(
      const Node* const fusion,
      std::vector<at::Tensor>&,
      at::ArrayRef<IValue>) override;
};

}}}} // namespace torch::jit::fuser::cpu
