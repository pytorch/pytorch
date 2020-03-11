#pragma once
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class CUDAFusionBackend : public FusionBackend {
public:
  virtual bool isFusible(const Node* const node) override;
  virtual bool isFusible(
      const Node* const fusion,
      const Node* const node) override;
  virtual int fuse(const Node* const node) override;
  virtual void compileFusion(Node* fusion) override;
  virtual void callFusion(
      const Node* const fusion,
      Stack& stack) override;
};

}}}} // namespace torch::jit::fuser::cuda
