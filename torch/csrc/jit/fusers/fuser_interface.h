#pragma once
#include "torch/csrc/jit/ir.h"

#include "ATen/ATen.h"

#include <vector>
#include <memory>
 
namespace torch { namespace jit { 
 
 constexpr int kCPUDevice = -1;
 
 TORCH_API struct CompiledFusionFunction {
  // Note: creates new tensors for outputs
  virtual void launch(
    at::ArrayRef<at::Tensor> inputs
  , std::vector<at::Tensor>& outputs) = 0;
   virtual ~CompiledFusionFunction() = default;
};

TORCH_API void FuseCPUGraph(std::shared_ptr<Graph>& graph);
TORCH_API void FuseCUDAGraph(std::shared_ptr<Graph>& graph);

//  TORCH_API std::shared_ptr<CompiledFusionFunction> getFusionFunction(Node* fusion_group);
 
} // namespace jit
} // namespace torch
