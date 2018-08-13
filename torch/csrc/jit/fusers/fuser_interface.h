#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/utils/disallow_copy.h"

#include "ATen/ATen.h"

#include <vector>
#include <memory>

namespace torch { namespace jit { 

constexpr int kCPUDevice = -1;

struct CompiledFusionFunction {
  // Note: creates new tensors for outputs
  virtual void launch(
    at::ArrayRef<at::Tensor> inputs
  , std::vector<at::Tensor>& outputs) = 0;

  virtual ~CompiledFusionFunction() = default;
};

std::shared_ptr<CompiledFusionFunction> getFusionFunction(Node* fusion_group);


} // namespace jit
} // namespace torch
