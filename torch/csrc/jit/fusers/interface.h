#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/stack.h"

#include "torch/csrc/WindowsTorchApiMacro.h"

#include "ATen/ATen.h"

#include <memory>
#include <vector>

namespace torch { namespace jit {

constexpr int kCPUDevice = -1;

struct TORCH_API FusionHandle {
  virtual void run(Stack& inputs) = 0;

  virtual ~FusionHandle() = 0;
};

TORCH_API std::shared_ptr<FusionHandle> getFusionHandle(Node* fusion_group);

TORCH_API bool canFuseOnCPU();
TORCH_API bool canFuseOnGPU();

// CPU fuser is disabled by default, but we still want to test it.
TORCH_API void overrideCanFuseOnCPU(bool value);

TORCH_API std::vector<at::Tensor> debugLaunchGraph(
  Graph& graph
, int device
, at::ArrayRef<at::Tensor> inputs);

} // namespace jit
} // namespace torch
