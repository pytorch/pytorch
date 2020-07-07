#pragma once

#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// return true or false on whether given fusion could be scheduled;
TORCH_CUDA_API bool scheduleFusion(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue> inputs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
