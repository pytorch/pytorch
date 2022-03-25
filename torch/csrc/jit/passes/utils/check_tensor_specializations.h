#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

namespace torch {
namespace jit {

// This file regisiters custom passes to allow for Python to check
// if Tensor type specializations are available for custom passes.

TORCH_API void addTensorTypeSpecializationDetectionPass();
TORCH_API void removeTensorTypeSpecializationDetectionPass();
TORCH_API bool passDetectedSpecializedTensors();
} // namespace jit
} // namespace torch
