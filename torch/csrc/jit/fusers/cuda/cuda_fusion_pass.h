#if defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
#pragma once

#include "torch/csrc/jit/ir.h"

#include "torch/csrc/WindowsTorchApiMacro.h"

namespace torch { namespace jit {

// NB: Be sure to run DCE before fusion, because dead instructions
// can prevent fusion opportunities from being exploited.
TORCH_API void CUDAFuserPass(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch

#endif // defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
