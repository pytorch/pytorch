#pragma once

#ifdef USE_CUDA

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

// NB: Be sure to run DCE before fusion, because dead instructions
// can prevent fusion opportunities from being exploited.
TORCH_API void FuseCUDAGraph(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch

#endif // USE_CUDA
