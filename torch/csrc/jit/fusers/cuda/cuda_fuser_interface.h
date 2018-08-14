#if defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/fusers/fuser_interface.h"

#include "ATen/ATen.h"

#include <memory>

namespace torch { namespace jit { 


std::shared_ptr<CompiledFusionFunction> getCUDAFusionFunction(Node* fusion_group);


} // namespace jit
} // namespace torch

#endif // defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
