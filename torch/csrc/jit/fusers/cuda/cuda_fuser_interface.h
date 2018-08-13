#ifdef USE_CUDA
#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/fusers/fuser_interface.h"

#include "ATen/ATen.h"

#include <memory>

namespace torch { namespace jit { 


std::shared_ptr<CompiledFusionFunction> getCUDAFusionFunction(Node* fusion_group);


} // namespace jit
} // namespace torch

#endif // USE_CUDA
