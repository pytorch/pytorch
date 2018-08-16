#if !(defined _WIN32)
#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/fusers/fuser_interface.h"

#include "ATen/ATen.h"

#include <memory>

namespace torch { namespace jit { 

std::shared_ptr<CompiledFusionFunction> getCPUFusionFunction(Node* fusion_group);

bool canCompileOnCPU();


} // namespace jit
} // namespace torch

#endif // !(defined _WIN32)
