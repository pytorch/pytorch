#pragma once
#include "torch/csrc/jit/fuser/config.h"
#if USE_CUDA_FUSER || USE_CPU_FUSER

#include "torch/csrc/WindowsTorchApiMacro.h"
#include "torch/csrc/jit/stack.h"

#include <cstdint>

namespace torch { namespace jit { namespace fuser {

// Runs the fusion associated with the key (see registerFusion() in interface.h)
// on the inputs taken from the given Stack.
TORCH_API bool runFusion(
  const int64_t key
, Stack& stack);

} // namespace fuser
} // namespace jit
} // namespace torch

#endif // USE_CUDA_FUSER || USE_CPU_FUSER
