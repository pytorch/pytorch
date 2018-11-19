#pragma once
#include "torch/csrc/jit/fuser/config.h"
#if USE_CUDA_FUSER || USE_CPU_FUSER

#include "torch/csrc/jit/stack.h"

#include <cstdlib>

namespace torch { namespace jit { namespace fuser {

void runFallback(int64_t key, Stack& stack);

} // namespace fuser
} // namespace jit
} // namespace torch

#endif // USE_CUDA_FUSER || USE_CPU_FUSER
