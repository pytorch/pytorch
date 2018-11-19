#pragma once
#include "torch/csrc/jit/fuser/config.h"
#if USE_CUDA_FUSER || USE_CPU_FUSER

#include "c10/util/Optional.h"
#include "torch/csrc/WindowsTorchApiMacro.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/fuser/kernel_spec.h"

#include <cstdint> 
#include <functional>

namespace torch { namespace jit { namespace fuser {

// A thread-safe cache interface.

// Stores the given graph, returning the key used to access it
TORCH_API int64_t store(std::shared_ptr<Graph> graph);

// Returns the graph corresponding to the given key (if it exists)
TORCH_API at::optional<KernelSpec*> retrieve(const int64_t key);

} // namespace fuser
} // namespace jit
} // namespace torch

#endif // USE_CUDA_FUSER || USE_CPU_FUSER
