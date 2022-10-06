#pragma once

#include <torch/csrc/Export.h>
#include <string>

namespace torch {
namespace cuda {

// C++-only versions of these, for python use
// those defined in cuda/Module.cpp which also record python state.
TORCH_CUDA_CU_API void _record_memory_history(bool enabled);
TORCH_CUDA_CU_API std::string _memory_snapshot_pickled();

} // namespace cuda
} // namespace torch
