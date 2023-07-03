#pragma once

#include <torch/csrc/Export.h>
#include <string>

namespace torch {
namespace cuda {

// C++-only versions of these, for python use
// those defined in cuda/Module.cpp which also record python state.
TORCH_CUDA_CU_API void _record_memory_history(
    bool enabled,
    bool record_context = true,
    int64_t trace_alloc_max_entries = 1,
    bool trace_alloc_record_context = false,
    bool record_cpp_context = false);

TORCH_CUDA_CU_API std::string _memory_snapshot_pickled();

} // namespace cuda
} // namespace torch
