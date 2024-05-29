#pragma once

#include <c10/util/Optional.h>
#include <torch/csrc/Export.h>
#include <cstdint>
#include <string>

namespace torch::cuda {

// C++-only versions of these, for python use
// those defined in cuda/Module.cpp which also record python state.
TORCH_CUDA_CU_API void _record_memory_history(
    bool enabled,
    bool record_context = true,
    int64_t trace_alloc_max_entries = 1,
    bool trace_alloc_record_context = false,
    bool record_cpp_context = false);

TORCH_CUDA_CU_API void _record_memory_history(
    std::optional<std::string> enabled = "all",
    std::optional<std::string> context = "all",
    const std::string& stacks = "all",
    size_t max_entries = SIZE_MAX);

TORCH_CUDA_CU_API std::string _memory_snapshot_pickled();

} // namespace torch::cuda
