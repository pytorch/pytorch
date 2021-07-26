#pragma once

#include <jit/runtime/profiling_record.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API void InsertAllocationNodes(
    std::shared_ptr<Graph>&,
    std::unique_ptr<ProfilingRecord>&);

} // namespace jit
} // namespace torch
