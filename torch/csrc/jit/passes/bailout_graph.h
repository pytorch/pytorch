#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir.h>

#include <list>
#include <vector>

namespace torch {
namespace jit {

using ::c10::ProfiledTensorTypePtr;

TORCH_API void InsertBailOuts(std::shared_ptr<Graph> graph);

TORCH_API std::shared_ptr<Graph> BuildBailOutGraphFrom(
    int64_t bailout_index,
    const std::shared_ptr<Graph>& orig,
    const std::shared_ptr<Graph>& target);
} // namespace jit
} // namespace torch
