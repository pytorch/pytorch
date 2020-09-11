#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>

#include <list>
#include <vector>

namespace torch {
namespace jit {

TORCH_API void InsertGuards(std::shared_ptr<Graph> graph);

TORCH_API void RemoveProfilingNodes(std::shared_ptr<Graph> graph);

} // namespace jit
} // namespace torch
