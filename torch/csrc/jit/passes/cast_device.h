#pragma once

#include <c10/core/Device.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API void castToDevice(std::shared_ptr<Graph>& graph, c10::Device device);

TORCH_API void castToDevice(script::Module& module, c10::Device device);

} // namespace jit
} // namespace torch
