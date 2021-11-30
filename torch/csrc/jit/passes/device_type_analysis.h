#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Transposes the weight matrix for frozen linear modules.
// and converts it into a matmul

// Should support everything except for when the device type is passed in as an arg
TORCH_API bool DeviceTypePropagation(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
