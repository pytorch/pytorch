#pragma once

#include <torch/csrc/jit/api/module.h>

namespace torch {
namespace jit {

TORCH_API Module DBRQuantAnnotateWithOrderLogger(Module& module, const Module& logger);

} // namespace jit
} // namespace torch
