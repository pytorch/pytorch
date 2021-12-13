#pragma once

#include <torch/csrc/jit/api/module.h>

namespace torch {
namespace jit {

TORCH_API Module DBRQuantization(const Module& module);

} // namespace jit
} // namespace torch
