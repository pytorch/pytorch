#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void HoistConvPackedParams(script::Module& m);

} // namespace jit
} // namespace torch
