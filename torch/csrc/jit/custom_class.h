#pragma once

#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/jit/script/compilation_unit.h>

#include <vector>

namespace torch {
namespace jit {

TORCH_API std::vector<c10::RegisterOperators>& registeredOps();
TORCH_API std::shared_ptr<script::CompilationUnit>& classCU();

} // namespace jit
} // namespace torch
