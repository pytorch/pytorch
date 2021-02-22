#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/api/module.h>

#include <cstdint>

namespace torch {
namespace jit {
namespace mobile {
using OperatorFunctor = std::function<void(Stack&)>;
TORCH_API OperatorFunctor operator_resolver(const c10::OperatorName& opname, int64_t op_version, int64_t model_version);
}
} // namespace jit
} // namespace torch
