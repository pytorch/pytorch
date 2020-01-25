#pragma once

#include <ATen/core/jit_type.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

struct TORCH_API HashType {
  size_t operator()(const TypePtr& type) const;
};

struct TORCH_API EqualType {
  bool operator()(const TypePtr& a, const TypePtr& b) const;
};

} // namespace jit
} // namespace torch
