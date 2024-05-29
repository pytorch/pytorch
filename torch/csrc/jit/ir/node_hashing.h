#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

struct TORCH_API HashNode {
  size_t operator()(const Node* k) const;
};

struct TORCH_API EqualNode {
  bool operator()(const Node* lhs, const Node* rhs) const;
};

} // namespace jit
} // namespace torch
