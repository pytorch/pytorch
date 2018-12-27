#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

struct HashNode {
  size_t operator()(const Node* k) const;
};

struct EqualNode {
  bool operator()(const Node* lhs, const Node* rhs) const;
};

} // namespace jit
} // namespace torch
