
copy: fbcode/caffe2/torch/csrc/jit/ir/node_hashing.h
copyrev: 63d625b3b52fb6eafab96b5361711a26849957cc

#pragma once

#include <torch/csrc/jit/ir/ir.h>

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
