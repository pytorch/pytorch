#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

struct HashNodeCSE {
  size_t operator()(const Node* k) const;
};

struct EqualNodeCSE {
  bool operator()(const Node* lhs, const Node* rhs) const;
};

}}
