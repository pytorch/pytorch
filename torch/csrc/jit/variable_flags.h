#pragma once
#include <iostream>
namespace torch { namespace autograd {
struct Variable;
}}

namespace torch { namespace jit {

struct VariableFlags {
  static VariableFlags of(const autograd::Variable& var);
  bool verify(const autograd::Variable& var) const;

  bool requires_grad;
  bool was_null;
};

static inline std::ostream & operator<<(std::ostream & out, const VariableFlags& v) {
  return out
    << "(requires_grad=" << v.requires_grad
    << ", was_null=" << v.was_null << ")";
}

}}
