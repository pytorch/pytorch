#pragma once
#include <iostream>
namespace torch { namespace autograd {
struct Variable;
}}

namespace torch { namespace jit {

struct VariableFlags {
  static VariableFlags of(const autograd::Variable& var);

  bool requires_grad;
  bool defined;
};

static inline std::ostream & operator<<(std::ostream & out, const VariableFlags& v) {
  return out
    << "(requires_grad=" << v.requires_grad
    << ", defined=" << v.defined << ")";
}

}}
