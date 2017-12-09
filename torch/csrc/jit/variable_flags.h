#pragma once

namespace torch { namespace autograd {
struct Variable;
}}

namespace torch { namespace jit {

struct VariableFlags {
  static VariableFlags of(const autograd::Variable& var);
  bool verify(const autograd::Variable& var) const;

  bool requires_grad;
  bool is_volatile;
  bool was_null;
};

}}
