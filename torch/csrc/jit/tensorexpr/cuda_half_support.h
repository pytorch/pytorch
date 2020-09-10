#pragma once

#include <torch/csrc/jit/codegen/fuser/cuda/resource_strings.h>
#include <torch/csrc/jit/tensorexpr/cuda_codegen.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// Walk the Statment looking for Half size loads/stores.
class CudaHalfChecker : public IRMutator {
 public:
  bool hasHalf() {
    return hasHalf_;
  }

  const Expr* mutate(const Load* v) override {
    const Expr* child = IRMutator::mutate(v);
    if (child->dtype().scalar_type() != ScalarType::Half) {
      return child;
    }

    hasHalf_ = true;

    // TODO discards lanes.
    return new Cast(kFloat, child);
  }

  Stmt* mutate(const Store* v) override {
    const Expr* new_val = v->value()->accept_mutator(this);

    if (v->value()->dtype().scalar_type() == ScalarType::Half) {
      // TODO discards lanes.
      new_val = new Cast(kHalf, new_val);
      hasHalf_ = true;
    }

    return new Store(v->buf(), v->indices(), new_val, v->mask());
  }

 private:
  bool hasHalf_{false};
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
