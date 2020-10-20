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
      inserted_half_casts_.insert(new_val);
      hasHalf_ = true;
    }

    return new Store(v->buf(), v->indices(), new_val, v->mask());
  }

  const Expr* mutate(const HalfImm* v) override {
    hasHalf_ = true;
    return new Cast(kFloat, v);
  }

  const Expr* mutate(const Cast* v) override {
    const Expr* child = v->src_value()->accept_mutator(this);

    // just don't allow half casts we didn't insert.
    if (v->dtype().scalar_type() == ScalarType::Half) {
      if (inserted_half_casts_.count(v) < 1) {
        // TODO: discards lanes.
        return new Cast(kFloat, child);
      }
    }

    if (child == v->src_value()) {
      return v;
    }

    return new Cast(v->dtype(), child);
  }

 private:
  bool hasHalf_{false};
  std::unordered_set<const Expr*> inserted_half_casts_;
};

class CudaHalfScalarRewriter : public IRMutator {
  Stmt* mutate(const Let* v) override {
    if (v->dtype().scalar_type() == ScalarType::Half) {
      // TODO: discards lanes.
      const Var* load_new_var = new Var(v->var()->name_hint(), kFloat);
      const Expr* new_value =
          new Cast(kFloat, v->value()->accept_mutator(this));
      var_map[v->var()] = load_new_var;

      return new Let(load_new_var, new_value);
    }

    return IRMutator::mutate(v);
  }

  const Expr* mutate(const Var* v) override {
    auto it = var_map.find(v);
    if (it != var_map.end()) {
      return it->second;
    }

    return v;
  }

 private:
  std::unordered_map<const Var*, const Var*> var_map;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
