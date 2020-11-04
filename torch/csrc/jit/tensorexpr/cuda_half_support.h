#pragma once

#include <torch/csrc/jit/codegen/fuser/cuda/resource_strings.h>
#include <torch/csrc/jit/tensorexpr/cuda_codegen.h>

namespace torch {
namespace jit {
namespace tensorexpr {

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
