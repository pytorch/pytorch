#pragma once

#include <unordered_map>
#include <vector>

#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>

namespace torch {
namespace jit {
namespace tensorexpr {

using VarMapping = std::vector<std::pair<const Var*, const Expr*>>;

class VarSubMutator : public IRMutator {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  VarSubMutator(const VarMapping& var_mapping) {
    for (const auto& entry : var_mapping) {
      const Var* key_var = entry.first;
      const Expr* value = entry.second;
      if (!key_var) {
        throw malformed_input("missing key in VarSubMutator");
      }
      var_mapping_[key_var] = value;
    }
  }

  const Expr* mutate(const Var* var) override {
    auto iter = var_mapping_.find(var);
    if (iter == var_mapping_.end()) {
      return var;
    }
    return iter->second;
  }

  const Expr* mutate(const ReduceOp* var) override {
    auto body = var->body()->accept_mutator(this);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<const Var*> new_inner;

    for (auto* v : var->reduce_args()) {
      const Expr* e = v->accept_mutator(this);
      if (const Var* new_var = dynamic_cast<const Var*>(e)) {
        new_inner.push_back(new_var);
      } else {
        VarFinder varFinder;
        e->accept(&varFinder);
        auto varlist = varFinder.vars();
        new_inner.insert(new_inner.end(), varlist.begin(), varlist.end());
      }
    }

    return new ReduceOp(body, new_inner, var->reducer());
  }

 private:
  std::unordered_map<const Var*, const Expr*> var_mapping_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
