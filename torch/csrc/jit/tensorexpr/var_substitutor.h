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

using VarMapping = std::vector<std::pair<VarPtr, ExprPtr>>;

class VarSubMutator : public IRMutator {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  VarSubMutator(const VarMapping& var_mapping) {
    for (auto& entry : var_mapping) {
      VarPtr key_var = entry.first;
      ExprPtr value = entry.second;
      if (!key_var) {
        throw malformed_input("missing key in VarSubMutator");
      }
      var_mapping_[key_var] = value;
    }
  }

  ExprPtr mutate(VarPtr var) override {
    auto iter = var_mapping_.find(var);
    if (iter == var_mapping_.end()) {
      return var;
    }
    return iter->second;
  }

  ExprPtr mutate(ReduceOpPtr var) override {
    auto body = var->body()->accept_mutator(this);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<VarPtr> new_inner;

    for (auto v : var->reduce_args()) {
      ExprPtr e = v->accept_mutator(this);
      if (VarPtr new_var = to<Var>(e)) {
        new_inner.push_back(new_var);
      } else {
        VarFinder varFinder;
        e->accept(&varFinder);
        auto varlist = varFinder.vars();
        new_inner.insert(new_inner.end(), varlist.begin(), varlist.end());
      }
    }

    return alloc<ReduceOp>(body, new_inner, var->reducer());
  }

 private:
  std::unordered_map<VarPtr, ExprPtr> var_mapping_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
