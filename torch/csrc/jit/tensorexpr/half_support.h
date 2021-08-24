#pragma once

#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// Walk the Statment looking for Half size loads/stores.
class HalfChecker : public IRVisitor {
 public:
  HalfChecker(const std::vector<CodeGen::BufferArg>& args) {
    for (const auto& BA : args) {
      hasHalf_ |= BA.dtype().scalar_type() == ScalarType::Half;
    }
  }

  bool hasHalf() {
    return hasHalf_;
  }

  void visit(LoadPtr v) override {
    hasHalf_ |= v->dtype().scalar_type() == ScalarType::Half;
    IRVisitor::visit(v);
  }

  void visit(StorePtr v) override {
    hasHalf_ |= v->buf()->dtype().scalar_type() == ScalarType::Half;
    IRVisitor::visit(v);
  }

  void visit(HalfImmPtr v) override {
    hasHalf_ = true;
  }

  void visit(CastPtr v) override {
    hasHalf_ |= v->dtype().scalar_type() == ScalarType::Half;
    IRVisitor::visit(v);
  }

 private:
  bool hasHalf_{false};
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class HalfRewriter : public IRMutator {
  ExprPtr mutate(LoadPtr v) override {
    ExprPtr child = IRMutator::mutate(v);
    if (child->dtype().scalar_type() != ScalarType::Half) {
      return child;
    }

    ExprPtr ret = alloc<Cast>(
        child->dtype().cloneWithScalarType(ScalarType::Float), child);

    inserted_half_casts_.insert(ret);
    return ret;
  }

  StmtPtr mutate(StorePtr v) override {
    // Since mutation changes the `value()` expression in-place, we need to
    // get the dtype of the `value()` before that is mutated.
    Dtype newType = v->value()->dtype();
    ExprPtr new_val = v->value()->accept_mutator(this);

    if (newType.scalar_type() == ScalarType::Half) {
      new_val =
          alloc<Cast>(newType.cloneWithScalarType(ScalarType::Half), new_val);
      inserted_half_casts_.insert(new_val);
    }

    v->set_value(new_val);
    return v;
  }

  ExprPtr mutate(HalfImmPtr v) override {
    return alloc<Cast>(kFloat, v);
  }

  ExprPtr mutate(CastPtr v) override {
    ExprPtr child = v->src_value()->accept_mutator(this);

    // just don't allow half casts we didn't insert.
    if (v->dtype().scalar_type() == ScalarType::Half) {
      if (inserted_half_casts_.count(v) < 1) {
        return child;
      }
    }

    // Remove Half(Float()) and friends.
    CastPtr cast_child = to<Cast>(child);
    if (cast_child) {
      if (v->dtype().is_floating_point() &&
          cast_child->dtype().is_floating_point()) {
        return alloc<Cast>(v->dtype(), cast_child->src_value());
      }
    }

    if (child == v->src_value()) {
      return v;
    }

    return alloc<Cast>(v->dtype(), child);
  }
  StmtPtr mutate(LetPtr v) override {
    if (v->dtype().scalar_type() == ScalarType::Half) {
      VarPtr load_new_var = alloc<Var>(v->var()->name_hint(), kFloat);
      ExprPtr new_value = alloc<Cast>(
          v->dtype().cloneWithScalarType(ScalarType::Float),
          v->value()->accept_mutator(this));
      var_map[v->var()] = load_new_var;

      return alloc<Let>(load_new_var, new_value);
    }

    return IRMutator::mutate(v);
  }

  ExprPtr mutate(VarPtr v) override {
    auto it = var_map.find(v);
    if (it != var_map.end()) {
      return it->second;
    }

    return v;
  }

 private:
  std::unordered_set<ExprPtr> inserted_half_casts_;
  std::unordered_map<VarPtr, VarPtr> var_map;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
