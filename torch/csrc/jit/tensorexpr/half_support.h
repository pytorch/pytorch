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

  void visit(const Load* v) override {
    hasHalf_ |= v->dtype().scalar_type() == ScalarType::Half;
    IRVisitor::visit(v);
  }

  void visit(const Store* v) override {
    hasHalf_ |= v->buf()->dtype().scalar_type() == ScalarType::Half;
    IRVisitor::visit(v);
  }

  void visit(const HalfImm* v) override {
    hasHalf_ = true;
  }

  void visit(const Cast* v) override {
    hasHalf_ |= v->dtype().scalar_type() == ScalarType::Half;
    IRVisitor::visit(v);
  }

 private:
  bool hasHalf_{false};
};

class HalfRewriter : public IRMutator {
  const Expr* mutate(const Load* v) override {
    const Expr* child = IRMutator::mutate(v);
    if (child->dtype().scalar_type() != ScalarType::Half) {
      return child;
    }

    const Expr* ret =
        new Cast(child->dtype().cloneWithScalarType(ScalarType::Float), child);

    inserted_half_casts_.insert(ret);
    return ret;
  }

  Stmt* mutate(const Store* v) override {
    const Expr* new_val = v->value()->accept_mutator(this);

    Dtype newType = v->value()->dtype();
    if (newType.scalar_type() == ScalarType::Half) {
      new_val =
          new Cast(newType.cloneWithScalarType(ScalarType::Half), new_val);
      inserted_half_casts_.insert(new_val);
    }

    return new Store(v->buf(), v->indices(), new_val, v->mask());
  }

  const Expr* mutate(const HalfImm* v) override {
    return new Cast(kFloat, v);
  }

  const Expr* mutate(const Cast* v) override {
    const Expr* child = v->src_value()->accept_mutator(this);

    // just don't allow half casts we didn't insert.
    if (v->dtype().scalar_type() == ScalarType::Half) {
      if (inserted_half_casts_.count(v) < 1) {
        return child;
      }
    }

    // Remove Half(Float()) and friends.
    const Cast* cast_child = dynamic_cast<const Cast*>(child);
    if (cast_child) {
      if (v->dtype().is_floating_point() &&
          cast_child->dtype().is_floating_point()) {
        return new Cast(v->dtype(), cast_child->src_value());
      }
    }

    if (child == v->src_value()) {
      return v;
    }

    return new Cast(v->dtype(), child);
  }
  Stmt* mutate(const Let* v) override {
    if (v->dtype().scalar_type() == ScalarType::Half) {
      const Var* load_new_var = new Var(v->var()->name_hint(), kFloat);
      const Expr* new_value = new Cast(
          v->dtype().cloneWithScalarType(ScalarType::Float),
          v->value()->accept_mutator(this));
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
  std::unordered_set<const Expr*> inserted_half_casts_;
  std::unordered_map<const Var*, const Var*> var_map;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
