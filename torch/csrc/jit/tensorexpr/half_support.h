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

  void visit(Load* v) override {
    hasHalf_ |= v->dtype().scalar_type() == ScalarType::Half;
    IRVisitor::visit(v);
  }

  void visit(Store* v) override {
    hasHalf_ |= v->buf()->dtype().scalar_type() == ScalarType::Half;
    IRVisitor::visit(v);
  }

  void visit(const HalfImm* v) override {
    hasHalf_ = true;
  }

  void visit(Cast* v) override {
    hasHalf_ |= v->dtype().scalar_type() == ScalarType::Half;
    IRVisitor::visit(v);
  }

 private:
  bool hasHalf_{false};
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class HalfRewriter : public IRMutator {
  Expr* mutate(Load* v) override {
    Expr* child = IRMutator::mutate(v);
    if (child->dtype().scalar_type() != ScalarType::Half) {
      return child;
    }

    Expr* ret =
        new Cast(child->dtype().cloneWithScalarType(ScalarType::Float), child);

    inserted_half_casts_.insert(ret);
    return ret;
  }

  Stmt* mutate(Store* v) override {
    Expr* new_val = v->value()->accept_mutator(this);

    Dtype newType = v->value()->dtype();
    if (newType.scalar_type() == ScalarType::Half) {
      new_val =
          new Cast(newType.cloneWithScalarType(ScalarType::Half), new_val);
      inserted_half_casts_.insert(new_val);
    }

    return new Store(v->buf(), v->indices(), new_val);
  }

  Expr* mutate(HalfImm* v) override {
    return new Cast(kFloat, v);
  }

  Expr* mutate(Cast* v) override {
    Expr* child = v->src_value()->accept_mutator(this);

    // just don't allow half casts we didn't insert.
    if (v->dtype().scalar_type() == ScalarType::Half) {
      if (inserted_half_casts_.count(v) < 1) {
        return child;
      }
    }

    // Remove Half(Float()) and friends.
    Cast* cast_child = dynamic_cast<Cast*>(child);
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
  Stmt* mutate(Let* v) override {
    if (v->dtype().scalar_type() == ScalarType::Half) {
      Var* load_new_var = new Var(v->var()->name_hint(), kFloat);
      Expr* new_value = new Cast(
          v->dtype().cloneWithScalarType(ScalarType::Float),
          v->value()->accept_mutator(this));
      var_map[v->var()] = load_new_var;

      return new Let(load_new_var, new_value);
    }

    return IRMutator::mutate(v);
  }

  Expr* mutate(Var* v) override {
    auto it = var_map.find(v);
    if (it != var_map.end()) {
      return it->second;
    }

    return v;
  }

 private:
  std::unordered_set<Expr*> inserted_half_casts_;
  std::unordered_map<Var*, Var*> var_map;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
