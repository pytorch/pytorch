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

  bool hasHalf() const {
    return hasHalf_;
  }

  bool hasBFloat16() const {
    return hasBFloat16_;
  }

  void visit(LoadPtr v) override {
    hasHalf_ |= v->dtype().scalar_type() == ScalarType::Half;
    hasBFloat16_ |= v->dtype().scalar_type() == ScalarType::BFloat16;
    IRVisitor::visit(v);
  }

  void visit(StorePtr v) override {
    hasHalf_ |= v->buf()->dtype().scalar_type() == ScalarType::Half;
    hasBFloat16_ |= v->buf()->dtype().scalar_type() == ScalarType::BFloat16;
    IRVisitor::visit(v);
  }

  void visit(HalfImmPtr v) override {
    hasHalf_ = true;
  }

  void visit(BFloat16ImmPtr v) override {
    hasBFloat16_ = true;
  }

  void visit(CastPtr v) override {
    hasHalf_ |= v->dtype().scalar_type() == ScalarType::Half;
    hasBFloat16_ |= v->dtype().scalar_type() == ScalarType::BFloat16;
    IRVisitor::visit(v);
  }

 private:
  bool hasHalf_{false};
  bool hasBFloat16_{false};
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class HalfRewriter : public IRMutator {
  ExprPtr mutate(LoadPtr v) override {
    ExprPtr child = IRMutator::mutate(v);
    if (!isHalf(child)) {
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
    auto newType = v->value()->dtype();
    ExprPtr new_val = v->value()->accept_mutator(this);

    if (isHalf(newType.scalar_type())) {
      new_val = alloc<Cast>(newType, new_val);
      inserted_half_casts_.insert(new_val);
    }

    v->set_value(new_val);
    return v;
  }

  ExprPtr mutate(HalfImmPtr v) override {
    return alloc<Cast>(kFloat, v);
  }

  ExprPtr mutate(BFloat16ImmPtr v) override {
    return alloc<Cast>(kFloat, v);
  }

  ExprPtr mutate(CastPtr v) override {
    ExprPtr child = v->src_value()->accept_mutator(this);

    // just don't allow half casts we didn't insert.
    if (isHalf(v)) {
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
    if (isHalf(v->dtype().scalar_type())) {
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

  template <typename T>
  ExprPtr mutateArithmetic(T v) {
    IRMutator::mutate(v);
    if (isHalf(v)) {
      v->set_dtype(v->dtype().cloneWithScalarType(c10::kFloat));
    }
    return v;
  }

  ExprPtr mutate(AddPtr v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(SubPtr v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(MulPtr v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(DivPtr v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(MaxPtr v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(MinPtr v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(CompareSelectPtr v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(BroadcastPtr v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(IfThenElsePtr v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(IntrinsicsPtr v) override {
    return mutateArithmetic(v);
  }

 private:
  static bool isHalf(ScalarType st) {
    return st == ScalarType::Half || st == ScalarType::BFloat16;
  }

  static bool isHalf(ExprPtr v) {
    return isHalf(v->dtype().scalar_type());
  }

  std::unordered_set<ExprPtr> inserted_half_casts_;
  std::unordered_map<VarPtr, VarPtr> var_map;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
