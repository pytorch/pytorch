#pragma once

#include "torch/csrc/jit/tensorexpr/eval.h"
#include "torch/csrc/jit/tensorexpr/ir_mutator.h"
#include "torch/csrc/jit/tensorexpr/ir_visitor.h"
#include "torch/csrc/jit/tensorexpr/types.h"

namespace torch {
namespace jit {
namespace tensorexpr {

/* Mutate the IR by collapsing expressions with constant terms down to a single
 * immediate. Uses the IR Evaluator as a source of truth.
 */
class ConstantFolder : public IRMutator {
 public:
  const Expr* mutate(const Add* v) override {
    return mutateBinaryOp(v, this);
  }

  const Expr* mutate(const Sub* v) override {
    return mutateBinaryOp(v, this);
  }

  const Expr* mutate(const Mul* v) override {
    return mutateBinaryOp(v, this);
  }

  const Expr* mutate(const Div* v) override {
    return mutateBinaryOp(v, this);
  }

  const Expr* mutate(const Mod* v) override {
    return mutateBinaryOp(v, this);
  }

  const Expr* mutate(const And* v) override {
    return mutateBinaryOp(v, this);
  }

  const Expr* mutate(const Xor* v) override {
    return mutateBinaryOp(v, this);
  }

  const Expr* mutate(const Lshift* v) override {
    return mutateBinaryOp(v, this);
  }

  const Expr* mutate(const Rshift* v) override {
    return mutateBinaryOp(v, this);
  }

  const Expr* mutate(const Max* v) override {
    return mutateBinaryOp(v, this, v->propagate_nans());
  }

  const Expr* mutate(const Min* v) override {
    return mutateBinaryOp(v, this, v->propagate_nans());
  }

  const Expr* mutate(const Intrinsics* v) override {
    std::vector<const Expr*> new_params;
    bool changed = false;
    bool allConstant = true;
    for (const auto* p : v->params()) {
      const Expr* new_child =  p->accept_mutator(this);
      new_params.push_back(new_child);

      changed |= p != new_child;
      allConstant &= new_child->isConstant();
    }

    const Expr* node = v;
    if (changed) {
      node = new Intrinsics(v->op_type(), new_params);
    }

    if (!allConstant) {
      return node;
    }

    return evaluateOp(node);
  }

 private:
  static const Expr* evaluateOp(const Expr* v) {
    ExprHandle handle(v);
    ExprEval<SimpleIREvaluator> eval(handle);

    switch (v->dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                                        \
  case ScalarType::Name: {                                           \
    Type val = eval.value<Type>();                                   \
    return getImmediateByType(v->dtype().scalar_type(), val).node(); \
  }
      AT_FORALL_SCALAR_TYPES_AND(Half, TYPE_CASE);
#undef TYPE_CASE
      default:
        LOG(FATAL) << "Unsupported datatype: " << v->dtype();
        return nullptr;
    }
    return nullptr;
  }

  static const Expr* newBinaryOpOfType(
      IRNodeType expr_type,
      const Expr* lhs,
      const Expr* rhs,
      bool option) {
    switch (expr_type) {
      case IRNodeType::kAdd:
        return new Add(lhs, rhs);
      case IRNodeType::kSub:
        return new Sub(lhs, rhs);
      case IRNodeType::kMul:
        return new Mul(lhs, rhs);
      case IRNodeType::kDiv:
        return new Div(lhs, rhs);
      case IRNodeType::kMod:
        return new Mod(lhs, rhs);
      case IRNodeType::kMax:
        return new Max(lhs, rhs, option);
      case IRNodeType::kMin:
        return new Min(lhs, rhs, option);
      case IRNodeType::kAnd:
        return new And(lhs, rhs);
      case IRNodeType::kXor:
        return new Xor(lhs, rhs);
      case IRNodeType::kLshift:
        return new Lshift(lhs, rhs);
      case IRNodeType::kRshift:
        return new Rshift(lhs, rhs);
      default:
        LOG(FATAL) << "unsupported expr_type: " << static_cast<int>(expr_type);
        return nullptr;
    }
  }

  template <typename Op>
  static const Expr* mutateBinaryOp(
      const BinaryOpNode<Op>* v,
      IRMutator* mutator,
      bool option = false) {
    const Expr* lhs = v->lhs();
    const Expr* rhs = v->rhs();
    const Expr* lhs_new = lhs->accept_mutator(mutator);
    const Expr* rhs_new = rhs->accept_mutator(mutator);

    const Expr* node = v;

    if (lhs != lhs_new || rhs != rhs_new) {
      node = newBinaryOpOfType(v->expr_type(), lhs_new, rhs_new, option);
    }

    // Can only fold if both sides are constant.
    if (!lhs_new->isConstant() || !rhs_new->isConstant()) {
      return node;
    }

    return evaluateOp(node);
  }
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
