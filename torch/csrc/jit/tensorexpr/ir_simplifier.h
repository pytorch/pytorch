#pragma once

#include "torch/csrc/jit/tensorexpr/eval.h"
#include "torch/csrc/jit/tensorexpr/ir_mutator.h"
#include "torch/csrc/jit/tensorexpr/ir_visitor.h"
#include "torch/csrc/jit/tensorexpr/types.h"

namespace torch {
namespace jit {
namespace tensorexpr {

// Uses the evaluator to fold an operation with constant terms.
// Expr v must be evaluatable without Vars.
static Expr* evaluateOp(const Expr* v) {
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
} // namespace tensorexpr

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

/* Interprets expr as an Immediate and returns the value as type T. */
template <typename T>
T immediateAs(const Expr* expr) {
  T val{0};
  switch (expr->dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                                          \
  case ScalarType::Name:                                               \
    if (const Name##Imm* imm = dynamic_cast<const Name##Imm*>(expr)) { \
      val = imm->value();                                              \
    } else {                                                           \
      LOG(FATAL) << "Bad expr: " << *expr << "\n";                     \
    }                                                                  \
    break;
    AT_FORALL_SCALAR_TYPES_AND2(Half, Bool, TYPE_CASE);
#undef TYPE_CASE
    default:
      LOG(FATAL) << "Unsupported datatype: " << expr->dtype();
  }

  return val;
}

/* Takes a LinearForm and converts it to Mul + (Add/Sub). */
const Expr* expandLinearForm(const LinearForm* v, IRMutator* mutator) {
  const Expr* mul = nullptr;
  const Expr* A = v->getA();
  const Expr* B = v->getB();
  const Expr* X = v->getX();
  // we only really care about 0 and 1, so double should be fine.
  double Aval = immediateAs<double>(A);
  double Bval = immediateAs<double>(B);

  // First handle A.
  if (Aval == 0) {
    if (Bval == 0) {
      return getImmediateByType(X->dtype(), 0).node();
    }
    return B;
  } else if (Aval == 1) {
    mul = X;
  } else if (Aval == -1) {
    return new Sub(B, X);
  } else if (Aval < 0) {
    // Negate A.
    ExprHandle zero = getImmediateByType(A->dtype(), 0);
    Sub* A_Sub = new Sub(zero.node(), A);

    return new Sub(B, new Mul(X, evaluateOp(A_Sub)));
  } else {
    mul = new Mul(X, A);
  }

  if (Bval == 0) {
    return mul;
  }

  return new Add(mul, B);
}

/* Expand any remaining LinearTerms into their component pieces */
class LinearFormExpander : public IRMutator {
 public:
  const Expr* mutate(const LinearForm* v) {
    return expandLinearForm(v, this);
  }
};

/* Simplify the IR by combining arithmetic expressions over a common term.
 */
class IRSimplifier : public IRMutator {
 public:
  const Expr* mutate(const Add* v) override {
    const Expr* lhs = v->lhs();
    const Expr* rhs = v->rhs();
    const Expr* lhs_new = lhs->accept_mutator(this);
    const Expr* rhs_new = rhs->accept_mutator(this);

    // Constant Folding.
    if (lhs_new->isConstant() && rhs_new->isConstant()) {
      const Expr* result = evaluateOp(v);
      return result;
    }

    const LinearForm* lhsLinear = dynamic_cast<const LinearForm*>(lhs_new);
    const LinearForm* rhsLinear = dynamic_cast<const LinearForm*>(rhs_new);

    if (lhsLinear && rhsLinear) {
      // Can add two LinearTerms if they reference the same Var.
      if (lhsLinear->getX() == rhsLinear->getX()) {
        Add* A_Add = new Add(lhsLinear->getA(), rhsLinear->getA());
        Add* B_Add = new Add(lhsLinear->getB(), rhsLinear->getB());

        LinearForm* linear = new LinearForm(
            lhsLinear->getX(), evaluateOp(A_Add), evaluateOp(B_Add));
        return linear;
      }

      // otherwise cannot simplify further.
      return expandAndRecurse(v->expr_type(), lhs_new, rhs_new);
    }

    // Can add a scalar into the B term of LinearTerm.
    if (lhsLinear && rhs_new->isConstant()) {
      Add* B_Add = new Add(lhsLinear->getB(), rhs_new);
      LinearForm* linear = new LinearForm(
          lhsLinear->getX(), lhsLinear->getA(), evaluateOp(B_Add));
      return linear;
    }

    if (rhsLinear && lhs_new->isConstant()) {
      Add* B_Add = new Add(rhsLinear->getB(), lhs_new);
      LinearForm* linear = new LinearForm(
          rhsLinear->getX(), rhsLinear->getA(), evaluateOp(B_Add));
      return linear;
    }

    // Can create a LinearTerm over any sub expression.
    if (lhs_new->isConstant()) {
      LinearForm* linear = new LinearForm(rhs_new);
      linear->setB(evaluateOp(lhs_new));
      return linear;
    }

    if (rhs_new->isConstant()) {
      LinearForm* linear = new LinearForm(lhs_new);
      linear->setB(evaluateOp(rhs_new));
      return linear;
    }

    /// Broadcasts are a bit more involved.
    if (const Broadcast* bc = dynamic_cast<const Broadcast*>(lhs_new)) {
      if (const Expr* ret = handleBroadcastAdd(bc, rhs_new)) {
        return ret;
      }
    }

    if (const Broadcast* bc = dynamic_cast<const Broadcast*>(rhs_new)) {
      if (const Expr* ret = handleBroadcastAdd(bc, lhs_new)) {
        return ret;
      }
    }

    // No change.
    if (lhs == lhs_new && rhs == rhs_new) {
      return v;
    }

    // Cannot simplify.
    return expandAndRecurse(v->expr_type(), lhs_new, rhs_new);
  }

  const Expr* mutate(const Sub* v) override {
    const Expr* lhs = v->lhs();
    const Expr* rhs = v->rhs();
    const Expr* lhs_new = lhs->accept_mutator(this);
    const Expr* rhs_new = rhs->accept_mutator(this);

    // Constant Folding.
    if (lhs_new->isConstant() && rhs_new->isConstant()) {
      const Expr* result = evaluateOp(v);
      return result;
    }

    const LinearForm* lhsLinear = dynamic_cast<const LinearForm*>(lhs_new);
    const LinearForm* rhsLinear = dynamic_cast<const LinearForm*>(rhs_new);

    if (lhsLinear && rhsLinear) {
      // Can sub two LinearTerms if they reference the same Var.
      if (lhsLinear->getX() == rhsLinear->getX()) {
        Sub* A_Sub = new Sub(lhsLinear->getA(), rhsLinear->getA());
        Sub* B_Sub = new Sub(lhsLinear->getB(), rhsLinear->getB());

        LinearForm* linear = new LinearForm(
            lhsLinear->getX(), evaluateOp(A_Sub), evaluateOp(B_Sub));
        return linear;
      }

      // otherwise cannot simplify further.
      return expandAndRecurse(v->expr_type(), lhs_new, rhs_new);
    }

    // Can just sub from B term if LHS is a LinearTerm.
    if (lhsLinear && rhs_new->isConstant()) {
      Sub* B_Sub = new Sub(lhsLinear->getB(), rhs_new);
      LinearForm* linear = new LinearForm(
          lhsLinear->getX(), lhsLinear->getA(), evaluateOp(B_Sub));
      return linear;
    }

    // Slightly more complicated if the RHS is LinearTerm.
    if (rhsLinear && lhs_new->isConstant()) {
      // The linear needs to be negated.
      ExprHandle zero = getImmediateByType(rhsLinear->getA()->dtype(), 0);
      Sub* A_Sub = new Sub(zero.node(), rhsLinear->getA());
      Sub* B_Sub = new Sub(rhsLinear->getB(), lhs_new);
      LinearForm* linear = new LinearForm(
          rhsLinear->getX(), evaluateOp(A_Sub), evaluateOp(B_Sub));
      return linear;
    }

    // Can create a new LinearTerm, but since the B term is defined as Add we
    // must negate it.
    if (rhs_new->isConstant()) {
      LinearForm* linear = new LinearForm(lhs_new);

      ExprHandle zero = getImmediateByType(linear->getA()->dtype(), 0);
      Sub* B_Sub = new Sub(zero.node(), rhs_new);
      linear->setB(evaluateOp(B_Sub));
      return linear;
    }

    // Can create a new LinearTerm with the A term -1 to negate the Expr.
    if (lhs_new->isConstant()) {
      // Negate by using -1 as the first linear.
      ExprHandle negOne = getImmediateByType(rhs_new->dtype(), -1);
      LinearForm* linear =
          new LinearForm(rhs_new, negOne.node(), evaluateOp(lhs_new));
      return linear;
    }

    // Nothing to do.
    if (lhs == lhs_new && rhs == rhs_new) {
      return v;
    }

    // Cannot simplify.
    return expandAndRecurse(v->expr_type(), lhs_new, rhs_new);
  }

  const Expr* mutate(const Mul* v) override {
    const Expr* lhs = v->lhs();
    const Expr* rhs = v->rhs();
    const Expr* lhs_new = lhs->accept_mutator(this);
    const Expr* rhs_new = rhs->accept_mutator(this);

    // Constant Folding.
    if (lhs_new->isConstant() && rhs_new->isConstant()) {
      return evaluateOp(v);
    }

    const LinearForm* lhsLinear = dynamic_cast<const LinearForm*>(lhs_new);
    const LinearForm* rhsLinear = dynamic_cast<const LinearForm*>(rhs_new);

    if (lhsLinear && rhsLinear) {
      // Lets not get into higher order terms.
      return expandAndRecurse(v->expr_type(), lhs_new, rhs_new);
    }

    // Easy to simplify into an existing LinearTerm by multiplying A and B.
    if (lhsLinear && rhs_new->isConstant()) {
      Mul* A_Mul = new Mul(lhsLinear->getA(), rhs_new);
      Mul* B_Mul = new Mul(lhsLinear->getB(), rhs_new);
      LinearForm* linear = new LinearForm(
          lhsLinear->getX(), evaluateOp(A_Mul), evaluateOp(B_Mul));
      return linear;
    }

    if (rhsLinear && lhs_new->isConstant()) {
      Mul* A_Mul = new Mul(rhsLinear->getA(), lhs_new);
      Mul* B_Mul = new Mul(rhsLinear->getB(), lhs_new);
      LinearForm* linear = new LinearForm(
          rhsLinear->getX(), evaluateOp(A_Mul), evaluateOp(B_Mul));
      return linear;
    }

    // Easy to create a new LinearTerm by setting term A.
    if (lhs_new->isConstant()) {
      LinearForm* linear = new LinearForm(rhs_new);
      linear->setA(evaluateOp(lhs_new));
      return linear;
    }

    if (rhs_new->isConstant()) {
      LinearForm* linear = new LinearForm(lhs_new);
      linear->setA(evaluateOp(rhs_new));
      return linear;
    }

    // Broadcasts have special logic.
    if (const Broadcast* bc = dynamic_cast<const Broadcast*>(lhs_new)) {
      if (const Expr* ret = handleBroadcastMul(bc, rhs_new)) {
        return ret;
      }
    }

    if (const Broadcast* bc = dynamic_cast<const Broadcast*>(rhs_new)) {
      if (const Expr* ret = handleBroadcastMul(bc, lhs_new)) {
        return ret;
      }
    }

    // Cannot be simplified, just exit.
    if (lhs == lhs_new && rhs == rhs_new) {
      return v;
    }

    return expandAndRecurse(v->expr_type(), lhs_new, rhs_new);
  }

  const Expr* mutate(const Div* v) override {
    // TODO div simplification will require a rational node.
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
      const Expr* new_child = p->accept_mutator(this);
      new_params.push_back(new_child);

      changed |= p != new_child;
      allConstant &= new_child->isConstant();
    }

    const Expr* node = v;
    if (changed) {
      node = new Intrinsics(v->op_type(), new_params);
    }

    if (!allConstant || !v->isPure()) {
      return node;
    }

    return evaluateOp(node);
  }

  const Expr* mutate(const Cast* v) override {
    if (v->src_value()->isConstant()) {
      return evaluateOp(v);
    }

    return v;
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

  static const Expr* simplify(const Expr* e) {
    IRSimplifier simplifier;
    e = e->accept_mutator(&simplifier);

    // There may be terms left in the IR, expand them.
    LinearFormExpander expander;
    e = e->accept_mutator(&expander);

    return e;
  }

  static ExprHandle simplify(const ExprHandle& e) {
    return ExprHandle(simplify(e.node()));
  }

  static Stmt* simplify(Stmt* s) {
    IRSimplifier simplifier;
    s = s->accept_mutator(&simplifier);

    // There may be terms left in the IR, expand them.
    LinearFormExpander expander;
    s = s->accept_mutator(&expander);
    return s;
  }

 private:
  /* Expands lhs and rhs if they are LinearTerms, creating a new op to hold
   * them. If either side expands to a constant term, attempt simplification of
   * the new op. */
  const Expr* expandAndRecurse(
      IRNodeType expr_type,
      const Expr* lhs,
      const Expr* rhs) {
    if (const LinearForm* lhsLinear = dynamic_cast<const LinearForm*>(lhs)) {
      lhs = expandLinearForm(lhsLinear, this);
    }
    if (const LinearForm* rhsLinear = dynamic_cast<const LinearForm*>(rhs)) {
      rhs = expandLinearForm(rhsLinear, this);
    }
    const Expr* result = newBinaryOpOfType(expr_type, lhs, rhs, false);

    // lhs or rhs can become constant during expansion, if either is now
    // constant we can keep merging into a linear term. Have another attempt to
    // simplify the new op.
    if (lhs->isConstant() || rhs->isConstant()) {
      return result->accept_mutator(this);
    }

    return result;
  }

  /* Handles optimization cases for Broadcast() + Other */
  const Expr* handleBroadcastAdd(const Broadcast* bc, const Expr* other) {
    if (bc->value()->isConstant() && immediateAs<int>(bc->value()) == 0) {
      return other;
    }

    if (const Ramp* r = dynamic_cast<const Ramp*>(other)) {
      // Add the broadcast to the start of the Ramp.
      const Expr* ret =
          new Ramp(new Add(bc->value(), r->base()), r->stride(), r->lanes());
      return ret->accept_mutator(this);
    }

    return nullptr;
  }

  /* Handles optimization cases for Broadcast() * Other */
  const Expr* handleBroadcastMul(const Broadcast* bc, const Expr* other) {
    if (bc->value()->isConstant() && immediateAs<int>(bc->value()) == 1) {
      return other;
    }

    if (const Ramp* r = dynamic_cast<const Ramp*>(other)) {
      // Multiply both start and stride by the broadcast value.
      const Expr* ret = new Ramp(
          new Mul(bc->value(), r->base()),
          new Mul(bc->value(), r->stride()),
          r->lanes());
      return ret->accept_mutator(this);
    }

    return nullptr;
  }
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
