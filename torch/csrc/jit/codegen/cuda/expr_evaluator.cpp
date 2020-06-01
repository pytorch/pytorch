
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

void EvaluationContext::bind(const Val* value, Int::ScalarType concrete_value) {
  TORCH_CHECK(value->isAnInt());
  TORCH_CHECK(!value->as<Int>()->value().has_value());
  TORCH_CHECK(fusion_->origin(value) == nullptr);
  bindings_[value] = concrete_value;
}

c10::optional<Int::ScalarType> EvaluationContext::concreteValue(
    const Val* value) const {
  const auto it = bindings_.find(value);
  return (it != bindings_.end()) ? c10::optional<Int::ScalarType>(it->second)
                                 : c10::nullopt;
}

void EvaluationContext::print() const {
  std::cout << "\nEvaluation context\n";
  std::cout << "--------------------\n";
  for (const auto& kv : bindings_) {
    const auto val = kv.first->as<Int>();
    std::cout << "i" << val->name() << " = " << kv.second;
    if (!val->isSymbolic()) {
      std::cout << " ; original value = " << *val->value();
    }
    std::cout << "\n";
  }
  std::cout << "--------------------\n\n";
}

c10::optional<Int::ScalarType> ExpressionEvaluator::evaluate(
    const Statement* expr,
    const EvaluationContext* context) {
  TORCH_CHECK(context != nullptr);
  ExpressionEvaluator evaluator(context);
  evaluator.OptInConstDispatch::handle(expr);
  return evaluator.result_;
}

void ExpressionEvaluator::handle(const Int* i) {
  if (i->value().has_value()) {
    result_ = i->value();
  } else if (const auto* def = context_->fusion()->origin(i)) {
    result_ = evaluate(def, context_);
  } else {
    const auto& bound_value = context_->concreteValue(i);
    if (bound_value.has_value()) {
      result_ = bound_value;
    }
  }
}

void ExpressionEvaluator::handle(const NamedScalar* i) {
  // nothing to do, leave the result "unknown"
}

void ExpressionEvaluator::handle(const UnaryOp* uop) {
  const auto in = evaluate(uop->in(), context_);
  if (in.has_value()) {
    switch (uop->getUnaryOpType()) {
      case UnaryOpType::Neg:
        result_ = -*in;
        break;
      case UnaryOpType::Cast:
        result_ = *in;
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

void ExpressionEvaluator::handle(const BinaryOp* bop) {
  TORCH_CHECK(bop->out()->isAnInt()); // not really needed
  const auto lhs = evaluate(bop->lhs(), context_);
  const auto rhs = evaluate(bop->rhs(), context_);
  if (lhs.has_value() && rhs.has_value()) {
    switch (bop->getBinaryOpType()) {
      case BinaryOpType::Add:
        result_ = *lhs + *rhs;
        break;
      case BinaryOpType::Sub:
        result_ = *lhs - *rhs;
        break;
      case BinaryOpType::Mul:
        result_ = *lhs * *rhs;
        break;
      case BinaryOpType::Div:
        TORCH_CHECK(*rhs != 0);
        result_ = *lhs / *rhs;
        break;
      case BinaryOpType::Mod:
        TORCH_CHECK(*rhs != 0);
        result_ = *lhs % *rhs;
        break;
      case BinaryOpType::CeilDiv:
        TORCH_CHECK(*rhs != 0);
        result_ = (*lhs + *rhs - 1) / *rhs;
        break;
      case BinaryOpType::And:
        result_ = Int::ScalarType(*lhs && *rhs);
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

} // namespace fuser
} // namespace jit
} // namespace torch
