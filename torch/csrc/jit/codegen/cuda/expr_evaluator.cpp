
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

void EvaluationContext::bind(const Val* value, Int::ScalarType concrete_value) {
  TORCH_INTERNAL_ASSERT(
      value->isAnInt(),
      "Expressoin Evaluation does not support values other than integers at this time.");

  if (value->isConstScalar()) {
    auto const_value = value->as<Int>()->value().value();
    TORCH_INTERNAL_ASSERT(
        concrete_value == const_value,
        "Tried to bind ",
        concrete_value,
        " to ",
        value,
        " however ",
        value,
        " is set to a constant ",
        const_value);
  }

  TORCH_INTERNAL_ASSERT(
      fusion_->origin(value) == nullptr,
      "Tried to bind to a value that is computed in the fusion IR. ",
      "Can only bind to symbolic values to the fusion that do not have an origin expr.");

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
    std::cout << kv.first << " = " << kv.second;
    if (kv.first->isConstScalar()) {
      std::cout << " ; original value = "
                << kv.first->as<Int>()->value().value();
    }
    std::cout << "\n";
  }
  std::cout << "--------------------\n\n";
}

c10::optional<Int::ScalarType> ExpressionEvaluator::evaluate(
    Val* val,
    const EvaluationContext* context) {
  TORCH_CHECK(context != nullptr);
  ExpressionEvaluator evaluator(context);
  evaluator.traverseFrom(context->fusion(), {val}, false);
  return evaluator.value(val);
}

c10::optional<Int::ScalarType> ExpressionEvaluator::value(
    const Statement* stmt) const {
  const auto it = values_.find(stmt);
  return (it != values_.end()) ? c10::optional<Int::ScalarType>(it->second)
                               : c10::nullopt;
}

void ExpressionEvaluator::handle(NamedScalar* i) {
  if (i->isAnInt()) {
    const auto& bound_value = context_->concreteValue(i);
    if (bound_value.has_value()) {
      values_[i] = *bound_value;
    }
  }
}

void ExpressionEvaluator::handle(Int* i) {
  if (i->value().has_value()) {
    values_[i] = *i->value();
  } else if (const auto* def = context_->fusion()->origin(i)) {
    const auto& def_result = value(def);
    if (def_result.has_value()) {
      values_[i] = *def_result;
    }
  } else {
    const auto& bound_value = context_->concreteValue(i);
    if (bound_value.has_value()) {
      values_[i] = *bound_value;
    }
  }
}

void ExpressionEvaluator::handle(UnaryOp* uop) {
  const auto in = value(uop->in());
  if (in.has_value()) {
    switch (uop->getUnaryOpType()) {
      case UnaryOpType::Neg:
        values_[uop] = -*in;
        break;
      case UnaryOpType::Cast:
        values_[uop] = *in;
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

void ExpressionEvaluator::handle(BinaryOp* bop) {
  const auto lhs = value(bop->lhs());
  const auto rhs = value(bop->rhs());
  if (lhs.has_value() && rhs.has_value()) {
    switch (bop->getBinaryOpType()) {
      case BinaryOpType::Add:
        values_[bop] = *lhs + *rhs;
        break;
      case BinaryOpType::Sub:
        values_[bop] = *lhs - *rhs;
        break;
      case BinaryOpType::Mul:
        values_[bop] = *lhs * *rhs;
        break;
      case BinaryOpType::Div:
        TORCH_CHECK(*rhs != 0);
        values_[bop] = *lhs / *rhs;
        break;
      case BinaryOpType::Mod:
        TORCH_CHECK(*rhs != 0);
        values_[bop] = *lhs % *rhs;
        break;
      case BinaryOpType::CeilDiv:
        TORCH_CHECK(*rhs != 0);
        values_[bop] = (*lhs + *rhs - 1) / *rhs;
        break;
      case BinaryOpType::And:
        values_[bop] = Int::ScalarType(*lhs && *rhs);
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

void ExpressionEvaluator::handle(kir::NamedScalar* i) {
  if (i->isAnInt()) {
    const auto& bound_value = context_->concreteValue(i);
    if (bound_value.has_value()) {
      values_[i] = *bound_value;
    }
  }
}

void ExpressionEvaluator::handle(kir::Int* i) {
  if (i->value().has_value()) {
    values_[i] = *i->value();
  } else if (const auto* def = context_->fusion()->origin(i)) {
    const auto& def_result = value(def);
    if (def_result.has_value()) {
      values_[i] = *def_result;
    }
  } else {
    const auto& bound_value = context_->concreteValue(i);
    if (bound_value.has_value()) {
      values_[i] = *bound_value;
    }
  }
}

void ExpressionEvaluator::handle(kir::UnaryOp* uop) {
  const auto in = value(uop->in());
  if (in.has_value()) {
    switch (uop->getUnaryOpType()) {
      case UnaryOpType::Neg:
        values_[uop] = -*in;
        break;
      case UnaryOpType::Cast:
        values_[uop] = *in;
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

void ExpressionEvaluator::handle(kir::BinaryOp* bop) {
  const auto lhs = value(bop->lhs());
  const auto rhs = value(bop->rhs());
  if (lhs.has_value() && rhs.has_value()) {
    switch (bop->getBinaryOpType()) {
      case BinaryOpType::Add:
        values_[bop] = *lhs + *rhs;
        break;
      case BinaryOpType::Sub:
        values_[bop] = *lhs - *rhs;
        break;
      case BinaryOpType::Mul:
        values_[bop] = *lhs * *rhs;
        break;
      case BinaryOpType::Div:
        TORCH_CHECK(*rhs != 0);
        values_[bop] = *lhs / *rhs;
        break;
      case BinaryOpType::Mod:
        TORCH_CHECK(*rhs != 0);
        values_[bop] = *lhs % *rhs;
        break;
      case BinaryOpType::CeilDiv:
        TORCH_CHECK(*rhs != 0);
        values_[bop] = (*lhs + *rhs - 1) / *rhs;
        break;
      case BinaryOpType::And:
        values_[bop] = Int::ScalarType(*lhs && *rhs);
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

} // namespace fuser
} // namespace jit
} // namespace torch
