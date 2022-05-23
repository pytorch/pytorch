
#include <torch/csrc/jit/codegen/cuda/evaluator_common.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

void ExpressionEvaluator::bind(Val* value, Int::ScalarType concrete_value) {
  TORCH_CHECK(value->isAnInt());
  auto val = value->getInt();
  if (val.has_value() && val.value() == concrete_value) {
    return;
  }
  TORCH_CHECK(!value->isConstScalar(), "Tried to bind to a constant value");
  TORCH_CHECK(
      value->definition() == nullptr,
      "Tried to bind to a value that is computed in the fusion IR");
  known_values_[value] = concrete_value;
}

c10::optional<Int::ScalarType> ExpressionEvaluator::evaluate(Val* value) {
  if (evaluator_precomputed_integers_ != nullptr) {
    return evaluator_precomputed_integers_->getMaybeValueFor(value);
  } else {
    auto maybe_concrete_value = getValue(value);
    if (!maybe_concrete_value.has_value()) {
      if (value->definition() != nullptr) {
        OptOutDispatch::handle(value->definition());
        maybe_concrete_value = getValue(value);
      }
    }
    return maybe_concrete_value;
  }
  return c10::nullopt;
}

void ExpressionEvaluator::print() const {
  std::cout << "\nEvaluation context\n";
  std::cout << "--------------------\n";
  for (const auto& kv : known_values_) {
    TORCH_INTERNAL_ASSERT(!kv.first->isConstScalar());
    std::cout << kv.first << " = " << kv.second << " ; "
              << *kv.first->getValType() << "\n";
  }
  std::cout << "--------------------\n\n";
}

c10::optional<Int::ScalarType> ExpressionEvaluator::getValue(Val* value) {
  TORCH_INTERNAL_ASSERT(
      value->isAnInt(),
      "Expression Evaluation does not support values other than integers at this time.");

  if (value->getValType().value() == ValType::Scalar) {
    if (value->as<Int>()->value().has_value()) {
      return value->as<Int>()->value();
    }
  }

  const auto it = known_values_.find(value);
  return it != known_values_.end() ? c10::optional<Int::ScalarType>(it->second)
                                   : c10::nullopt;
}

void ExpressionEvaluator::handle(UnaryOp* uop) {
  const auto in = evaluate(uop->in());
  if (in.has_value()) {
    switch (uop->getUnaryOpType()) {
      case UnaryOpType::Neg:
        known_values_[uop->out()] = -*in;
        break;
      case UnaryOpType::Cast:
        known_values_[uop->out()] = *in;
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

void ExpressionEvaluator::handle(BinaryOp* bop) {
  const auto lhs = evaluate(bop->lhs());
  const auto rhs = evaluate(bop->rhs());
  if (lhs.has_value() && rhs.has_value()) {
    switch (bop->getBinaryOpType()) {
      case BinaryOpType::Add:
        known_values_[bop->out()] = *lhs + *rhs;
        break;
      case BinaryOpType::Sub:
        known_values_[bop->out()] = *lhs - *rhs;
        break;
      case BinaryOpType::Mul:
        known_values_[bop->out()] = *lhs * *rhs;
        break;
      case BinaryOpType::Div:
        TORCH_CHECK(*rhs != 0);
        known_values_[bop->out()] = *lhs / *rhs;
        break;
      case BinaryOpType::Mod:
        TORCH_CHECK(*rhs != 0);
        known_values_[bop->out()] = *lhs % *rhs;
        break;
      case BinaryOpType::CeilDiv:
        TORCH_CHECK(*rhs != 0);
        known_values_[bop->out()] = (*lhs + *rhs - 1) / *rhs;
        break;
      case BinaryOpType::And:
        known_values_[bop->out()] = Int::ScalarType(*lhs && *rhs);
        break;
      case BinaryOpType::Max:
        known_values_[bop->out()] = std::max(*lhs, *rhs);
        break;
      case BinaryOpType::Min:
        known_values_[bop->out()] = std::min(*lhs, *rhs);
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
