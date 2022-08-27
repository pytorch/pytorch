
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

namespace {

bool equals(Val* value, const IntOrDouble& concrete_value) {
  switch (value->getDataType().value()) {
    case DataType::Int: {
      if (!concrete_value.is_int()) {
        return false;
      }
      auto val = value->getInt();
      return val.has_value() && val.value() == concrete_value.as<int64_t>();
    }
    case DataType::Double: {
      if (concrete_value.is_int()) {
        return false;
      }
      auto val = value->getDouble();
      return val.has_value() && val.value() == concrete_value.as<double>();
    }
    default:
      TORCH_INTERNAL_ASSERT(false);
  }
}

template <typename T>
c10::optional<IntOrDouble> toOptionalIntOrDouble(c10::optional<T> i) {
  if (!i) {
    return c10::nullopt;
  }
  return IntOrDouble(i.value());
}

} // namespace

void ExpressionEvaluator::bind(Val* value, const IntOrDouble& concrete_value) {
  if (equals(value, concrete_value)) {
    return;
  }
  TORCH_CHECK(!value->isConstScalar(), "Tried to bind to a constant value");
  TORCH_CHECK(
      value->definition() == nullptr,
      "Tried to bind to a value that is computed in the fusion IR");
  known_values_[value] = concrete_value;
}

c10::optional<IntOrDouble> ExpressionEvaluator::evaluate(Val* value) {
  if (evaluator_precomputed_values_ != nullptr) {
    return toOptionalIntOrDouble(
        evaluator_precomputed_values_->getMaybeValueFor(value));
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

c10::optional<IntOrDouble> ExpressionEvaluator::getValue(Val* value) {
  TORCH_INTERNAL_ASSERT(
      value->isAnInt() || value->isADouble(),
      "Expression Evaluation does not support values other than integers at this time.");

  if (value->getValType().value() == ValType::Scalar) {
    if (value->isAnInt() && value->as<Int>()->value().has_value()) {
      return toOptionalIntOrDouble(value->as<Int>()->value());
    }
    if (value->isADouble() && value->as<Double>()->value().has_value()) {
      return toOptionalIntOrDouble(value->as<Double>()->value());
    }
  }

  const auto it = known_values_.find(value);
  return it != known_values_.end() ? c10::optional<IntOrDouble>(it->second)
                                   : c10::nullopt;
}

void ExpressionEvaluator::handle(UnaryOp* uop) {
  const auto in = evaluate(uop->in());
  if (in.has_value()) {
    switch (uop->getUnaryOpType()) {
      case UnaryOpType::Neg:
        known_values_[uop->out()] = -*in;
        break;
      case UnaryOpType::Set:
        known_values_[uop->out()] = *in;
        break;
      case UnaryOpType::Cast:
        if (uop->out()->getDataType() == DataType::Int) {
          known_values_[uop->out()] = in->cast<int64_t>();
        } else if (uop->out()->getDataType() == DataType::Double) {
          known_values_[uop->out()] = in->cast<double>();
        } else {
          TORCH_INTERNAL_ASSERT(false, "dtype not supported in evaluator");
        }
        break;
      default:
        TORCH_CHECK(
            !"Unexpected operator type ",
            uop->getUnaryOpType(),
            " in ",
            uop->toString());
    }
  }
}

void ExpressionEvaluator::handle(BinaryOp* bop) {
  using namespace IntOrDouble_functions;
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
        known_values_[bop->out()] = ceildiv(*lhs, *rhs);
        break;
      case BinaryOpType::And:
        known_values_[bop->out()] = *lhs && *rhs;
        break;
      case BinaryOpType::Max:
        known_values_[bop->out()] = max(*lhs, *rhs);
        break;
      case BinaryOpType::Min:
        known_values_[bop->out()] = min(*lhs, *rhs);
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
