
#include <torch/csrc/jit/codegen/cuda/evaluator_common.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

bool equals(const Val* value, const EvaluatorValue& concrete_value) {
  switch (value->getDataType().value()) {
    case DataType::Int: {
      if (!concrete_value.isInt()) {
        return false;
      }
      auto val = value->getInt();
      return val.has_value() && val.value() == concrete_value.as<int64_t>();
    }
    case DataType::Double: {
      if (!concrete_value.isDouble()) {
        return false;
      }
      auto val = value->getDouble();
      return val.has_value() && val.value() == concrete_value.as<double>();
    }
    case DataType::Bool: {
      if (!concrete_value.isBool()) {
        return false;
      }
      auto val = value->getBool();
      return val.has_value() && val.value() == concrete_value.as<bool>();
    }
    default:
      TORCH_INTERNAL_ASSERT(false);
  }
}

template <typename T>
c10::optional<EvaluatorValue> toOptionalEvaluatorValue(c10::optional<T> i) {
  if (!i) {
    return c10::nullopt;
  }
  return EvaluatorValue(i.value());
}

} // namespace

void ExpressionEvaluator::bind_(
    const Val* value,
    const EvaluatorValue& concrete_value) {
  if (equals(value, concrete_value)) {
    return;
  }
  TORCH_CHECK(value->isScalar());
  TORCH_CHECK(
      value->dtype() == DataType::Int || value->dtype() == DataType::Double ||
      value->dtype() == DataType::Bool);
  TORCH_CHECK(!value->isConstScalar(), "Tried to bind to a constant value");
  TORCH_CHECK(
      value->definition() == nullptr,
      "Tried to bind to a value that is computed in the Fusion IR: ",
      value->toInlineString(),
      " with ",
      concrete_value);
  if (value->isA<NamedScalar>()) {
    known_named_scalars_[value->as<NamedScalar>()->name()] = concrete_value;
  } else {
    known_values_[value] = concrete_value;
  }
}

void ExpressionEvaluator::bind_(
    const std::string& name,
    const EvaluatorValue& concrete_value) {
  known_named_scalars_[name] = concrete_value;
}

void ExpressionEvaluator::bind(
    ParallelType pt,
    Int::ScalarType concrete_value) {
  TORCH_INTERNAL_ASSERT(isParallelTypeThread(pt));
  if (precomputed_values_) {
    // Need to bind the thread value to integer machine
    //  in pre-computed mode.
    precomputed_values_->bindConcreteParallelTypeValue(pt, concrete_value);
  } else {
    bind(stringifyThreadSize(pt), EvaluatorValue(concrete_value));
  }
}

c10::optional<EvaluatorValue> ExpressionEvaluator::evaluate(const Val* value) {
  if (precomputed_values_ && precomputed_values_->ready()) {
    if (precomputed_values_->getMaybeValueFor(value).has_value()) {
      return toOptionalEvaluatorValue(
          precomputed_values_->getMaybeValueFor(value));
    }
  }

  auto maybe_concrete_value = getValue(value);
  if (!maybe_concrete_value.has_value()) {
    if (value->definition() != nullptr) {
      FUSER_PERF_SCOPE("ExpressionEvaluator::evaluate");
      OptInConstDispatch::handle(value->definition());
      maybe_concrete_value = getValue(value);
    }
  }
  return maybe_concrete_value;
}

c10::optional<EvaluatorValue> ExpressionEvaluator::getValue(const Val* value) {
  TORCH_INTERNAL_ASSERT(
      value->isAnInt() || value->isADouble() || value->isABool(),
      value->toString(),
      " is not a supported type in expression evaluation.");

  if (value->isScalar() && value->isConst()) {
    if (value->isADouble()) {
      return toOptionalEvaluatorValue(value->as<Double>()->value());
    }
    if (value->isABool()) {
      return toOptionalEvaluatorValue(value->as<Bool>()->value());
    }
    if (value->isAnInt()) {
      return toOptionalEvaluatorValue(value->as<Int>()->value());
    }
    TORCH_INTERNAL_ASSERT(
        false, "Data type not supported by ExpressionEvaluator");
  }

  if (value->isA<NamedScalar>()) {
    const auto it = known_named_scalars_.find(value->as<NamedScalar>()->name());
    if (it != known_named_scalars_.end()) {
      return c10::optional<EvaluatorValue>(it->second);
    }
  }

  const auto it = known_values_.find(value);
  return it != known_values_.end() ? c10::optional<EvaluatorValue>(it->second)
                                   : c10::nullopt;
}

void ExpressionEvaluator::print() const {
  std::cout << "\nEvaluation context\n";
  std::cout << "--------------------\n";
  for (const auto& kv : known_values_) {
    TORCH_INTERNAL_ASSERT(!kv.first->isConstScalar());
    std::cout << kv.first << " = " << kv.second << " ; "
              << *kv.first->getValType() << "\n";
  }

  for (const auto& kv : known_named_scalars_) {
    std::cout << kv.first << " = " << kv.second << " ;\n";
  }

  std::cout << "\nPre-computed Values\n";
  if (precomputed_values_ != nullptr) {
    precomputed_values_->print();
  }
  std::cout << "--------------------\n\n";
}

void ExpressionEvaluator::handle(const UnaryOp* uop) {
  using namespace EvaluatorValue_functions;
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
          known_values_[uop->out()] = EvaluatorValue(in->cast<int64_t>());
        } else if (uop->out()->getDataType() == DataType::Double) {
          known_values_[uop->out()] = EvaluatorValue(in->cast<double>());
        } else if (uop->out()->getDataType() == DataType::Bool) {
          known_values_[uop->out()] = EvaluatorValue(in->cast<bool>());
        } else {
          TORCH_INTERNAL_ASSERT(false, "dtype not supported in evaluator");
        }
        break;
      case UnaryOpType::Abs:
        known_values_[uop->out()] = abs(*in);
        break;
      default:
        TORCH_CHECK(
            false,
            "Unexpected operator type ",
            uop->getUnaryOpType(),
            " in ",
            uop->toString());
    }
  }
}

void ExpressionEvaluator::handle(const BinaryOp* bop) {
  using namespace EvaluatorValue_functions;
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
      case BinaryOpType::Or:
        known_values_[bop->out()] = *lhs || *rhs;
        break;
      case BinaryOpType::Xor:
        known_values_[bop->out()] = *lhs ^ *rhs;
        break;
      case BinaryOpType::Eq:
        known_values_[bop->out()] = *lhs == *rhs;
        break;
      case BinaryOpType::NE:
        known_values_[bop->out()] = *lhs != *rhs;
        break;
      case BinaryOpType::GT:
        known_values_[bop->out()] = *lhs > *rhs;
        break;
      case BinaryOpType::GE:
        known_values_[bop->out()] = *lhs >= *rhs;
        break;
      case BinaryOpType::LT:
        known_values_[bop->out()] = *lhs < *rhs;
        break;
      case BinaryOpType::LE:
        known_values_[bop->out()] = *lhs <= *rhs;
        break;
      case BinaryOpType::Max:
        known_values_[bop->out()] = max(*lhs, *rhs);
        break;
      case BinaryOpType::Min:
        known_values_[bop->out()] = min(*lhs, *rhs);
        break;
      default:
        TORCH_CHECK(
            false,
            "Unexpected operator type: ",
            bop->getBinaryOpType(),
            " in ",
            bop->toString());
    }
  }
}

void ExpressionEvaluator::handle(const TernaryOp* top) {
  using namespace EvaluatorValue_functions;
  const auto in1 = evaluate(top->in1());
  const auto in2 = evaluate(top->in2());
  const auto in3 = evaluate(top->in3());
  if (in1.has_value() && in2.has_value() && in3.has_value()) {
    switch (top->getTernaryOpType()) {
      case TernaryOpType::Where:
        known_values_[top->out()] = in1->as<bool>() ? *in2 : *in3;
        break;
      default:
        TORCH_CHECK(
            false,
            "Unexpected operator type: ",
            top->getTernaryOpType(),
            " in ",
            top->toString());
    }
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
