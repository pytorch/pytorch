
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

namespace {

template <typename T>
c10::optional<IntOrDouble> toOptionalIntOrDouble(c10::optional<T> i) {
  if (!i) {
    return c10::nullopt;
  }
  return IntOrDouble(i.value());
}

} // namespace

void ExpressionEvaluator::bind(const Val* value, IntOrDouble concrete_value) {
  TORCH_CHECK(value->isScalar());
  TORCH_CHECK(
      value->dtype() == DataType::Int || value->dtype() == DataType::Double);
  TORCH_CHECK(!value->isConstScalar(), "Tried to bind to a constant value");
  TORCH_CHECK(
      value->definition() == nullptr,
      "Tried to bind to a value that is computed in the kernel IR: ",
      value->toInlineString(),
      " with ",
      concrete_value);
  known_values_[value] = concrete_value;
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
    known_parallel_dimensions_[pt] = concrete_value;
  }
}

c10::optional<IntOrDouble> ExpressionEvaluator::evaluate(const Val* value) {
  if (precomputed_values_ && precomputed_values_->ready()) {
    if (precomputed_values_->getMaybeValueFor(value).has_value()) {
      return toOptionalIntOrDouble(
          precomputed_values_->getMaybeValueFor(value));
    }
  }

  if (value->isScalar() && value->isConst()) {
    if (value->isADouble()) {
      return toOptionalIntOrDouble(value->as<Double>()->value());
    }
    return toOptionalIntOrDouble(value->as<Int>()->value());
  } else {
    FUSER_PERF_SCOPE("kir::ExpressionEvaluator::evaluate");

    TORCH_CHECK(value->isScalar(), value->toString());
    TORCH_CHECK(
        value->dtype() == DataType::Int || value->dtype() == DataType::Double,
        value->toString());

    // Is the value known (either explicit binding or memoized)?
    const auto pre_eval_it = known_values_.find(value);
    if (pre_eval_it != known_values_.end()) {
      return pre_eval_it->second;
    }

    OptOutConstDispatch::handle(value);

    const auto post_eval_it = known_values_.find(value);
    return post_eval_it != known_values_.end()
        ? c10::optional<IntOrDouble>(post_eval_it->second)
        : c10::nullopt;
  }
  return c10::nullopt;
}

bool ExpressionEvaluator::isConst(const Val* value) {
  return ExpressionEvaluator().evaluate(value).has_value();
}

void ExpressionEvaluator::print() const {
  std::cout << "\nEvaluation context\n";
  std::cout << "--------------------\n";
  for (const auto& kv : known_values_) {
    std::cout << kv.first->toString() << " = " << kv.second << "\n";
  }
  std::cout << "\nPre-computed Values\n";
  if (precomputed_values_ != nullptr) {
    precomputed_values_->print();
  }
  std::cout << "--------------------\n\n";
}

void ExpressionEvaluator::handle(const Int* value) {
  TORCH_INTERNAL_ASSERT(!value->isConst());
  if (auto def = value->definition()) {
    OptOutConstDispatch::handle(def);
  }
}

void ExpressionEvaluator::handle(const Double* value) {
  TORCH_INTERNAL_ASSERT(!value->isConst());
  if (auto def = value->definition()) {
    OptOutConstDispatch::handle(def);
  }
}

void ExpressionEvaluator::handle(const NamedScalar* named_scalar) {
  const auto& name = named_scalar->name();
  for (auto pt : kParallelTypeThreads) {
    auto pt_val_it = known_parallel_dimensions_.find(pt);
    if (pt_val_it == known_parallel_dimensions_.end()) {
      continue;
    }
    if (name == stringifyThreadSize(pt)) {
      known_values_[named_scalar] = pt_val_it->second;
      return;
    }
  }
}

void ExpressionEvaluator::handle(const UnaryOp* unary_op) {
  const auto in = evaluate(unary_op->in());
  if (in.has_value()) {
    switch (unary_op->getUnaryOpType()) {
      case UnaryOpType::Neg:
        known_values_[unary_op->out()] = -*in;
        break;
      case UnaryOpType::Set:
        known_values_[unary_op->out()] = *in;
        break;
      case UnaryOpType::Cast:
        if (unary_op->out()->getDataType() == DataType::Int) {
          known_values_[unary_op->out()] = in->cast<int64_t>();
        } else if (unary_op->out()->getDataType() == DataType::Double) {
          known_values_[unary_op->out()] = in->cast<double>();
        } else {
          TORCH_INTERNAL_ASSERT(false, "dtype not supported in evaluator");
        }
        break;
      default:
        TORCH_CHECK(
            false,
            "Unexpected operator type ",
            unary_op->getUnaryOpType(),
            " in ",
            unary_op->toString());
    }
  }
}

void ExpressionEvaluator::handle(const BinaryOp* binary_op) {
  using namespace IntOrDouble_functions;
  const auto lhs = evaluate(binary_op->lhs());
  const auto rhs = evaluate(binary_op->rhs());
  if (lhs.has_value() && rhs.has_value()) {
    switch (binary_op->getBinaryOpType()) {
      case BinaryOpType::Add:
        known_values_[binary_op->out()] = *lhs + *rhs;
        break;
      case BinaryOpType::Sub:
        known_values_[binary_op->out()] = *lhs - *rhs;
        break;
      case BinaryOpType::Mul:
        known_values_[binary_op->out()] = *lhs * *rhs;
        break;
      case BinaryOpType::Div:
        TORCH_CHECK(*rhs != 0);
        known_values_[binary_op->out()] = *lhs / *rhs;
        break;
      case BinaryOpType::Mod:
        TORCH_CHECK(*rhs != 0);
        known_values_[binary_op->out()] = *lhs % *rhs;
        break;
      case BinaryOpType::CeilDiv:
        TORCH_CHECK(*rhs != 0);
        known_values_[binary_op->out()] = ceildiv(*lhs, *rhs);
        break;
      case BinaryOpType::And:
        known_values_[binary_op->out()] = Int::ScalarType(*lhs && *rhs);
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
