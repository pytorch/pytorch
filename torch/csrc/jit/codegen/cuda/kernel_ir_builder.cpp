#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

Val* IrBuilder::newResult(DataType dtype) {
  switch (dtype) {
    case DataType::Bool:
      return create<Bool>(c10::nullopt);
    case DataType::Double:
      return create<Double>(c10::nullopt);
    case DataType::Int:
      return create<Int>(c10::nullopt);
    default:
      TORCH_CHECK(false, "Unexpected data type");
  }
}

Val* IrBuilder::newArithmeticExpr(BinaryOpType op_type, Val* lhs, Val* rhs) {
  TORCH_CHECK(lhs->dtype() == rhs->dtype(), "Incompatible operand types");
  auto result = newResult(lhs->dtype());
  create<BinaryOp>(op_type, result, lhs, rhs);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return result;
}

Val* IrBuilder::newLogicExpr(BinaryOpType op_type, Val* lhs, Val* rhs) {
  auto result = create<Bool>(c10::nullopt);
  create<BinaryOp>(op_type, result, lhs, rhs);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return result;
}

Val* IrBuilder::whereExpr(Val* pred, Val* lhs, Val* rhs) {
  TORCH_CHECK(lhs->dtype() == rhs->dtype(), "Incompatible operand types");
  auto result = newResult(lhs->dtype());
  create<TernaryOp>(TernaryOpType::Where, result, pred, lhs, rhs);
  return result;
}

Val* IrBuilder::negExpr(Val* val) {
  auto result = newResult(val->dtype());
  create<UnaryOp>(UnaryOpType::Neg, result, val);
  return result;
}

Val* IrBuilder::notExpr(Val* val) {
  auto result = newResult(val->dtype());
  create<UnaryOp>(UnaryOpType::Not, result, val);
  return result;
}

Val* IrBuilder::setExpr(Val* val) {
  auto result = newResult(val->dtype());
  create<UnaryOp>(UnaryOpType::Set, result, val);
  return result;
}

Val* IrBuilder::setExprNamedScalar(const std::string& name, Val* val) {
  auto result = create<NamedScalar>(name, val->dtype());
  create<UnaryOp>(UnaryOpType::Set, result, val);
  return result;
}

Val* IrBuilder::addressExprNamedScalar(const std::string& name, Val* val) {
  auto result = create<NamedScalar>(name, DataType::Int);
  create<UnaryOp>(UnaryOpType::Address, result, val);
  return result;
}

Val* IrBuilder::andExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::And, lhs, rhs);
}

Val* IrBuilder::eqExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::Eq, lhs, rhs);
}

Val* IrBuilder::gtExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::GT, lhs, rhs);
}

Val* IrBuilder::ltExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::LT, lhs, rhs);
}

Val* IrBuilder::leExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::LE, lhs, rhs);
}

Val* IrBuilder::geExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::GE, lhs, rhs);
}

Val* IrBuilder::addExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Add, lhs, rhs);
}

Val* IrBuilder::subExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Sub, lhs, rhs);
}

Val* IrBuilder::mulExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Mul, lhs, rhs);
}

Val* IrBuilder::divExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Div, lhs, rhs);
}

Val* IrBuilder::ceilDivExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::CeilDiv, lhs, rhs);
}

Val* IrBuilder::modExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Mod, lhs, rhs);
}

Val* IrBuilder::maxExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Max, lhs, rhs);
}

Val* IrBuilder::minExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Min, lhs, rhs);
}

Int* IrBuilder::zeroVal() {
  if (zero_ == nullptr) {
    zero_ = create<kir::Int>(0);
  }
  return zero_;
}

Int* IrBuilder::oneVal() {
  if (one_ == nullptr) {
    one_ = create<kir::Int>(1);
  }
  return one_;
}

Bool* IrBuilder::falseVal() {
  if (false_ == nullptr) {
    false_ = create<kir::Bool>(false);
  }
  return false_;
}

Bool* IrBuilder::trueVal() {
  if (true_ == nullptr) {
    true_ = create<kir::Bool>(true);
  }
  return true_;
}

NamedScalar* IrBuilder::magicZeroVal() {
  if (magic_zero_ == nullptr) {
    magic_zero_ = create<kir::NamedScalar>("nvfuser_zero", DataType::Int);
  }
  return magic_zero_;
}

Val* SimplifyingIrBuilder::negExpr(Val* val) {
  if (auto int_val = dynamic_cast<kir::Int*>(val)) {
    if (int_val->isConst()) {
      return create<Int>(-int_val->value().value());
    }
  }
  return IrBuilder::negExpr(val);
}

Val* SimplifyingIrBuilder::notExpr(Val* val) {
  if (auto bool_val = dynamic_cast<Bool*>(val)) {
    if (bool_val->isConst()) {
      if (bool_val->value().value()) {
        return falseVal();
      } else {
        return trueVal();
      }
    }
  }
  return IrBuilder::notExpr(val);
}

Val* SimplifyingIrBuilder::addExpr(Int* lhs, Int::ScalarType rhs) {
  if (rhs == 0) {
    return lhs;
  } else if (lhs == nullptr) {
    return IrBuilder::create<kir::Int>(rhs);
  } else if (lhs->isConst()) {
    return IrBuilder::create<kir::Int>(lhs->value().value() + rhs);
  } else if (rhs > 0) {
    return IrBuilder::addExpr(lhs, IrBuilder::create<kir::Int>(rhs));
  } else {
    return IrBuilder::subExpr(lhs, IrBuilder::create<kir::Int>(-rhs));
  }
}

Val* SimplifyingIrBuilder::addExpr(Int* lhs, Int* rhs) {
  if (rhs == nullptr) {
    return lhs;
  } else if (lhs == nullptr) {
    return rhs;
  } else if (lhs->isConst()) {
    return addExpr(rhs, lhs->value().value());
  } else if (rhs->isConst()) {
    return addExpr(lhs, rhs->value().value());
  } else {
    return IrBuilder::addExpr(lhs, rhs);
  }
}

Val* SimplifyingIrBuilder::addExpr(Val* lhs, Val* rhs) {
  TORCH_INTERNAL_ASSERT(lhs != nullptr || rhs != nullptr);
  if (lhs == nullptr || lhs->isZeroInt()) {
    return rhs;
  } else if (rhs == nullptr || rhs->isZeroInt()) {
    return lhs;
  }
  auto lhs_int = dynamic_cast<Int*>(lhs);
  auto rhs_int = dynamic_cast<Int*>(rhs);
  if (lhs_int != nullptr && rhs_int != nullptr) {
    return addExpr(lhs_int, rhs_int);
  } else {
    return IrBuilder::addExpr(lhs, rhs);
  }
}

Val* SimplifyingIrBuilder::subExpr(Val* lhs, Val* rhs) {
  return addExpr(lhs, negExpr(rhs));
}

Val* SimplifyingIrBuilder::andExpr(Val* lhs, Val* rhs) {
  TORCH_INTERNAL_ASSERT(!(lhs == nullptr && rhs == nullptr));

  if (lhs == nullptr) {
    return rhs;
  } else if (rhs == nullptr) {
    return lhs;
  }

  bool lhs_definitely_true = false;
  bool lhs_definitely_false = false;
  auto lhs_bool = dynamic_cast<Bool*>(lhs);
  if (lhs_bool && lhs_bool->isConst()) {
    lhs_definitely_true = lhs_bool->value().value();
    lhs_definitely_false = !lhs_bool->value().value();
  }
  auto rhs_bool = dynamic_cast<Bool*>(rhs);
  bool rhs_definitely_true = false;
  bool rhs_definitely_false = false;
  if (rhs_bool && rhs_bool->isConst()) {
    rhs_definitely_true = rhs_bool->value().value();
    rhs_definitely_false = !rhs_bool->value().value();
  }

  if (lhs_definitely_true && rhs_definitely_true) {
    return trueVal();
  } else if (lhs_definitely_false || rhs_definitely_false) {
    return falseVal();
  } else if (lhs_definitely_true) {
    return rhs;
  } else if (rhs_definitely_true) {
    return lhs;
  }

  return IrBuilder::andExpr(lhs, rhs);
}

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
