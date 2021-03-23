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
  return result;
}

Val* IrBuilder::newLogicExpr(BinaryOpType op_type, Val* lhs, Val* rhs) {
  auto result = create<Bool>(c10::nullopt);
  create<BinaryOp>(op_type, result, lhs, rhs);
  return result;
}

Val* IrBuilder::negExpr(Val* val) {
  auto result = newResult(val->dtype());
  create<UnaryOp>(UnaryOpType::Neg, result, val);
  return result;
}

Val* IrBuilder::andExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::And, lhs, rhs);
}

Val* IrBuilder::eqExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::Eq, lhs, rhs);
}

Val* IrBuilder::ltExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::LT, lhs, rhs);
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

Int* IrBuilder::zero() {
  if (zero_ == nullptr) {
    zero_ = create<kir::Int>(0);
  }
  return zero_;
}

Int* IrBuilder::one() {
  if (one_ == nullptr) {
    one_ = create<kir::Int>(1);
  }
  return one_;
}

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
