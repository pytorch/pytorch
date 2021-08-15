#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

bool isLoweredScalar(const Val* val) {
  switch (val->getValType().value()) {
    case ValType::KirNamedScalar:
    case ValType::KirScalar:
      return true;
    default:
      return false;
  }
}

bool isLoweredVal(const Val* val) {
  switch (val->getValType().value()) {
    case ValType::TensorIndex:
    case ValType::KirNamedScalar:
    case ValType::KirScalar:
    case ValType::KirTensorDomain:
    case ValType::KirIterDomain:
    case ValType::KirTensorView:
      return true;
    default:
      return false;
  }
}

Val* IrBuilder::newResult(const Val* lhs, const Val* rhs) {
  TORCH_CHECK(isLoweredScalar(lhs));
  TORCH_CHECK(isLoweredScalar(rhs));
  TORCH_CHECK(lhs->getDataType() == rhs->getDataType());

  // Allocate a compatible result value
  switch (lhs->getDataType().value()) {
    case DataType::Bool:
      return create<Bool>(c10::nullopt);
    case DataType::Float:
      return create<Float>(c10::nullopt);
    case DataType::Half:
      return create<Half>(c10::nullopt);
    case DataType::Int:
      return create<Int>(c10::nullopt);
    default:
      TORCH_CHECK(false, "Unexpected data type");
  }
}

Val* IrBuilder::newArithmeticExpr(BinaryOpType op_type, Val* lhs, Val* rhs) {
  auto result = newResult(lhs, rhs);
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

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
