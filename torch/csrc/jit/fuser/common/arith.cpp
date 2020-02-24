#include <c10/util/Exception.h>
#include <torch/csrc/jit/fuser/common/arith.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
#include <torch/csrc/jit/fuser/common/type.h>

namespace torch{
namespace jit{
namespace fuser{


TORCH_API Val* new_val_like(const Val* const val, DataType dtype){
  switch (val->getValType().value()) {
    case (ValType::Tensor):
      throw std::runtime_error("Tensors cannot be intermediate values in this IR, must use TensorViews.");
    case(ValType::TensorView):
      return static_cast<const TensorView* const>(val)->cloneForOutput(dtype);
    case (ValType::Scalar):
      switch (dtype) {
        case (DataType::Float):
          return new Float();
        case (DataType::Int):
          return new Int();
        default:
          break;
      }
    default:
      break;
  }
  throw std::runtime_error("Cannot promote types."); //Todo print val and data types in the error
}

TORCH_API Val* new_val_like(const Val* const val){
  return new_val_like(val, val->getDataType().value());
}

TORCH_API Val* promote_new(Val* v1, Val* v2) {
  TORCH_CHECK(v1->isVal() && v2->isVal());
  TORCH_CHECK(
      v1->getDataType() != DataType::Null &&
      v2->getDataType() != DataType::Null);

  ValType out_vtype =
      promote_type(v1->getValType().value(), v2->getValType().value());
  DataType out_dtype =
      promote_type(v1->getDataType().value(), v2->getDataType().value());

  return new_val_like(v1, out_dtype);

}

TORCH_API Val* cast_op(DataType dtype, Val* v1){
  if( !is_cast_legal(v1->getDataType().value(), dtype) ) {
	std::stringstream err;
	err << "Illegal Cast of DataTypes From: " << v1->getDataType().value() << " To: " << dtype;
    throw std::runtime_error(err.str());
  }
  Val* out = new_val_like(v1, dtype);
  Statement* expr = new UnaryOp(UnaryOpType::Cast, out, v1);
  return out;
}

TORCH_API Val* unary_op(UnaryOpType type, Val* v1){
  Val* out = new_val_like(v1);
  Statement* expr = new UnaryOp(type, out, v1);
  return out;
}

TORCH_API Val* binary_op(BinaryOpType type, Val* v1, Val* v2){
  Val* out = promote_new(v1, v2);
  Statement* expr = new BinaryOp(type, out, v1, v2);
  return out;
}

TORCH_API Val* add(Val* v1, Val* v2){
  return binary_op(BinaryOpType::Add, v1, v2);
}

TORCH_API Val* sub(Val* v1, Val* v2){
  return binary_op(BinaryOpType::Sub, v1, v2);
}

TORCH_API Val* mul(Val* v1, Val* v2){
  return binary_op(BinaryOpType::Mul, v1, v2);
}

TORCH_API Val* div(Val* v1, Val* v2){
  return binary_op(BinaryOpType::Div, v1, v2);
}

TORCH_API Val* mod(Val* v1, Val* v2){
  return binary_op(BinaryOpType::Mod, v1, v2);
}

TORCH_API Val* ceilDiv(Val* v1, Val* v2){
  TORCH_CHECK(v1->getDataType().value() == DataType::Int
      && v2->getDataType().value() == DataType::Int);
  return binary_op(BinaryOpType::CeilDiv, v1, v2);
}

} // namespace fuser
} // namespace jit
} // namespace torch
