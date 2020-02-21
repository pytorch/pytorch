#include <c10/util/Exception.h>
#include <torch/csrc/jit/fuser/common/arith.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
#include <torch/csrc/jit/fuser/common/type.h>

namespace torch{
namespace jit{
namespace fuser{
TORCH_API Val* new_val(ValType vtype, DataType dtype){
  switch (vtype) {
    case (ValType::Tensor):
      return new Tensor(dtype); // TODO add dtype here.
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

TORCH_API Val* promote_new(const Val* v1, const Val* v2) {
  TORCH_CHECK(v1->isVal() && v2->isVal());
  TORCH_CHECK(
      v1->getDataType() != DataType::Null &&
      v2->getDataType() != DataType::Null);

  ValType out_vtype =
      promote_type(v1->getValType().value(), v2->getValType().value());
  DataType out_dtype =
      promote_type(v1->getDataType().value(), v2->getDataType().value());

  return new_val(out_vtype, out_dtype);
}

TORCH_API Val* cast_op(const DataType dtype, const Val* v1){
  if( !is_cast_legal(v1->getDataType().value(), dtype) ) {
	std::stringstream err;
	err << "Illegal Cast of DataTypes From: " << v1->getDataType().value() << " To: " << dtype;
    throw std::runtime_error(err.str());
  }
  Val* out = new_val(v1->getValType().value(), dtype);
  Statement* expr = new UnaryOp(UnaryOpType::Cast, out, v1);
  return out;
}

TORCH_API Val* unary_op(UnaryOpType type, const Val* v1){
  Val* out = new_val(v1->getValType().value(), v1->getDataType().value());
  Statement* expr = new UnaryOp(type, out, v1);
  return out;
}

TORCH_API Val* binary_op(BinaryOpType type, const Val* v1, const Val* v2){
  Val* out = promote_new(v1, v2);
  Statement* expr = new BinaryOp(type, out, v1, v2);
  return out;
}

TORCH_API Val* add(const Val* v1, const Val* v2){
  return binary_op(BinaryOpType::Add, v1, v2);
}

TORCH_API Val* sub(const Val* v1, const Val* v2){
  return binary_op(BinaryOpType::Sub, v1, v2);
}

TORCH_API Val* mul(const Val* v1, const Val* v2){
  return binary_op(BinaryOpType::Mul, v1, v2);
}

TORCH_API Val* div(const Val* v1, const Val* v2){
  return binary_op(BinaryOpType::Div, v1, v2);
}

TORCH_API Val* mod(const Val* v1, const Val* v2){
  return binary_op(BinaryOpType::Mod, v1, v2);
}

} // namespace fuser
} // namespace jit
} // namespace torch
