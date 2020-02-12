#include <c10/util/Exception.h>
#include <torch/csrc/jit/fuser/common/arith.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
#include <torch/csrc/jit/fuser/common/type.h>

<<<<<<< HEAD
namespace torch {
namespace jit {
namespace fuser {
// Return new value of type that v1 and v2 promotes to
TORCH_API Val* promote_new(Val* v1, Val* v2) {
  TORCH_CHECK(v1->isVal() && v2->isVal());
  TORCH_CHECK(
      v1->getDataType() != DataType::Null &&
      v2->getDataType() != DataType::Null);

  ValType out_vtype =
      promote_type(v1->getValType().value(), v2->getValType().value());
  DataType out_dtype =
      promote_type(v1->getDataType().value(), v2->getDataType().value());

  switch (out_vtype) {
    case (ValType::Tensor):
      return new Tensor(out_dtype); // TODO add dtype here.
    case (ValType::Scalar):
      switch (out_dtype) {
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
  std::runtime_error("Did not recognize out type.");
  return new Int(-1);
}

TORCH_API Val* add(Val* v1, Val* v2) {
  Val* out = promote_new(v1, v2);
  Statement* expr = new Add(out, v1, v2);
  return out;
=======
namespace torch{
namespace jit{
namespace fuser{
TORCH_API Val* new_val(ValType type){
    switch(type){
        case(ValType::Tensor):
            return new Tensor();
        case(ValType::Float):
            return new Float();
        case(ValType::Int):
            return new Int();
    }
    std::runtime_error("Did not recognize out type.");
    return new Int(-1);
}

TORCH_API Val* cast_op(const DataType dtype, Val* v1){
  if( !is_cast_legal(v1->getDataType().value(), dtype) ) {
	std::stringstream err;
	err << "Illegal Cast of DataTypes From: " << v1->getDataType().value() << " To: " << dtype;
    throw std::runtime_error(err.str());
  }
  Val* out = new_val(v1->getValType().value(), dtype);
  Statement* expr = new UnaryOp(UnaryOpType::Cast, out, v1);
  return out;
}

TORCH_API Val* unary_op(UnaryOpType type, Val* v1){
    Val* out = new_val(v1->getValType().value());
    Statement* expr = new UnaryOp(type, out, v1);
    return out;
}

TORCH_API Val* binary_op(BinaryOpType type, Val* v1, Val* v2){
    ValType out_type = promote_scalar(v1->getValType().value(), v2->getValType().value());
    Val* out = new_val(out_type);
    Statement* expr = new BinaryOp(type, out, v1, v2);
    return out;
>>>>>>> Create BinaryOp and UnaryOp Exprs.
}

} // namespace fuser
} // namespace jit
} // namespace torch
