#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/codegen/cuda/ir_internal_nodes.h>

namespace torch {
namespace jit {
namespace fuser {

// Will return a new value of type val with the DataType dtype, if it's a
// tensorview it will propagate the shape information from val.
TORCH_CUDA_API Val* newValLike(const Val* const val, DataType dtype) {
  switch (val->getValType().value()) {
    case (ValType::TensorView):
      return static_cast<const TensorView* const>(val)->newForOutput(dtype);
    case (ValType::NamedScalar):
    case (ValType::Scalar):
      switch (dtype) {
        case (DataType::Float):
          return new Float();
        case (DataType::Half):
          return new Half();
        case (DataType::Int):
          return new Int();
        default:
          break;
      }
    default:
      break;
  }

  TORCH_CHECK(
      false,
      "Could not generate a new value of type ",
      val->getValType().value(),
      " with data type ",
      val->getDataType().value());
}

TORCH_CUDA_API Val* newValLike(const Val* const val) {
  return newValLike(val, val->getDataType().value());
}

TORCH_CUDA_API Val* promoteNew(Val* v1, Val* v2) {
  // Can't promote two types if they aren't both
  // values with valid data types.
  TORCH_CHECK(v1->isVal() && v2->isVal());
  TORCH_CHECK(
      v1->getDataType() != DataType::Null &&
      v2->getDataType() != DataType::Null);

  ValType out_vtype =
      promote_type(v1->getValType().value(), v2->getValType().value());
  DataType out_dtype =
      promote_type(v1->getDataType().value(), v2->getDataType().value());

  if (out_vtype == v2->getValType().value())
    return newValLike(v2, out_dtype);

  return newValLike(v1, out_dtype);
}

TORCH_CUDA_API Val* castOp(DataType dtype, Val* v1) {
  if (v1->getDataType().value() == dtype)
    return v1;

  if (!is_cast_legal(v1->getDataType().value(), dtype)) {
    TORCH_CHECK(
        false,
        "Illegal Cast value from  DataType: ",
        v1->getDataType().value(),
        " to DataType: ",
        dtype);
  }

  Val* out = newValLike(v1, dtype);
  Statement* expr = new UnaryOp(UnaryOpType::Cast, out, v1);
  return out;
}

TORCH_CUDA_API Val* unaryOp(UnaryOpType type, Val* v1) {
  Val* out = newValLike(v1);
  Statement* expr = new UnaryOp(type, out, v1);
  return out;
}

TORCH_CUDA_API Val* binaryOp(BinaryOpType type, Val* v1, Val* v2) {
  Val* out = promoteNew(v1, v2);
  if (type >= BinaryOpType::Mod) {
    if (out->getDataType().value() != DataType::Int)
      out = newValLike(out, DataType::Int);
  }
  Statement* expr = new BinaryOp(type, out, v1, v2);
  return out;
}

TORCH_CUDA_API Val* add(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::Add, v1, v2);
}

TORCH_CUDA_API Val* sub(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::Sub, v1, v2);
}

TORCH_CUDA_API Val* mul(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::Mul, v1, v2);
}

TORCH_CUDA_API Val* div(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::Div, v1, v2);
}

TORCH_CUDA_API Val* mod(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::Mod, v1, v2);
}

TORCH_CUDA_API Val* lt(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::LT, v1, v2);
}

TORCH_CUDA_API Val* ceilDiv(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::CeilDiv, v1, v2);
}

TORCH_CUDA_API Val* andOp(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::And, v1, v2);
}

TORCH_CUDA_API Val* add_alpha(Val* v1, Val* v2, Val* s) {
  TORCH_CHECK(
      s->getValType().value() == ValType::Scalar,
      "Alpha value should be a Scalar Valtype and not ",
      s->getValType().value());

  Val* intrm = binaryOp(BinaryOpType::Mul, v2, s);
  return binaryOp(BinaryOpType::Add, v1, intrm);
}

TORCH_CUDA_API Val* sub_alpha(Val* v1, Val* v2, Val* s) {
  TORCH_CHECK(
      s->getValType().value() == ValType::Scalar,
      "Alpha value should be a Scalar Valtype and not ",
      s->getValType().value());

  Val* intrm = binaryOp(BinaryOpType::Mul, v2, s);
  return binaryOp(BinaryOpType::Sub, v1, intrm);
}

TORCH_CUDA_API Val* lerp(Val* start, Val* end, Val* weight) {
  Val* intrm1 = binaryOp(BinaryOpType::Sub, end, start);
  Val* intrm2 = binaryOp(BinaryOpType::Mul, weight, intrm1);
  return binaryOp(BinaryOpType::Add, start, intrm2);
}

TORCH_CUDA_API Val* addcmul(Val* v1, Val* v2, Val* v3, Val* s) {
  TORCH_CHECK(
      s->getValType().value() == ValType::Scalar,
      "Alpha value should be a Scalar Valtype and not ",
      s->getValType().value());

  Val* intrm1 = binaryOp(BinaryOpType::Mul, v3, s);
  Val* intrm2 = binaryOp(BinaryOpType::Mul, v2, intrm1);
  return binaryOp(BinaryOpType::Add, v1, intrm2);
}

TORCH_CUDA_API Val* where(Val* c, Val* v1, Val* v2) {
  TORCH_CHECK(
      c->getDataType().value() == DataType::Int,
      "Condition should be of DataType Int, not ",
      c->getDataType().value());

  Val* out = promoteNew(v1, v2);
  Statement* expr = new TernaryOp(TernaryOpType::Where, out, c, v1, v2);
  return out;
}

TORCH_CUDA_API Val* threshold(Val* in, Val* thresh, Val* value) {
  TORCH_CHECK(
         in->getDataType().value() == thresh->getDataType().value()
      && in->getDataType().value() == value->getDataType().value(),
      "All input DataType values should match the input ",
      in->getDataType().value());
  TORCH_CHECK(
         thresh->getValType().value() == ValType::Scalar
      && value->getValType().value() == ValType::Scalar,
      "Thresh and Value values should be Scalars");

  Val* out = newValLike(in);
  Statement* expr = new TernaryOp(TernaryOpType::Threshold, out, in, thresh, value);
  return out;
}

TORCH_CUDA_API Val* clamp(Val* in, Val* min_val, Val* max_val) {
  TORCH_CHECK(
         in->getDataType().value() == min_val->getDataType().value()
      && in->getDataType().value() == max_val->getDataType().value(),
      "All input DataType values should match the input ",
      in->getDataType().value());
  TORCH_CHECK(
         min_val->getValType().value() == ValType::Scalar
      && max_val->getValType().value() == ValType::Scalar,
      "Min and Max values should be Scalars");

  Val* out = newValLike(in);
  Statement* expr = new TernaryOp(TernaryOpType::Clamp, out, in, min_val, max_val);
  return out;
}

} // namespace fuser
} // namespace jit
} // namespace torch
