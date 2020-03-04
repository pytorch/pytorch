#include "torch/csrc/jit/tensorexpr/ir.h"

#include "torch/csrc/jit/tensorexpr/buffer.h"

namespace torch {
namespace jit {
namespace tensorexpr {

static Dtype ChooseDtype(const Dtype& buffer_dtype, const Dtype& index_dtype) {
  return Dtype(buffer_dtype, index_dtype.lanes());
}

Load::Load(const Buffer& buffer, const Expr* index, const Expr* mask)
    : Load(
          ChooseDtype(buffer.dtype(), index->dtype()),
          buffer.data(),
          index,
          mask) {}

Load::Load(
    Dtype dtype,
    const Var* base_handle,
    const Expr* index,
    const Expr* mask)
    : ExprNodeBase(dtype),
      base_handle_(base_handle),
      index_(index),
      mask_(mask) {
  CHECK_EQ(base_handle_->dtype(), kHandle);
  CHECK_EQ(index->dtype().lanes(), mask->dtype().lanes());
  CHECK_EQ(index->dtype().scalar_type(), ScalarType::Int);
}

Store::Store(
    const Buffer& buffer,
    const Expr* index,
    const Expr* value,
    const Expr* mask)
    : Store(buffer.data(), index, value, mask) {
  CHECK_EQ(buffer.dtype().scalar_type(), value->dtype().scalar_type());
  CHECK_EQ(buffer.dtype().scalar_type(), value->dtype().scalar_type());
}

Dtype Intrinsics::IntrinsicsDtype(IntrinsicsOp op_type, Dtype dt1) {
  // TODO: check the op_type and make a real decision
  return dt1;
}

Dtype Intrinsics::IntrinsicsDtype(IntrinsicsOp op_type, Dtype dt1, Dtype dt2) {
  // TODO: check the op_type and make a real decision
  return dt1;
}

Dtype Intrinsics::IntrinsicsDtype(
    IntrinsicsOp op_type,
    const std::vector<const Expr*>& params) {
  // TODO: check the op_type an dmake a real decision
  CHECK_GE(params.size(), 1ULL);
  return params[0]->dtype();
}

int Intrinsics::OpArgCount(IntrinsicsOp op_type) {
  switch (op_type) {
    case kSin:
    case kCos:
    case kTan:
    case kAsin:
    case kAcos:
    case kAtan:
    case kSinh:
    case kCosh:
    case kTanh:
    case kExp:
    case kExpm1:
    case kFabs:
    case kLog:
    case kLog2:
    case kLog10:
    case kLog1p:
    case kErf:
    case kErfc:
    case kSqrt:
    case kRsqrt:
    case kCeil:
    case kFloor:
    case kRound:
    case kTrunc:
    case kFrac:
    case kLgamma:
      return 1;
    case kRand:
      return 0;
    case kAtan2:
    case kFmod:
    case kPow:
    case kRemainder:
      return 2;
    default:
      throw std::runtime_error("invalid op_type: " + std::to_string(op_type));
  }
}

std::vector<const Expr*> ExprHandleVectorToExprVector(
    const std::vector<ExprHandle>& v) {
  std::vector<const Expr*> result(v.size());
  for (size_t i = 0; i < v.size(); i++) {
    result[i] = v[i].node();
  }
  return result;
}

std::vector<ExprHandle> ExprVectorToExprHandleVector(
    const std::vector<const Expr*>& v) {
  std::vector<ExprHandle> result(v.size());
  for (size_t i = 0; i < v.size(); i++) {
    result[i] = ExprHandle(v[i]);
  }
  return result;
}

std::vector<const Var*> VarHandleVectorToVarVector(
    const std::vector<VarHandle>& v) {
  std::vector<const Var*> result(v.size());
  for (size_t i = 0; i < v.size(); i++) {
    result[i] = v[i].node();
  }
  return result;
}

std::vector<VarHandle> VarVectorToVarHandleVector(
    const std::vector<const Var*>& v) {
  std::vector<VarHandle> result(v.size());
  for (size_t i = 0; i < v.size(); i++) {
    result[i] = VarHandle(v[i]);
  }
  return result;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
