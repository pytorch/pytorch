#include <torch/csrc/jit/tensorexpr/ir.h>

#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

static Dtype ChooseDtype(const Dtype& buffer_dtype, const Dtype& index_dtype) {
  return Dtype(buffer_dtype, index_dtype.lanes());
}

static Dtype dtypeOfIndices(const std::vector<const Expr*>& indices) {
  if (!indices.size()) {
    // Return something so we can handle scalar buffers.
    return kInt;
  }
  Dtype dt = indices.at(0)->dtype();
  for (size_t i = 1; i < indices.size(); ++i) {
    if (indices.at(i)->dtype() != dt) {
      throw malformed_input("dtype mismatch in dtypeOfIndices");
    }
  }
  return dt;
}

static bool indicesValid(const std::vector<const Expr*>& indices) {
  if (indices.size() == 0) {
    return true;
  }
  Dtype index_dtype = dtypeOfIndices(indices);
  if (indices.size() > 1 && index_dtype.lanes() > 1) {
    // Multilane is only allowed in a flattened (i.e. 1D) index
    return false;
  }
  if (index_dtype.scalar_type() != ScalarType::Int) {
    return false;
  }
  return true;
}

void Load::verify_dtypes() const {
  if (indices_.size() > 0 && buf_->base_handle()->dtype() != kHandle) {
    throw malformed_input(
        "Load base handle dtype must be Handle", buf_->base_handle());
  }

  if (!indicesValid(indices_)) {
    throw malformed_input("invalid indices in Load");
  }
  Dtype index_dtype = dtypeOfIndices(indices_);
  if (index_dtype.lanes() != mask_->dtype().lanes()) {
    throw malformed_input("lane mismatch in Load mask");
  }
}

Load::Load(
    Dtype dtype,
    const Buf* buf,
    const std::vector<const Expr*>& indices,
    const Expr* mask)
    : ExprNodeBase(dtype), buf_(buf), indices_(indices), mask_(mask) {
  verify_dtypes();
}

Load::Load(
    const Buf* buf,
    const std::vector<const Expr*>& indices,
    const Expr* mask)
    : Load(
          ChooseDtype(buf->dtype(), dtypeOfIndices(indices)),
          buf,
          indices,
          mask) {}

ExprHandle Load::make(
    Dtype dtype,
    const BufHandle& buf,
    const std::vector<ExprHandle>& indices,
    const ExprHandle& mask) {
  return ExprHandle(new Load(
      dtype, buf.node(), ExprHandleVectorToExprVector(indices), mask.node()));
}

ExprHandle Load::make(
    const BufHandle& buf,
    const std::vector<ExprHandle>& indices,
    const ExprHandle& mask) {
  return Load::make(buf.dtype(), buf, indices, mask);
}

Store::Store(
    const Buf* buf,
    std::vector<const Expr*> indices,
    const Expr* value,
    const Expr* mask)
    : buf_(buf), indices_(std::move(indices)), value_(value), mask_(mask) {
  if (indices_.size() > 0 && buf->base_handle()->dtype() != kHandle) {
    throw malformed_input("Store base handle must be Handle");
  }
  /*
  TODO: Reenable the checks.
  The reason they are disabled is that kernel.cpp is using Buffers somewhat
  loosely: we don't set dimensions properly and just construct index expressions
  directly. We should harden that part and then we'd be able to turn on these
  checks.

  if (!indicesValid(indices)) {
    throw malformed_input();
  }
  if (!mask || !value) {
    throw malformed_input();
  }
  Dtype index_dtype = dtypeOfIndices(indices);
  if (index_dtype.lanes() != mask->dtype().lanes()) {
    throw malformed_input();
  }
  if (index_dtype.lanes() != value->dtype().lanes()) {
    throw malformed_input();
  }
  */
}

Store* Store::make(
    const BufHandle& buf,
    const std::vector<ExprHandle>& indices,
    const ExprHandle& value,
    const ExprHandle& mask) {
  return new Store(
      buf.node(),
      ExprHandleVectorToExprVector(indices),
      value.node(),
      mask.node());
}

Store* Store::make(
    const BufHandle& buf,
    const std::vector<ExprHandle>& indices,
    const ExprHandle& value) {
  return new Store(
      buf.node(),
      ExprHandleVectorToExprVector(indices),
      value.node(),
      ExprHandle(1).node());
}

const Expr* flatten_index(
    const std::vector<const Expr*>& dims,
    const std::vector<const Expr*>& indices) {
  // Handle already flattened indices first
  if (indices.size() == 1) {
    return indices[0];
  }

  size_t ndim = dims.size();
  if (ndim != indices.size()) {
    throw malformed_input("dimensions mismatch in flatten_index");
  }
  if (ndim == 0) {
    return new IntImm(0);
  }
  std::vector<const Expr*> strides(ndim);
  // stride[i] = stride[i+1]*dims[i+1], i < ndim-1
  // stride[i] = 1,                     i = ndim-1
  strides[ndim - 1] = new IntImm(1);
  for (size_t i = 1; i < ndim; i++) {
    strides[ndim - 1 - i] = new Mul(strides[ndim - i], dims[ndim - i]);
  }

  const Expr* total_index = new IntImm(0);
  for (size_t i = 0; i < ndim; i++) {
    total_index = new Add(total_index, new Mul(indices[i], strides[i]));
  }
  return total_index;
}

Dtype Intrinsics::IntrinsicsDtype(IntrinsicsOp op_type, Dtype dt1) {
  if (op_type == kIsNan) {
    return dt1.cloneWithScalarType(ScalarType::Int);
  }
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
  // TODO: check the op_type and make a real decision
  // Doesnt this fail with kRand?
  if (params.size() == 0) {
    throw malformed_input("invalid params in Intrinsics");
  } else if (params.size() == 1) {
    return IntrinsicsDtype(op_type, params[0]->dtype());
  } else if (params.size() == 2) {
    return IntrinsicsDtype(op_type, params[0]->dtype(), params[1]->dtype());
  }
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
    case kSigmoid:
    case kExp:
    case kExpm1:
    case kAbs:
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
    case kIsNan:
      return 1;
    case kRand:
      return 0;
    case kAtan2:
    case kFmod:
    case kPow:
    case kRemainder:
      return 2;
    default:
      throw std::runtime_error("invalid op_type: " + c10::to_string(op_type));
  }
}

ExternalCall* ExternalCall::make(
    BufHandle buf,
    const std::string& func_name,
    const std::vector<BufHandle>& buf_args,
    const std::vector<ExprHandle>& args) {
  std::vector<const Buf*> buf_arg_nodes;
  buf_arg_nodes.reserve(buf_args.size());
  for (const BufHandle& buf_arg : buf_args) {
    buf_arg_nodes.push_back(buf_arg.node());
  }
  return new ExternalCall(
      buf.node(), func_name, buf_arg_nodes, ExprHandleVectorToExprVector(args));
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
