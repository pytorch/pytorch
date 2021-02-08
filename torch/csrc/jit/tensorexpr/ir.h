#pragma once

#include <string>
#include <vector>

#include <c10/util/string_utils.h>
#include <torch/csrc/jit/tensorexpr/exceptions.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>

#include <ATen/core/ivalue.h>

namespace torch {
namespace jit {
namespace tensorexpr {

enum CompareSelectOperation {
  kEQ = 0,
  kGT,
  kGE,
  kLT,
  kLE,
  kNE,
};

inline int getPrecedence(IRNodeType ty) {
  // Match C++ operator precedence rules, since some pretty-print expressions to
  // C++. SEE: https://en.cppreference.com/w/cpp/language/operator_precedence
  switch (ty) {
    case kPrimitive:
      return 0;
    case kCast:
    case kBitCast:
      return 2;
    case kAdd:
    case kSub:
      return 6;
    case kMul:
    case kDiv:
    case kMod:
      return 5;
    case kMax:
    case kMin:
      return 99;
    case kAnd:
      return 11;
    case kOr:
      return 13;
    case kLshift:
    case kRshift:
      return 7;
    case kXor:
      return 12;
    case kCompareSelect:
      return 16;
    default:
      return 99;
  }
}

class Placeholder;

class Cast : public ExprNode<Cast> {
 public:
  const Expr* src_value() const {
    return src_value_;
  }
  static ExprHandle make(Dtype dtype, const ExprHandle& src_value) {
    return ExprHandle(new Cast(dtype, src_value.node()));
  }
  Cast(Dtype dtype, const Expr* src_value)
      : ExprNodeBase(dtype, kCast), src_value_(src_value) {}

  bool isConstant() const override {
    return src_value_->isConstant();
  }

 private:
  const Expr* src_value_;
};

template <typename T>
ExprHandle cast(const ExprHandle& src_value) {
  return Cast::make(Dtype(ToDtype<T>(), src_value.dtype().lanes()), src_value);
}

// This is a bitwise cast, akin to bitcast in LLVM
class BitCast : public ExprNode<BitCast> {
 public:
  const Expr* src_value() const {
    return src_value_;
  }
  static ExprHandle make(Dtype dtype, const ExprHandle& src_value) {
    return ExprHandle(new BitCast(dtype, src_value.node()));
  }
  BitCast(Dtype dtype, const Expr* src_value)
      : ExprNodeBase(dtype, kBitCast), src_value_(src_value) {
    TORCH_CHECK(src_value_->dtype().byte_size() == dtype.byte_size());
  }

  bool isConstant() const override {
    return src_value_->isConstant();
  }

 private:
  const Expr* src_value_;
};

template <typename T>
ExprHandle bitcast(const ExprHandle& src_value) {
  return BitCast::make(
      Dtype(ToDtype<T>(), src_value.dtype().lanes()), src_value);
}

// Represent the expression node for binary operators.
// A CRTP pattern to share common code among the operators.
template <typename Op>
class BinaryOpNode : public ExprNode<Op> {
 public:
  const Expr* lhs() const {
    return this->lhs_;
  }
  const Expr* rhs() const {
    return this->rhs_;
  }

  static ExprHandle make(const ExprHandle& lhs, const ExprHandle& rhs) {
    return ExprHandle(new Op(lhs.node(), rhs.node()));
  }

  BinaryOpNode(
      const Expr* lhs_v,
      const Expr* rhs_v,
      IRNodeType expr_type,
      ScalarType ret_type = ScalarType::None)
      : ExprNode<Op>(
            BinaryOpDtype(lhs_v->dtype(), rhs_v->dtype(), ret_type),
            expr_type),
        lhs_(CastIfNeeded(lhs_v, ExprNode<Op>::dtype())),
        rhs_(CastIfNeeded(rhs_v, ExprNode<Op>::dtype())) {}

 private:
  static const Expr* CastIfNeeded(const Expr* expr, Dtype dst_dtype) {
    if (expr->dtype() == dst_dtype) {
      return expr;
    }
    return Cast::make(dst_dtype, ExprHandle(expr)).node();
  }

  const Expr* lhs_;
  const Expr* rhs_;
};

class Add : public BinaryOpNode<Add> {
 public:
  Add(const Expr* lhs, const Expr* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kAdd) {}
};

class Sub : public BinaryOpNode<Sub> {
 public:
  Sub(const Expr* lhs, const Expr* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kSub) {}
};

class Mul : public BinaryOpNode<Mul> {
 public:
  Mul(const Expr* lhs, const Expr* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kMul) {}
};

class Div : public BinaryOpNode<Div> {
 public:
  Div(const Expr* lhs, const Expr* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kDiv) {}
};

class Mod : public BinaryOpNode<Mod> {
 public:
  Mod(const Expr* lhs, const Expr* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kMod) {}
};

template <typename Op>
class BitwiseOpNode : public BinaryOpNode<Op> {
 public:
  BitwiseOpNode(const Expr* lhs, const Expr* rhs, IRNodeType type)
      : BinaryOpNode<Op>(lhs, rhs, type) {}

  static ExprHandle make(const ExprHandle& lhs, const ExprHandle& rhs) {
    if (!lhs.dtype().is_integral()) {
      throw unsupported_dtype();
    }
    if (lhs.dtype() != rhs.dtype()) {
      throw malformed_input("lhs/rhs dtype mismatch");
    }
    return BinaryOpNode<Op>::make(lhs, rhs);
  }
};

class And : public BitwiseOpNode<And> {
 public:
  And(const Expr* lhs, const Expr* rhs)
      : BitwiseOpNode(lhs, rhs, IRNodeType::kAnd) {}
};

class Or : public BitwiseOpNode<Or> {
 public:
  Or(const Expr* lhs, const Expr* rhs)
      : BitwiseOpNode(lhs, rhs, IRNodeType::kOr) {}
};

class Xor : public BitwiseOpNode<Xor> {
 public:
  Xor(const Expr* lhs, const Expr* rhs)
      : BitwiseOpNode(lhs, rhs, IRNodeType::kXor) {}
};

class Lshift : public BitwiseOpNode<Lshift> {
 public:
  Lshift(const Expr* lhs, const Expr* rhs)
      : BitwiseOpNode(lhs, rhs, IRNodeType::kLshift) {}
};

class Rshift : public BitwiseOpNode<Rshift> {
 public:
  Rshift(const Expr* lhs, const Expr* rhs)
      : BitwiseOpNode(lhs, rhs, IRNodeType::kRshift) {}
};

class Max : public BinaryOpNode<Max> {
 private:
  bool propagate_nans_;

 public:
  Max(const Expr* lhs, const Expr* rhs, bool propagate_nans)
      : BinaryOpNode(lhs, rhs, IRNodeType::kMax),
        propagate_nans_(propagate_nans) {}

  bool propagate_nans() const {
    return propagate_nans_;
  }

  static ExprHandle make(const ExprHandle& lhs, const ExprHandle& rhs) = delete;
  static ExprHandle make(
      const ExprHandle& lhs,
      const ExprHandle& rhs,
      bool propagate_nans) {
    return ExprHandle(new Max(lhs.node(), rhs.node(), propagate_nans));
  }
};

class Min : public BinaryOpNode<Min> {
 private:
  bool propagate_nans_;

 public:
  Min(const Expr* lhs, const Expr* rhs, bool propagate_nans)
      : BinaryOpNode(lhs, rhs, IRNodeType::kMin),
        propagate_nans_(propagate_nans) {}

  bool propagate_nans() const {
    return propagate_nans_;
  }

  static ExprHandle make(const ExprHandle& lhs, const ExprHandle& rhs) = delete;
  static ExprHandle make(
      const ExprHandle& lhs,
      const ExprHandle& rhs,
      bool propagate_nans) {
    return ExprHandle(new Min(lhs.node(), rhs.node(), propagate_nans));
  }
};

// Encode typed immediate values e.g. IntImm, FloatImm.
#define IMM_DECLARE(Type, Name)                               \
  class Name##Imm : public ExprNode<Name##Imm> {              \
   public:                                                    \
    Name##Imm(Type value)                                     \
        : ExprNodeBase(k##Name, kPrimitive), value_(value) {} \
    bool isConstant() const override {                        \
      return true;                                            \
    }                                                         \
    Type value() const {                                      \
      return value_;                                          \
    }                                                         \
    static ExprHandle make(Type value) {                      \
      return ExprHandle(new Name##Imm(value));                \
    }                                                         \
                                                              \
   private:                                                   \
    Type value_;                                              \
  };
AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_DECLARE);
#undef IMM_DECLARE

// Get immediate by ScalarType.
template <typename T>
Expr* getImmediateByType(ScalarType immType, T initialVal) {
  switch (immType) {
#define TYPE_CASE(Type, Name) \
  case ScalarType::Name:      \
    return new Name##Imm(initialVal);
    AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
    default:
      throw unsupported_dtype();
  }
  return nullptr;
}

template <typename T>
Expr* getImmediateByType(Dtype dtype, T initialVal) {
  return getImmediateByType<T>(dtype.scalar_type(), initialVal);
}

template <typename T>
T immediateAs(const Expr* e) {
#define TYPE_CASE(Type, Name)                                     \
  if (const Name##Imm* imm = dynamic_cast<const Name##Imm*>(e)) { \
    return imm->value();                                          \
  }
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
  throw unsupported_dtype();
  return 0;
}

template <typename T>
bool immediateEquals(const Expr* e, T val) {
#define TYPE_CASE(Type, Name)                                     \
  if (const Name##Imm* imm = dynamic_cast<const Name##Imm*>(e)) { \
    return imm->value() == val;                                   \
  }
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
  throw unsupported_dtype();
  return false;
}

template <typename T>
bool immediateIsNegative(const T* e) {
#define TYPE_CASE(Type, Name)                                     \
  if (const Name##Imm* imm = dynamic_cast<const Name##Imm*>(e)) { \
    return imm->value() < 0;                                      \
  }
  AT_FORALL_SCALAR_TYPES_AND(Half, TYPE_CASE);
#undef TYPE_CASE
  return false;
}

// Represents a ramp vector node:
//     [base, base + 1 * stride, ... , base + (lanes - 1) * stride]
class Ramp : public ExprNode<Ramp> {
 public:
  const Expr* base() const {
    return base_;
  }
  const Expr* stride() const {
    return stride_;
  }
  static ExprHandle make(
      const ExprHandle& base,
      const ExprHandle& stride,
      int lanes) {
    return ExprHandle(new Ramp(base.node(), stride.node(), lanes));
  }
  int lanes() const {
    return lanes_;
  }

  Ramp(const Expr* base, const Expr* stride, int lanes)
      : ExprNodeBase(Dtype(base->dtype(), lanes), kRamp),
        base_(base),
        stride_(stride),
        lanes_(lanes) {
    if (stride->dtype() != base->dtype()) {
      throw malformed_input("Bad stride in Ramp");
    }
  }

 private:
  const Expr* base_;
  const Expr* stride_;
  int lanes_;
};

class TORCH_API Load : public ExprNode<Load> {
 public:
  const Var* base_handle() const {
    return buf_->base_handle();
  }
  std::vector<const Expr*> indices() const {
    return indices_;
  }
  const Expr* flat_index() const {
    TORCH_CHECK(indices_.size() == 1, "Indices haven't been flattened.");
    return indices_[0];
  }
  const Expr* mask() const {
    return mask_;
  }
  const Buf* buf() const {
    return buf_;
  }
  static ExprHandle make(
      Dtype dtype,
      const BufHandle& buf,
      const std::vector<ExprHandle>& indices,
      const ExprHandle& mask);
  static ExprHandle make(
      const BufHandle& buf,
      const std::vector<ExprHandle>& indices,
      const ExprHandle& mask);

  Load(
      Dtype dtype,
      const Buf* base_handle,
      const std::vector<const Expr*>& indices,
      const Expr* mask);
  Load(
      const Buf* base_handle,
      const std::vector<const Expr*>& indices,
      const Expr* mask);

 private:
  void verify_dtypes() const;

  const Buf* buf_;
  std::vector<const Expr*> indices_;
  const Expr* mask_;
};

class Broadcast : public ExprNode<Broadcast> {
 public:
  const Expr* value() const {
    return value_;
  }
  int lanes() const {
    return lanes_;
  }
  static ExprHandle make(const ExprHandle& value, int lanes) {
    return ExprHandle(new Broadcast(value.node(), lanes));
  }
  Broadcast(const Expr* value, int lanes)
      : ExprNodeBase(Dtype(value->dtype(), lanes), kBroadcast),
        value_(value),
        lanes_(lanes) {}

 private:
  const Expr* value_;
  int lanes_;
};

class IfThenElse : public ExprNode<IfThenElse> {
 public:
  const Expr* condition() const {
    return condition_;
  }

  // Lazily evaluated only if condition is true
  const Expr* true_value() const {
    return true_;
  }

  // Lazily evaluated only if condition is false
  const Expr* false_value() const {
    return false_;
  }

  static ExprHandle make(
      const ExprHandle& c,
      const ExprHandle& t,
      const ExprHandle& f) {
    return ExprHandle(new IfThenElse(c.node(), t.node(), f.node()));
  }

  IfThenElse(const Expr* c, const Expr* t, const Expr* f)
      : ExprNodeBase(t->dtype()), condition_(c), true_(t), false_(f) {
    if (!c->dtype().is_integral()) {
      throw unsupported_dtype();
    }
    if (c->dtype().lanes() != 1) {
      throw unsupported_dtype();
    }
    if (t->dtype() != f->dtype()) {
      throw malformed_input("Bad dtype in IfThenElse");
    }
  }

 private:
  const Expr* condition_;
  const Expr* true_;
  const Expr* false_;
};

class BaseCallNode : public Expr {
 public:
  enum CallType {
    kIntrinsics,
    kFunctionCall,
  };

  int nparams() const {
    return params_.size();
  }

  const Expr* param(int index) const {
    return params_[index];
  }
  const std::vector<const Expr*>& params() const {
    return params_;
  }

  virtual std::string func_name() const = 0;

  CallType call_type() const {
    return call_type_;
  }

 protected:
  BaseCallNode(
      Dtype dtype,
      CallType call_type,
      const std::vector<const Expr*>& params)
      : Expr(dtype), call_type_(call_type), params_(params) {}

 private:
  // The handler for the default ir_mutator to make a copy of this node with new
  // params.
  virtual const Expr* DefaultMutator(
      const std::vector<const Expr*>& new_params) const = 0;

  template <class U, class B>
  friend class ExprNode;
  friend class IRMutator;

  CallType call_type_;
  std::vector<const Expr*> params_;
};

template <typename Op>
class CallNode : public ExprNode<Op, BaseCallNode> {
 public:
  using BaseClass = ExprNode<Op, BaseCallNode>;
  using BaseClass::BaseClass;
};

class TORCH_API CompareSelect : public ExprNode<CompareSelect> {
 public:
  CompareSelectOperation compare_select_op() const {
    return compare_op_;
  }
  const Expr* lhs() const {
    return this->lhs_;
  }
  const Expr* rhs() const {
    return this->rhs_;
  }
  const Expr* ret_val1() const {
    return this->ret_val1_;
  }
  const Expr* ret_val2() const {
    return this->ret_val2_;
  }

  static ExprHandle make(
      const ExprHandle& lhs,
      const ExprHandle& rhs,
      CompareSelectOperation cmp_op) {
    if (lhs.dtype() != rhs.dtype()) {
      throw malformed_input("bad dtype in CompareSelect");
    }
    return ExprHandle(new CompareSelect(
        lhs.node(),
        rhs.node(),
        IntImm::make(1).node(),
        IntImm::make(0).node(),
        cmp_op));
  }

  static ExprHandle make(
      const ExprHandle& lhs,
      const ExprHandle& rhs,
      const ExprHandle& ret_val1,
      const ExprHandle& ret_val2,
      CompareSelectOperation cmp_op) {
    if (lhs.dtype() != rhs.dtype() || ret_val1.dtype() != ret_val2.dtype()) {
      throw malformed_input("bad dtype in CompareSelect");
    }
    return ExprHandle(new CompareSelect(
        lhs.node(), rhs.node(), ret_val1.node(), ret_val2.node(), cmp_op));
  }

  CompareSelect(
      const Expr* lhs,
      const Expr* rhs,
      const Expr* ret_val1,
      const Expr* ret_val2,
      CompareSelectOperation cmp_op)
      : ExprNodeBase(ret_val1->dtype()),
        lhs_(lhs),
        rhs_(rhs),
        ret_val1_(ret_val1),
        ret_val2_(ret_val2),
        compare_op_(cmp_op) {
    if (ret_val1->dtype() != ret_val2->dtype()) {
      throw malformed_input("bad dtype in CompareSelect");
    }
  }

  CompareSelect(const Expr* lhs, const Expr* rhs, CompareSelectOperation cmp_op)
      : ExprNodeBase(kInt),
        lhs_(lhs),
        rhs_(rhs),
        ret_val1_(new IntImm(1)),
        ret_val2_(new IntImm(0)),
        compare_op_(cmp_op) {}

 private:
  const Expr* lhs_;
  const Expr* rhs_;
  const Expr* ret_val1_;
  const Expr* ret_val2_;
  CompareSelectOperation compare_op_;
};

enum IntrinsicsOp {
  kSin,
  kCos,
  kTan,
  kAsin,
  kAcos,
  kAtan,
  kAtan2,
  kSinh,
  kCosh,
  kTanh,
  kSigmoid,
  kExp,
  kExpm1,
  kAbs,
  kLog,
  kLog2,
  kLog10,
  kLog1p,
  kErf,
  kErfc,
  kSqrt,
  kRsqrt,
  kPow,
  kCeil,
  kFloor,
  kRound,
  kTrunc,
  kFmod,
  kRemainder,
  kLgamma,
  kFrac,
  kIsNan,
  kRand, // We need more discussions on this. Should we consider stateful?
};

class Intrinsics : public CallNode<Intrinsics> {
 public:
  static ExprHandle make(IntrinsicsOp op_type, const ExprHandle& v1) {
    return ExprHandle(new Intrinsics(op_type, v1.node()));
  }

  static ExprHandle make(
      IntrinsicsOp op_type,
      const ExprHandle& v1,
      const ExprHandle& v2) {
    return ExprHandle(new Intrinsics(op_type, v1.node(), v2.node()));
  }

  static ExprHandle make(
      IntrinsicsOp op_type,
      const std::vector<ExprHandle>& params) {
    std::vector<const Expr*> params_nodes(params.size());
    for (size_t i = 0; i < params.size(); i++) {
      params_nodes[i] = params[i].node();
    }
    return ExprHandle(new Intrinsics(op_type, params_nodes));
  }

  static ExprHandle make(IntrinsicsOp op_type, Dtype dtype) {
    return ExprHandle(new Intrinsics(op_type, dtype));
  }

  IntrinsicsOp op_type() const {
    return op_type_;
  }

  std::string func_name() const override {
    switch (op_type()) {
      case kSin:
        return "sin";
      case kCos:
        return "cos";
      case kTan:
        return "tan";
      case kAsin:
        return "asin";
      case kAcos:
        return "acos";
      case kAtan:
        return "atan";
      case kAtan2:
        return "atan2";
      case kSinh:
        return "sinh";
      case kCosh:
        return "cosh";
      case kTanh:
        return "tanh";
      case kSigmoid:
        return "sigmoid";
      case kExp:
        return "exp";
      case kAbs:
        return "abs";
      case kLog:
        return "log";
      case kLog2:
        return "log2";
      case kLog10:
        return "log10";
      case kLog1p:
        return "log1p";
      case kErf:
        return "erf";
      case kSqrt:
        return "sqrt";
      case kRsqrt:
        return "rsqrt";
      case kPow:
        return "pow";
      case kCeil:
        return "ceil";
      case kFloor:
        return "floor";
      case kRound:
        return "round";
      case kTrunc:
        return "trunc";
      case kRand:
        return "rand";
      case kFmod:
        return "fmod";
      case kRemainder:
        return "remainder";
      case kLgamma:
        return "lgamma";
      case kExpm1:
        return "expm1";
      case kErfc:
        return "erfc";
      case kFrac:
        return "frac";
      case kIsNan:
        return "isnan";
      default:
        throw std::runtime_error(
            "invalid op_type: " + c10::to_string(op_type()));
    }
  }
  using BaseClass = CallNode<Intrinsics>;

  Intrinsics(IntrinsicsOp op_type, Dtype dtype)
      : BaseClass(IntrinsicsDtype(op_type, dtype), kIntrinsics, {}),
        op_type_(op_type) {
    if (OpArgCount(op_type) != 0) {
      throw malformed_input("bad arg count in Intrinsics");
    }
  }

  Intrinsics(IntrinsicsOp op_type, const Expr* v1)
      : BaseClass(IntrinsicsDtype(op_type, v1->dtype()), kIntrinsics, {v1}),
        op_type_(op_type) {
    if (OpArgCount(op_type) != 1) {
      throw malformed_input("bad arg count in Intrinsics");
    }
  }

  Intrinsics(IntrinsicsOp op_type, const Expr* v1, const Expr* v2)
      : BaseClass(
            IntrinsicsDtype(op_type, v1->dtype(), v2->dtype()),
            kIntrinsics,
            {v1, v2}),
        op_type_(op_type) {
    if (OpArgCount(op_type) != 2) {
      throw malformed_input("bad arg count in Intrinsics");
    }
  }

  Intrinsics(IntrinsicsOp op_type, const std::vector<const Expr*>& params)
      : BaseClass(IntrinsicsDtype(op_type, params), kIntrinsics, params),
        op_type_(op_type) {
    if (OpArgCount(op_type) != nparams()) {
      throw malformed_input("bad arg count in Intrinsics");
    }
  }

  bool isPure() const {
    return op_type_ != kRand;
  }

 private:
  TORCH_API static int OpArgCount(IntrinsicsOp op_type);

  const Expr* DefaultMutator(
      const std::vector<const Expr*>& new_params) const override {
    return new Intrinsics(this->op_type(), new_params);
  }

  TORCH_API static Dtype IntrinsicsDtype(IntrinsicsOp op_type, Dtype dt1);
  TORCH_API static Dtype IntrinsicsDtype(
      IntrinsicsOp op_type,
      Dtype dt1,
      Dtype dt2);
  TORCH_API static Dtype IntrinsicsDtype(
      IntrinsicsOp op_type,
      const std::vector<const Expr*>& params);

  IntrinsicsOp op_type_;
};

class Polynomial;
class Term;
class MaxTerm;
class MinTerm;

class FunctionCall;

TORCH_API std::vector<const Expr*> ExprHandleVectorToExprVector(
    const std::vector<ExprHandle>&);
TORCH_API std::vector<ExprHandle> ExprVectorToExprHandleVector(
    const std::vector<const Expr*>&);
TORCH_API std::vector<const Var*> VarHandleVectorToVarVector(
    const std::vector<VarHandle>&);
TORCH_API std::vector<VarHandle> VarVectorToVarHandleVector(
    const std::vector<const Var*>&);
TORCH_API const Expr* flatten_index(
    const std::vector<const Expr*>& dims,
    const std::vector<const Expr*>& indices);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
