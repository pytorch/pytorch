#pragma once

#include <string>
#include <utility>
#include <vector>

#include <c10/util/string_utils.h>
#include <torch/csrc/jit/tensorexpr/exceptions.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
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

enum CompareSelectBias {
  kUnbiased,
  kLikely,
  kUnlikely,
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

class TORCH_API Cast : public ExprNode<Cast> {
 public:
  ExprPtr src_value() const {
    return src_value_;
  }

  void set_src_value(ExprPtr src_value) {
    src_value_ = std::move(src_value);
  }

  static ExprHandle make(Dtype dtype, const ExprHandle& src_value) {
    return ExprHandle(alloc<Cast>(dtype, src_value.node()));
  }
  Cast(Dtype dtype, ExprPtr src_value)
      : ExprNodeBase(dtype, kCast), src_value_(std::move(src_value)) {}

  bool isConstant() const override {
    return src_value_->isConstant();
  }

 private:
  ExprPtr src_value_;
};

template <typename T>
ExprHandle cast(const ExprHandle& src_value) {
  return Cast::make(Dtype(ToDtype<T>(), src_value.dtype().lanes()), src_value);
}

// This is a bitwise cast, akin to bitcast in LLVM
class TORCH_API BitCast : public ExprNode<BitCast> {
 public:
  ExprPtr src_value() const {
    return src_value_;
  }

  void set_src_value(ExprPtr src_value) {
    src_value_ = std::move(src_value);
  }

  static ExprHandle make(Dtype dtype, const ExprHandle& src_value) {
    return ExprHandle(alloc<BitCast>(dtype, src_value.node()));
  }
  BitCast(Dtype dtype, ExprPtr src_value)
      : ExprNodeBase(dtype, kBitCast), src_value_(std::move(src_value)) {
    TORCH_CHECK(src_value_->dtype().byte_size() == dtype.byte_size());
  }

  bool isConstant() const override {
    return src_value_->isConstant();
  }

 private:
  ExprPtr src_value_;
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
  ExprPtr lhs() const {
    return this->lhs_;
  }
  ExprPtr rhs() const {
    return this->rhs_;
  }

  void set_lhs(ExprPtr lhs) {
    lhs_ = std::move(lhs);
  }

  void set_rhs(ExprPtr rhs) {
    rhs_ = std::move(rhs);
  }

  static ExprHandle make(const ExprHandle& lhs, const ExprHandle& rhs) {
    return ExprHandle(alloc<Op>(lhs.node(), rhs.node()));
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  BinaryOpNode(
      ExprPtr lhs_v,
      ExprPtr rhs_v,
      IRNodeType expr_type,
      ScalarType ret_type = ScalarType::Undefined)
      : ExprNode<Op>(
            // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
            BinaryOpDtype(lhs_v->dtype(), rhs_v->dtype(), ret_type),
            expr_type),
        lhs_(CastIfNeeded(std::move(lhs_v), ExprNode<Op>::dtype())),
        rhs_(CastIfNeeded(std::move(rhs_v), ExprNode<Op>::dtype())) {}

 private:
  static ExprPtr CastIfNeeded(ExprPtr expr, Dtype dst_dtype) {
    if (expr->dtype() == dst_dtype) {
      return expr;
    }
    return Cast::make(dst_dtype, ExprHandle(std::move(expr))).node();
  }

  ExprPtr lhs_;
  ExprPtr rhs_;
};

namespace detail {
template <typename T>
void bin_op_deducer(BinaryOpNode<T>);
bool bin_op_deducer(...);
} // namespace detail

class TORCH_API Add : public BinaryOpNode<Add> {
 public:
  Add(ExprPtr lhs, ExprPtr rhs)
      : BinaryOpNode(std::move(lhs), std::move(rhs), IRNodeType::kAdd) {}
};

class TORCH_API Sub : public BinaryOpNode<Sub> {
 public:
  Sub(ExprPtr lhs, ExprPtr rhs)
      : BinaryOpNode(std::move(lhs), std::move(rhs), IRNodeType::kSub) {}
};

class TORCH_API Mul : public BinaryOpNode<Mul> {
 public:
  Mul(ExprPtr lhs, ExprPtr rhs)
      : BinaryOpNode(std::move(lhs), std::move(rhs), IRNodeType::kMul) {}
};

class TORCH_API Div : public BinaryOpNode<Div> {
 public:
  Div(ExprPtr lhs, ExprPtr rhs)
      : BinaryOpNode(std::move(lhs), std::move(rhs), IRNodeType::kDiv) {}
};

class TORCH_API Mod : public BinaryOpNode<Mod> {
 public:
  Mod(ExprPtr lhs, ExprPtr rhs)
      : BinaryOpNode(std::move(lhs), std::move(rhs), IRNodeType::kMod) {}
};

template <typename Op>
class BitwiseOpNode : public BinaryOpNode<Op> {
 public:
  BitwiseOpNode(ExprPtr lhs, ExprPtr rhs, IRNodeType type)
      : BinaryOpNode<Op>(std::move(lhs), std::move(rhs), type) {}

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

class TORCH_API And : public BitwiseOpNode<And> {
 public:
  And(ExprPtr lhs, ExprPtr rhs)
      : BitwiseOpNode(std::move(lhs), std::move(rhs), IRNodeType::kAnd) {}
};

class TORCH_API Or : public BitwiseOpNode<Or> {
 public:
  Or(ExprPtr lhs, ExprPtr rhs)
      : BitwiseOpNode(std::move(lhs), std::move(rhs), IRNodeType::kOr) {}
};

class TORCH_API Xor : public BitwiseOpNode<Xor> {
 public:
  Xor(ExprPtr lhs, ExprPtr rhs)
      : BitwiseOpNode(std::move(lhs), std::move(rhs), IRNodeType::kXor) {}
};

class TORCH_API Lshift : public BitwiseOpNode<Lshift> {
 public:
  Lshift(ExprPtr lhs, ExprPtr rhs)
      : BitwiseOpNode(std::move(lhs), std::move(rhs), IRNodeType::kLshift) {}
};

class TORCH_API Rshift : public BitwiseOpNode<Rshift> {
 public:
  Rshift(ExprPtr lhs, ExprPtr rhs)
      : BitwiseOpNode(std::move(lhs), std::move(rhs), IRNodeType::kRshift) {}
};

// TODO: add TORCH_API
// Currently adding it results in a compilation error on Windows
class Max : public BinaryOpNode<Max> {
 private:
  bool propagate_nans_;

 public:
  Max(ExprPtr lhs, ExprPtr rhs, bool propagate_nans)
      : BinaryOpNode(std::move(lhs), std::move(rhs), IRNodeType::kMax),
        propagate_nans_(propagate_nans) {}

  bool propagate_nans() const {
    return propagate_nans_;
  }

  static ExprHandle make(const ExprHandle& lhs, const ExprHandle& rhs) = delete;
  static ExprHandle make(
      const ExprHandle& lhs,
      const ExprHandle& rhs,
      bool propagate_nans) {
    return ExprHandle(alloc<Max>(lhs.node(), rhs.node(), propagate_nans));
  }
};

// TODO: add TORCH_API
// Currently adding it results in a compilation error on Windows
class Min : public BinaryOpNode<Min> {
 private:
  bool propagate_nans_;

 public:
  Min(ExprPtr lhs, ExprPtr rhs, bool propagate_nans)
      : BinaryOpNode(std::move(lhs), std::move(rhs), IRNodeType::kMin),
        propagate_nans_(propagate_nans) {}

  bool propagate_nans() const {
    return propagate_nans_;
  }

  static ExprHandle make(const ExprHandle& lhs, const ExprHandle& rhs) = delete;
  static ExprHandle make(
      const ExprHandle& lhs,
      const ExprHandle& rhs,
      bool propagate_nans) {
    return ExprHandle(alloc<Min>(lhs.node(), rhs.node(), propagate_nans));
  }
};

// Encode typed immediate values e.g. IntImm, FloatImm.
#define IMM_DECLARE(Type, Name)                               \
  class TORCH_API Name##Imm : public ExprNode<Name##Imm> {    \
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
      return ExprHandle(alloc<Name##Imm>(value));             \
    }                                                         \
                                                              \
   private:                                                   \
    Type value_;                                              \
  };
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_DECLARE);
#undef IMM_DECLARE

// Get immediate by ScalarType.
template <typename T>
ExprPtr getImmediateByType(ScalarType immType, T initialVal) {
  switch (immType) {
#define TYPE_CASE(Type, Name) \
  case ScalarType::Name:      \
    return alloc<Name##Imm>(Type(initialVal));
    // NOLINTNEXTLINE(bugprone-branch-clone)
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
#undef TYPE_CASE
    default:
      throw unsupported_dtype();
  }
  return nullptr;
}

template <typename T>
ExprPtr getImmediateByType(Dtype dtype, T initialVal) {
  return getImmediateByType<T>(dtype.scalar_type(), initialVal);
}

template <typename T>
ExprPtr immLike(const ExprPtr& e, T v) {
  return getImmediateByType<T>(e->dtype(), v);
}

template <typename T>
ExprPtr immLike(const ExprHandle& e, T v) {
  return immLike(e.node(), v);
}

inline std::optional<int64_t> intValue(const ExprPtr& e) {
#define TYPE_CASE(Type, Name)      \
  if (auto v = to<Name##Imm>(e)) { \
    return v->value();             \
  }
  AT_FORALL_INT_TYPES(TYPE_CASE);
#undef TYPE_CASE
  return c10::nullopt;
}

inline std::optional<int64_t> intValue(const ExprHandle& e) {
  return intValue(e.node());
}

template <typename T>
T immediateAs(const ExprPtr& e) {
#define TYPE_CASE(Type, Name)                \
  if (Name##ImmPtr imm = to<Name##Imm>(e)) { \
    return imm->value();                     \
  }
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
#undef TYPE_CASE
  throw unsupported_dtype();
  return 0;
}

template <typename T>
T immediateAs(const ExprHandle& e) {
  return immediateAs<T>(e.node());
}

template <typename T>
bool immediateEquals(const ExprPtr& e, T val) {
#define TYPE_CASE(Type, Name)                \
  if (Name##ImmPtr imm = to<Name##Imm>(e)) { \
    return imm->value() == val;              \
  }
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
#undef TYPE_CASE
  throw unsupported_dtype();
  return false;
}

TORCH_API bool immediateIsNegative(const ExprPtr& e);

TORCH_API bool immediateIsPositive(const ExprPtr& e);

TORCH_API bool immediateIsZero(const ExprPtr& e);

// Represents a ramp vector node:
//     [base, base + 1 * stride, ... , base + (lanes - 1) * stride]
class TORCH_API Ramp : public ExprNode<Ramp> {
 public:
  ExprPtr base() const {
    return base_;
  }
  ExprPtr stride() const {
    return stride_;
  }

  void set_base(ExprPtr base) {
    base_ = std::move(base);
  }

  void set_stride(ExprPtr stride) {
    stride_ = std::move(stride);
  }

  static ExprHandle make(
      const ExprHandle& base,
      const ExprHandle& stride,
      int lanes) {
    if (stride.dtype() != base.dtype()) {
      throw malformed_input("Bad stride in Ramp");
    }
    return ExprHandle(alloc<Ramp>(base.node(), stride.node(), lanes));
  }
  int lanes() const {
    return lanes_;
  }

  Ramp(ExprPtr base, ExprPtr stride, int lanes)
      : ExprNodeBase(Dtype(base->dtype(), lanes)),
        base_(std::move(base)),
        stride_(std::move(stride)),
        lanes_(lanes) {}

 private:
  ExprPtr base_;
  ExprPtr stride_;
  int lanes_;
};

class TORCH_API Load : public ExprNode<Load> {
 public:
  VarPtr base_handle() const {
    return buf_->base_handle();
  }
  std::vector<ExprPtr> indices() const {
    return indices_;
  }
  ExprPtr flat_index() const {
    TORCH_CHECK(indices_.size() == 1, "Indices haven't been flattened.");
    return indices_[0];
  }
  BufPtr buf() const {
    return buf_;
  }

  void set_buf(BufPtr buf) {
    buf_ = std::move(buf);
  }

  void set_indices(std::vector<ExprPtr> indices) {
    indices_ = std::move(indices);
  }

  static ExprHandle make(
      Dtype dtype,
      const BufHandle& buf,
      const std::vector<ExprHandle>& indices);
  static ExprHandle make(
      const BufHandle& buf,
      const std::vector<ExprHandle>& indices);

  Load(Dtype dtype, BufPtr base_handle, std::vector<ExprPtr> indices);
  Load(BufPtr base_handle, const std::vector<ExprPtr>& indices);

 private:
  BufPtr buf_;
  std::vector<ExprPtr> indices_;
};

class TORCH_API Broadcast : public ExprNode<Broadcast> {
 public:
  ExprPtr value() const {
    return value_;
  }

  void set_value(ExprPtr value) {
    value_ = std::move(value);
  }

  int lanes() const {
    return lanes_;
  }
  static ExprHandle make(const ExprHandle& value, int lanes) {
    return ExprHandle(alloc<Broadcast>(value.node(), lanes));
  }
  Broadcast(ExprPtr value, int lanes)
      : ExprNodeBase(Dtype(value->dtype(), lanes)),
        value_(std::move(value)),
        lanes_(lanes) {}

 private:
  ExprPtr value_;
  int lanes_;
};

class TORCH_API IfThenElse : public ExprNode<IfThenElse> {
 public:
  ExprPtr condition() const {
    return condition_;
  }

  // Lazily evaluated only if condition is true
  ExprPtr true_value() const {
    return true_;
  }

  // Lazily evaluated only if condition is false
  ExprPtr false_value() const {
    return false_;
  }

  void set_condition(ExprPtr condition) {
    condition_ = std::move(condition);
  }

  void set_true_value(ExprPtr true_value) {
    true_ = std::move(true_value);
  }

  void set_false_value(ExprPtr false_value) {
    false_ = std::move(false_value);
  }

  static ExprHandle make(
      const ExprHandle& c,
      const ExprHandle& t,
      const ExprHandle& f) {
    if (!c.dtype().is_integral()) {
      throw unsupported_dtype();
    }
    if (c.dtype().lanes() != 1) {
      throw unsupported_dtype();
    }
    if (t.dtype() != f.dtype()) {
      throw malformed_input("Bad dtype in IfThenElse");
    }
    return ExprHandle(alloc<IfThenElse>(c.node(), t.node(), f.node()));
  }

  IfThenElse(ExprPtr c, ExprPtr t, ExprPtr f)
      : ExprNodeBase(t->dtype()),
        condition_(std::move(c)),
        true_(std::move(t)),
        false_(std::move(f)) {}

 private:
  ExprPtr condition_;
  ExprPtr true_;
  ExprPtr false_;
};

class TORCH_API CompareSelect : public ExprNode<CompareSelect> {
 public:
  CompareSelectOperation compare_select_op() const {
    return compare_op_;
  }
  ExprPtr lhs() const {
    return this->lhs_;
  }
  ExprPtr rhs() const {
    return this->rhs_;
  }
  ExprPtr ret_val1() const {
    return this->ret_val1_;
  }
  ExprPtr ret_val2() const {
    return this->ret_val2_;
  }

  void set_lhs(ExprPtr lhs) {
    lhs_ = std::move(lhs);
  }

  void set_rhs(ExprPtr rhs) {
    rhs_ = std::move(rhs);
  }

  void set_ret_val1(ExprPtr ret_val1) {
    ret_val1_ = std::move(ret_val1);
  }

  void set_ret_val2(ExprPtr ret_val2) {
    ret_val2_ = std::move(ret_val2);
  }

  CompareSelectBias bias() const {
    return bias_;
  }

  static ExprHandle make(
      const ExprHandle& lhs,
      const ExprHandle& rhs,
      CompareSelectOperation cmp_op,
      CompareSelectBias bias = kUnbiased) {
    if (lhs.dtype() != rhs.dtype()) {
      throw malformed_input("bad dtype in CompareSelect");
    }
    return ExprHandle(alloc<CompareSelect>(
        lhs.node(),
        rhs.node(),
        IntImm::make(1).node(),
        IntImm::make(0).node(),
        cmp_op,
        bias));
  }

  static ExprHandle make(
      const ExprHandle& lhs,
      const ExprHandle& rhs,
      const ExprHandle& ret_val1,
      const ExprHandle& ret_val2,
      CompareSelectOperation cmp_op,
      CompareSelectBias bias = kUnbiased) {
    if (lhs.dtype() != rhs.dtype() || ret_val1.dtype() != ret_val2.dtype()) {
      throw malformed_input("bad dtype in CompareSelect");
    }
    return ExprHandle(alloc<CompareSelect>(
        lhs.node(),
        rhs.node(),
        ret_val1.node(),
        ret_val2.node(),
        cmp_op,
        bias));
  }

  CompareSelect(
      ExprPtr lhs,
      ExprPtr rhs,
      ExprPtr ret_val1,
      ExprPtr ret_val2,
      CompareSelectOperation cmp_op,
      CompareSelectBias bias = kUnbiased)
      : ExprNodeBase(ret_val1->dtype()),
        lhs_(std::move(lhs)),
        rhs_(std::move(rhs)),
        ret_val1_(std::move(ret_val1)),
        ret_val2_(std::move(ret_val2)),
        compare_op_(cmp_op),
        bias_(bias) {}

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  CompareSelect(
      ExprPtr lhs,
      ExprPtr rhs,
      CompareSelectOperation cmp_op,
      CompareSelectBias bias = kUnbiased)
      : ExprNodeBase(kInt),
        lhs_(std::move(lhs)),
        rhs_(std::move(rhs)),
        ret_val1_(alloc<IntImm>(1)),
        ret_val2_(alloc<IntImm>(0)),
        compare_op_(cmp_op),
        bias_(bias) {}

 private:
  ExprPtr lhs_;
  ExprPtr rhs_;
  ExprPtr ret_val1_;
  ExprPtr ret_val2_;
  CompareSelectOperation compare_op_;
  CompareSelectBias bias_;
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
  kMaxIntrinsicsOp,
};

class TORCH_API Intrinsics : public ExprNode<Intrinsics> {
 public:
  static ExprHandle make(IntrinsicsOp op_type, const ExprHandle& v1) {
    return ExprHandle(alloc<Intrinsics>(op_type, v1.node()));
  }

  static ExprHandle make(
      IntrinsicsOp op_type,
      const ExprHandle& v1,
      const ExprHandle& v2) {
    return ExprHandle(alloc<Intrinsics>(op_type, v1.node(), v2.node()));
  }

  static ExprHandle make(
      IntrinsicsOp op_type,
      const std::vector<ExprHandle>& params) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<ExprPtr> params_nodes(params.size());
    for (size_t i = 0; i < params.size(); i++) {
      params_nodes[i] = params[i].node();
    }
    return ExprHandle(alloc<Intrinsics>(op_type, params_nodes));
  }

  static ExprHandle make(IntrinsicsOp op_type, Dtype dtype) {
    return ExprHandle(alloc<Intrinsics>(op_type, dtype));
  }

  IntrinsicsOp op_type() const {
    return op_type_;
  }

  std::string func_name() const {
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

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Intrinsics(IntrinsicsOp op_type, Dtype dtype)
      : ExprNodeBase(IntrinsicsDtype(op_type, dtype)),
        params_({}),
        op_type_(op_type) {
    if (OpArgCount(op_type) != 0) {
      throw malformed_input("bad arg count in Intrinsics");
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Intrinsics(IntrinsicsOp op_type, ExprPtr v1)
      : ExprNodeBase(IntrinsicsDtype(op_type, v1->dtype())),
        params_({std::move(v1)}),
        op_type_(op_type) {
    if (OpArgCount(op_type) != 1) {
      throw malformed_input("bad arg count in Intrinsics");
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Intrinsics(IntrinsicsOp op_type, ExprPtr v1, ExprPtr v2)
      : ExprNodeBase(IntrinsicsDtype(op_type, v1->dtype(), v2->dtype())),
        params_({std::move(v1), std::move(v2)}),
        op_type_(op_type) {
    if (OpArgCount(op_type) != 2) {
      throw malformed_input("bad arg count in Intrinsics");
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Intrinsics(IntrinsicsOp op_type, const std::vector<ExprPtr>& params)
      : ExprNodeBase(IntrinsicsDtype(op_type, params)),
        params_(params),
        op_type_(op_type) {
    if (OpArgCount(op_type) != nparams()) {
      throw malformed_input("bad arg count in Intrinsics");
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Intrinsics(
      IntrinsicsOp op_type,
      Dtype dtype,
      const std::vector<ExprPtr>& params)
      : ExprNodeBase(IntrinsicsDtype(op_type, dtype)),
        params_(params),
        op_type_(op_type) {
    if (OpArgCount(op_type) != nparams()) {
      throw malformed_input("bad arg count in Intrinsics");
    }
  }

  bool isPure() const {
    return op_type_ != kRand;
  }

  int nparams() const {
    return params_.size();
  }

  ExprPtr param(int index) const {
    return params_[index];
  }
  const std::vector<ExprPtr>& params() const {
    return params_;
  }

  void set_params(std::vector<ExprPtr> params) {
    params_ = std::move(params);
  }

  static int OpArgCount(IntrinsicsOp op_type);

 private:
  static Dtype IntrinsicsDtype(IntrinsicsOp op_type, Dtype dt1);
  static Dtype IntrinsicsDtype(IntrinsicsOp op_type, Dtype dt1, Dtype dt2);
  static Dtype IntrinsicsDtype(
      IntrinsicsOp op_type,
      const std::vector<ExprPtr>& params);

  std::vector<ExprPtr> params_;
  IntrinsicsOp op_type_;
};

TORCH_API std::vector<ExprPtr> ExprHandleVectorToExprVector(
    const std::vector<ExprHandle>&);
TORCH_API std::vector<ExprHandle> ExprVectorToExprHandleVector(
    const std::vector<ExprPtr>&);
TORCH_API std::vector<VarPtr> VarHandleVectorToVarVector(
    const std::vector<VarHandle>&);
TORCH_API std::vector<VarHandle> VarVectorToVarHandleVector(
    const std::vector<VarPtr>&);
TORCH_API ExprPtr flatten_index(
    const std::vector<ExprPtr>& dims,
    const std::vector<ExprPtr>& indices,
    const std::vector<ExprPtr>& strides);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
