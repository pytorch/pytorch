/**
 * This file implements the core classes for Tensor Expressions.
 *
 * The structure of the expressions is inspired by Halide/TVM IR.
 */
#pragma once

#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/mem_arena.h>
#include <torch/csrc/jit/tensorexpr/types.h>

namespace torch {
namespace jit {
namespace tensorexpr {

enum IRNodeType {
  kPrimitive,
  kAdd,
  kSub,
  kMul,
  kDiv,
  kMod,
  kMax,
  kMin,
  kAnd,
  kOr,
  kLshift,
  kRshift,
  kXor,
  kCompareSelect,
  kLet,
  kCast,
  kBitCast,
  kBroadcast,
  kRamp,
  kPolynomial,
  kTerm,
  kRoundOff,
  kMaxTerm,
  kMinTerm,
  kNone,
  kExtra
};

// The common base between all expression node.
class Expr : public KernelScopedObject {
 public:
  explicit Expr(Dtype dtype, IRNodeType expr_type = kNone)
      : dtype_(dtype), expr_type_(expr_type) {}
  Dtype dtype() const {
    return dtype_;
  }
  TORCH_API virtual void accept(IRVisitor* visitor) const = 0;
  virtual const Expr* accept_mutator(IRMutator* mutator) const = 0;

  IRNodeType expr_type() const {
    return expr_type_;
  }
  // Is this a fixed (constant) immediate value.
  virtual bool isConstant() const {
    return false;
  }

 private:
  Dtype dtype_;
  IRNodeType expr_type_;
};

// A CRTP pattern to accept visitors for children class,
// and dispatch back to the children.
template <class Op, class Base = Expr>
class ExprNode : public Base {
 public:
  using ExprNodeBase = ExprNode<Op>;
  void accept(IRVisitor* visitor) const override {
    visitor->visit(static_cast<const Op*>(this));
  }
  const Expr* accept_mutator(IRMutator* mutator) const override;
  // pass the constructor to the base class
  using Base::Base;
};

// A wrapper object to the underlying ExprNode.
// Also serves the primary way to build and operate on other expressions.
class TORCH_API ExprHandle {
 public:
  ExprHandle() {}
  explicit ExprHandle(const Expr* node)
      : base_expr_node_(const_cast<Expr*>(node)) {}

  Expr* node() {
    return base_expr_node_;
  }

  const Expr* node() const {
    return base_expr_node_;
  }

  bool empty() const {
    return base_expr_node_ == nullptr;
  }

#define IMM_EXPR_DECLARE(Type, Name) ExprHandle(Type v);
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_EXPR_DECLARE);
#undef IMM_EXPR_DECLARE

  template <class Op>
  Op* AsNode() {
    return dynamic_cast<Op*>(this->node());
  }

  template <class Op>
  const Op* AsNode() const {
    return const_cast<ExprHandle*>(this)->AsNode<Op>();
  }

  Dtype dtype() const {
    return node()->dtype();
  }

  // Handling the math operators.
  ExprHandle operator+(const ExprHandle& other) const;
  ExprHandle operator-(const ExprHandle& other) const;
  ExprHandle operator*(const ExprHandle& other) const;
  ExprHandle operator/(const ExprHandle& other) const;
  ExprHandle operator%(const ExprHandle& other) const;
  ExprHandle operator==(const ExprHandle& other) const;
  ExprHandle operator!=(const ExprHandle& other) const;
  ExprHandle operator>(const ExprHandle& other) const;
  ExprHandle operator>=(const ExprHandle& other) const;
  ExprHandle operator<(const ExprHandle& other) const;
  ExprHandle operator<=(const ExprHandle& other) const;
  ExprHandle operator&(const ExprHandle& other) const;
  ExprHandle operator|(const ExprHandle& other) const;
  ExprHandle operator^(const ExprHandle& other) const;
  ExprHandle operator<<(const ExprHandle& other) const;
  ExprHandle operator>>(const ExprHandle& other) const;

 private:
  Expr* base_expr_node_ = nullptr;
};

// The underlying representation node to a Var.
// Currently, each Var object represents a unique variable, even though the
// names might be the same. We should consider add a unique_name as well.
class Var : public ExprNode<Var> {
 public:
  static ExprHandle make(const std::string& name_hint, Dtype dtype) {
    return ExprHandle(new Var(name_hint, dtype));
  }
  static ExprHandle make(Dtype dtype) {
    return ExprHandle(new Var("", dtype));
  }

  // TODO: unique_name
  const std::string& name_hint() const {
    return name_hint_;
  }

  Var(const std::string& name_hint, Dtype dtype)
      : ExprNodeBase(dtype, kPrimitive), name_hint_(name_hint) {}

 private:
  std::string name_hint_;
};

class TORCH_API Buf : public ExprNode<Buf> {
 public:
  static ExprHandle make(
      const std::string& name_hint,
      const std::vector<ExprHandle>& dims,
      Dtype dtype);
  static ExprHandle make(const std::vector<ExprHandle>& dims, Dtype dtype);

  // TODO: unique_name
  const Var* base_handle() const {
    return base_handle_;
  }
  const std::string& name_hint() const {
    return base_handle_->name_hint();
  }

  Buf(const std::string& name_hint,
      const std::vector<const Expr*>& dims,
      Dtype dtype,
      const Expr* initializer = nullptr)
      : Buf(new Var(name_hint, kHandle), dims, dtype, initializer) {}

  Buf(const Var* var,
      const std::vector<const Expr*>& dims,
      Dtype dtype,
      const Expr* initializer = nullptr)
      : ExprNodeBase(dtype, kPrimitive),
        base_handle_(var),
        dims_(dims),
        initializer_(initializer) {
    TORCH_CHECK(var);
  }

  size_t ndim() const {
    return dims_.size();
  }
  const Expr* dim(size_t index) const {
    if (index >= ndim()) {
      throw out_of_range_index();
    }
    return dims_[index];
  }
  std::vector<const Expr*> dims() const {
    return dims_;
  }
  void set_dims(std::vector<const Expr*> dims) {
    dims_ = dims;
  };

  const Expr* initializer() const {
    return initializer_;
  };

 private:
  const Var* base_handle_;
  std::vector<const Expr*> dims_;
  const Expr* initializer_;
};

class TORCH_API BufHandle : public ExprHandle {
 public:
  BufHandle(
      const std::string& name_hint,
      const std::vector<ExprHandle>& dims,
      Dtype dtype)
      : ExprHandle(Buf::make(name_hint, dims, dtype)) {}
  explicit BufHandle(const Buf* node) : ExprHandle(node) {}
  const Buf* node() const {
    return static_cast<const Buf*>(ExprHandle::node());
  }
  bool operator==(const BufHandle& other) const {
    return this->node() == other.node();
  }
  bool operator!=(const BufHandle& other) const {
    return !(*this == other);
  }

  const std::string& name_hint() const {
    return this->node()->name_hint();
  }
  bool empty() const {
    return (this->node() == nullptr);
  }
};

// An expression to construct the underlying variable node.
// Note: do not store any info here, since it is often possible to slice this
// object. For example: VarHandle x('x'); ExprHandle x2 = x;
class VarHandle : public ExprHandle {
 public:
  VarHandle() : ExprHandle(nullptr) {}
  explicit VarHandle(Dtype dtype) : ExprHandle(Var::make(dtype)) {}
  VarHandle(const std::string& name_hint, Dtype dtype)
      : ExprHandle(Var::make(name_hint, dtype)) {}
  explicit VarHandle(const Var* node) : ExprHandle(node) {}
  const Var* node() const {
    return static_cast<const Var*>(ExprHandle::node());
  }
  bool operator==(const VarHandle& other) const {
    return this->node() == other.node();
  }
  bool operator!=(const VarHandle& other) const {
    return !(*this == other);
  }

  const std::string& name_hint() const {
    return this->node()->name_hint();
  }
  bool empty() const {
    return (this->node() == nullptr);
  }
};

template <class Op, class Base>
const Expr* ExprNode<Op, Base>::accept_mutator(IRMutator* mutator) const {
  ExprNode* this_mutable = const_cast<ExprNode*>(this);
  return mutator->mutate(static_cast<Op*>(this_mutable));
}

inline bool same_node(const ExprHandle& expr1, const ExprHandle& expr2) {
  return expr1.AsNode<Expr>() == expr2.AsNode<Expr>();
}

TORCH_API ExprHandle sin(const ExprHandle& v);
TORCH_API ExprHandle cos(const ExprHandle& v);
TORCH_API ExprHandle tan(const ExprHandle& v);
TORCH_API ExprHandle asin(const ExprHandle& v);
TORCH_API ExprHandle acos(const ExprHandle& v);
TORCH_API ExprHandle atan(const ExprHandle& v);
TORCH_API ExprHandle sinh(const ExprHandle& v);
TORCH_API ExprHandle cosh(const ExprHandle& v);
TORCH_API ExprHandle tanh(const ExprHandle& v);
TORCH_API ExprHandle sigmoid(const ExprHandle& v);
TORCH_API ExprHandle exp(const ExprHandle& v);
TORCH_API ExprHandle expm1(const ExprHandle& v);
TORCH_API ExprHandle abs(const ExprHandle& v);
TORCH_API ExprHandle log(const ExprHandle& v);
TORCH_API ExprHandle fast_tanh(const ExprHandle& v);
TORCH_API ExprHandle fast_sigmoid(const ExprHandle& v);
TORCH_API ExprHandle fast_log(const ExprHandle& v);
TORCH_API ExprHandle log_vml(const ExprHandle& v);
TORCH_API ExprHandle log2(const ExprHandle& v);
TORCH_API ExprHandle log10(const ExprHandle& v);
TORCH_API ExprHandle log1p(const ExprHandle& v);
TORCH_API ExprHandle erf(const ExprHandle& v);
TORCH_API ExprHandle erfc(const ExprHandle& v);
TORCH_API ExprHandle sqrt(const ExprHandle& v);
TORCH_API ExprHandle rsqrt(const ExprHandle& v);
TORCH_API ExprHandle ceil(const ExprHandle& v);
TORCH_API ExprHandle floor(const ExprHandle& v);
TORCH_API ExprHandle round(const ExprHandle& v);
TORCH_API ExprHandle trunc(const ExprHandle& v);
TORCH_API ExprHandle frac(const ExprHandle& v);
TORCH_API ExprHandle lgamma(const ExprHandle& v);
TORCH_API ExprHandle atan2(const ExprHandle& v1, const ExprHandle& v2);
TORCH_API ExprHandle pow(const ExprHandle& v1, const ExprHandle& v2);
TORCH_API ExprHandle fmod(const ExprHandle& v1, const ExprHandle& v2);
TORCH_API ExprHandle remainder(const ExprHandle& v1, const ExprHandle& v2);
TORCH_API ExprHandle isnan(const ExprHandle& v1);

TORCH_API ExprHandle
ifThenElse(const ExprHandle& c, const ExprHandle& t, const ExprHandle& f);

TORCH_API ExprHandle expr_to_vec(ExprHandle v, int lanes);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
