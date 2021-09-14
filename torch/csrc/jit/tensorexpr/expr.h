/**
 * This file implements the core classes for Tensor Expressions.
 *
 * The structure of the expressions is inspired by Halide/TVM IR.
 */
#pragma once

#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
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
  kCast,
  kBitCast,
  kOther,
};

// The common base between all expression node.
class TORCH_API Expr : public std::enable_shared_from_this<Expr> {
 public:
  explicit Expr(Dtype dtype, IRNodeType expr_type = kOther)
      : dtype_(dtype), expr_type_(expr_type) {}
  virtual ~Expr() = default;
  Dtype dtype() const {
    return dtype_;
  }
  virtual void accept(IRVisitor* visitor) = 0;
  virtual ExprPtr accept_mutator(IRMutator* mutator) = 0;

  IRNodeType expr_type() const {
    return expr_type_;
  }
  // Is this a fixed (constant) immediate value.
  virtual bool isConstant() const {
    return false;
  }

  void set_dtype(Dtype dtype) {
    dtype_ = dtype;
  }

  /*
   * Make a deep copy of the given expression.
   *
   * All sub-expressions inside the given expressions are also cloned. Note
   * that the variables are not deep-copied since they are immutable.
   */
  static ExprPtr clone(ExprPtr s);

 protected:
  std::shared_ptr<Expr> getptr() {
    return shared_from_this();
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
  void accept(IRVisitor* visitor) override {
    visitor->visit(static_to<Op>(Base::getptr()));
  }
  ExprPtr accept_mutator(IRMutator* mutator) override;
  // pass the constructor to the base class
  using Base::Base;
};

// A wrapper object to the underlying ExprNode.
// Also serves the primary way to build and operate on other expressions.
class TORCH_API ExprHandle {
 public:
  ExprHandle() = default;
  explicit ExprHandle(ExprPtr node) : base_expr_node_(node) {}

  ExprPtr node() {
    return base_expr_node_;
  }

  ExprPtr node() const {
    return base_expr_node_;
  }

  bool empty() const {
    return base_expr_node_ == nullptr;
  }

#define IMM_EXPR_DECLARE(Type, Name) ExprHandle(Type v);
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_EXPR_DECLARE);
#undef IMM_EXPR_DECLARE

  template <class Op>
  NodePtr<Op> AsNode() {
    return to<Op>(this->node());
  }

  template <class Op>
  NodePtr<Op> AsNode() const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
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
  ExprHandle operator&&(const ExprHandle& other) const;
  ExprHandle operator||(const ExprHandle& other) const;
  ExprHandle operator^(const ExprHandle& other) const;
  ExprHandle operator<<(const ExprHandle& other) const;
  ExprHandle operator>>(const ExprHandle& other) const;

 private:
  ExprPtr base_expr_node_ = nullptr;
};

// The underlying representation node to a Var.
// Currently, each Var object represents a unique variable, even though the
// names might be the same. We should consider add a unique_name as well.
class TORCH_API Var : public ExprNode<Var> {
 public:
  static ExprHandle make(const std::string& name_hint, Dtype dtype) {
    return ExprHandle(alloc<Var>(name_hint, dtype));
  }
  static ExprHandle make(Dtype dtype) {
    return ExprHandle(alloc<Var>("", dtype));
  }

  // TODO: unique_name
  const std::string& name_hint() const {
    return name_hint_;
  }

  void set_name_hint(const std::string& name) {
    name_hint_ = name;
  }

  void set_name_hint(std::string&& name) {
    name_hint_ = name;
  }

  Var(std::string name_hint, Dtype dtype)
      : ExprNodeBase(dtype, kPrimitive), name_hint_(std::move(name_hint)) {}

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
  VarPtr base_handle() const {
    return base_handle_;
  }
  void set_base_handle(VarPtr base_handle) {
    base_handle_ = base_handle;
  }

  const std::string& name_hint() const {
    return base_handle_->name_hint();
  }
  void set_name_hint(const std::string& name_hint) {
    base_handle_->set_name_hint(name_hint);
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Buf(const std::string& name_hint,
      const std::vector<ExprPtr>& dims,
      Dtype dtype,
      ExprPtr initializer = nullptr)
      : Buf(alloc<Var>(name_hint, kHandle), dims, dtype, initializer) {}

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Buf(VarPtr var,
      std::vector<ExprPtr> dims,
      Dtype dtype,
      ExprPtr initializer = nullptr)
      : ExprNodeBase(dtype, kPrimitive),
        base_handle_(var),
        dims_(std::move(dims)),
        initializer_(initializer) {
    TORCH_CHECK(var);
  }

  size_t ndim() const {
    return dims_.size();
  }
  ExprPtr dim(size_t index) const {
    if (index >= ndim()) {
      throw out_of_range_index();
    }
    return dims_[index];
  }
  std::vector<ExprPtr> dims() const {
    return dims_;
  }
  void set_dims(std::vector<ExprPtr> dims) {
    dims_ = dims;
  };

  ExprPtr initializer() const {
    return initializer_;
  };

  bool hasConstantDims() const {
    for (auto d : dims_) {
      if (!d->isConstant()) {
        return false;
      }
    }
    return true;
  }

 private:
  VarPtr base_handle_;
  std::vector<ExprPtr> dims_;
  ExprPtr initializer_;
};

class TORCH_API BufHandle : public ExprHandle {
 public:
  BufHandle(
      const std::string& name_hint,
      const std::vector<ExprHandle>& dims,
      Dtype dtype)
      : ExprHandle(Buf::make(name_hint, dims, dtype)) {}

  BufHandle(const std::vector<ExprHandle>& dims, Dtype dtype)
      : ExprHandle(Buf::make("_", dims, dtype)) {}

  explicit BufHandle(Dtype dtype) : ExprHandle(Buf::make("_", {}, dtype)) {}

  explicit BufHandle(BufPtr node) : ExprHandle(node) {}
  BufPtr node() const {
    return static_to<Buf>(ExprHandle::node());
  }
  BufPtr node() {
    return static_to<Buf>(ExprHandle::node());
  }

  template <typename... Ts>
  inline ExprHandle load(const Ts&... ts) const;

  template <typename T>
  inline ExprHandle load(const std::vector<T>& args) const;

  inline ExprHandle load(const std::vector<ExprHandle>& args) const;

  StorePtr store(const std::vector<ExprHandle>& args, const ExprHandle& val)
      const;

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

  size_t ndim() const {
    return node()->ndim();
  }

  std::vector<ExprHandle> dims() const;

  ExprHandle dim(size_t index) const {
    return ExprHandle(node()->dim(index));
  }
};

// An expression to construct the underlying variable node.
// Note: do not store any info here, since it is often possible to slice this
// object. For example: VarHandle x('x'); ExprHandle x2 = x;
class TORCH_API VarHandle : public ExprHandle {
 public:
  // Creates an empty VarHandle whose base Var is set to nullptr.
  VarHandle() : ExprHandle() {}

  explicit VarHandle(Dtype dtype) : ExprHandle(Var::make(dtype)) {}

  VarHandle(const std::string& name_hint, Dtype dtype)
      : ExprHandle(Var::make(name_hint, dtype)) {}

  explicit VarHandle(VarPtr node) : ExprHandle(node) {}

  VarPtr node() const {
    return static_to<Var>(ExprHandle::node());
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
ExprPtr ExprNode<Op, Base>::accept_mutator(IRMutator* mutator) {
  return mutator->mutate(static_to<Op>(Base::getptr()));
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
TORCH_API ExprHandle Relu(const ExprHandle& v1);

TORCH_API ExprHandle
ifThenElse(const ExprHandle& c, const ExprHandle& t, const ExprHandle& f);

TORCH_API ExprHandle expr_to_vec(ExprHandle v, int lanes);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
