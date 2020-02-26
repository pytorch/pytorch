/**
 * This file implements the core classes for Tensor Expressions.
 *
 * The structure of the expressions is inspired by Halide/TVM IR.
 */
#pragma once

#include "torch/csrc/jit/tensorexpr/ir_mutator.h"
#include "torch/csrc/jit/tensorexpr/ir_visitor.h"
#include "torch/csrc/jit/tensorexpr/types.h"
#include "torch/csrc/jit/tensorexpr/mem_arena.h"

namespace torch {
namespace jit {
namespace tensorexpr {

// The common base between all expression node.
class ExprHandle;
class Expr : public KernelScopedObject {
 public:
  explicit Expr(Dtype dtype) : dtype_(dtype) {}
  Dtype dtype() const {
    return dtype_;
  }
  TORCH_API virtual void accept(IRVisitor* visitor) const = 0;
  virtual const Expr* accept_mutator(IRMutator* mutator) const = 0;

 private:
  Dtype dtype_;
};

// The common base between all statement node.
class Stmt : public KernelScopedObject {
 public:
  Stmt() {}
  TORCH_API virtual void accept(IRVisitor* visitor) const = 0;
  virtual Stmt* accept_mutator(IRMutator* mutator) = 0;
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

template <class Op>
class StmtNode : public Stmt {
 public:
  using StmtNodeBase = StmtNode<Op>;
  void accept(IRVisitor* visitor) const override {
    visitor->visit(static_cast<const Op*>(this));
  }
  Stmt* accept_mutator(IRMutator* mutator) override;
  StmtNode() {}
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

  ExprHandle(int v);
  ExprHandle(float v);

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
  ExprHandle operator^(const ExprHandle& other) const;
  ExprHandle operator<<(const ExprHandle& other) const;
  ExprHandle operator>>(const ExprHandle& other) const;

 private:
  Expr* base_expr_node_ = nullptr;
};

template <class Op, class Base>
const Expr* ExprNode<Op, Base>::accept_mutator(IRMutator* mutator) const {
  ExprNode* this_mutable = const_cast<ExprNode*>(this);
  return mutator->mutate(static_cast<Op*>(this_mutable));
}

template <class Op>
Stmt* StmtNode<Op>::accept_mutator(IRMutator* mutator) {
  StmtNode* this_mutable = const_cast<StmtNode*>(this);
  return mutator->mutate(static_cast<Op*>(this_mutable));
}

inline bool same_node(const ExprHandle& expr1, const ExprHandle& expr2) {
  return expr1.AsNode<Expr>() == expr2.AsNode<Expr>();
}

inline bool same_node(Stmt* stmt1, Stmt* stmt2) {
  return stmt1 == stmt2;
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
TORCH_API ExprHandle exp(const ExprHandle& v);
TORCH_API ExprHandle expm1(const ExprHandle& v);
TORCH_API ExprHandle fabs(const ExprHandle& v);
TORCH_API ExprHandle log(const ExprHandle& v);
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

TORCH_API ExprHandle ifThenElse(const ExprHandle& c, const ExprHandle& t, const ExprHandle& f);


} // namespace tensorexpr
} // namespace jit
} // namespace torch
