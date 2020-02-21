/**
 * This file implements the core classes for Tensor Expressions.
 *
 * The structure of the expressions is inspired by Halide/TVM IR.
 */
#pragma once

#include "torch/csrc/jit/tensorexpr/types.h"
#include "torch/csrc/jit/tensorexpr/mem_arena.h"

namespace torch {
namespace jit {
namespace tensorexpr {

// The common base between all expression node.
class Expr;
class BaseExprNode : public KernelScopedObject {
 public:
  explicit BaseExprNode(Dtype dtype) : dtype_(dtype) {}
  Dtype dtype() const {
    return dtype_;
  }

 private:
  Dtype dtype_;
};

// The common base between all statement node.
class BaseStmtNode : public KernelScopedObject {
 public:
  BaseStmtNode() {}
};

// A CRTP pattern to accept visitors for children class,
// and dispatch back to the children.
template <class Op, class Base = BaseExprNode>
class ExprNode : public Base {
 public:
  using ExprNodeBase = ExprNode<Op>;
  // pass the constructor to the base class
  using Base::Base;
};

template <class Op>
class StmtNode : public BaseStmtNode {
 public:
  using StmtNodeBase = StmtNode<Op>;
  StmtNode() {}
};

// A wrapper object to the underlying ExprNode.
// Also serves the primary way to build and operate on other expressions.
class TORCH_API Expr {
 public:
  Expr() {}
  explicit Expr(const BaseExprNode* node)
      : base_expr_node_(const_cast<BaseExprNode*>(node)) {}

  BaseExprNode* node() {
    return base_expr_node_;
  }

  const BaseExprNode* node() const {
    return base_expr_node_;
  }

  bool empty() const {
    return base_expr_node_ == nullptr;
  }

  Expr(int v);
  Expr(float v);

  template <class Op>
  Op* AsNode() {
    return dynamic_cast<Op*>(this->node());
  }

  template <class Op>
  const Op* AsNode() const {
    return const_cast<Expr*>(this)->AsNode<Op>();
  }

  Dtype dtype() const {
    return node()->dtype();
  }

  // Handling the math operators.
  Expr operator+(const Expr& other) const;
  Expr operator-(const Expr& other) const;
  Expr operator*(const Expr& other) const;
  Expr operator/(const Expr& other) const;
  Expr operator==(const Expr& other) const;
  Expr operator!=(const Expr& other) const;
  Expr operator>(const Expr& other) const;
  Expr operator>=(const Expr& other) const;
  Expr operator<(const Expr& other) const;
  Expr operator<=(const Expr& other) const;

 private:
  BaseExprNode* base_expr_node_ = nullptr;
};

class Stmt {
 public:
  Stmt() {}
  explicit Stmt(const BaseStmtNode* node)
      : base_stmt_node_(const_cast<BaseStmtNode*>(node)) {}

  BaseStmtNode* node() {
    return base_stmt_node_;
  }

  const BaseStmtNode* node() const {
    return base_stmt_node_;
  }

  bool empty() const {
    return node() == nullptr;
  }

  template <class Op>
  const Op* AsNode() const {
    return dynamic_cast<const Op*>(this->node());
  }

 private:
  BaseStmtNode* base_stmt_node_ = nullptr;
};

inline bool same_node(const Expr& expr1, const Expr& expr2) {
  return expr1.AsNode<BaseExprNode>() == expr2.AsNode<BaseExprNode>();
}

inline bool same_node(const Stmt& stmt1, const Stmt& stmt2) {
  return stmt1.AsNode<BaseStmtNode>() == stmt2.AsNode<BaseStmtNode>();
}

TORCH_API Expr ifThenElse(const Expr& c, const Expr& t, const Expr& f);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
