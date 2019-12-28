#pragma once

#include "torch/csrc/jit/compiler/include/ir_visitor.h"
#include "torch/csrc/jit/compiler/include/refcount.h"
#include "torch/csrc/jit/compiler/include/types.h"

namespace torch {
namespace jit {
namespace compiler {

// The commomn class between all IR nodes.
class IRNode : public RefCounted {
 public:
  virtual void accept(IRVisitor* visitor) const = 0;
  virtual ~IRNode() {}
};

// The common base between all expression node.
class BaseExprNode : public IRNode {
 public:
  explicit BaseExprNode(Dtype dtype) : dtype_(dtype) {}
  Dtype dtype() const {
    return dtype_;
  }

 private:
  Dtype dtype_;
};

// The common base between all statement node.
class BaseStmtNode : public IRNode {
 public:
  BaseStmtNode() {}
};

// A CRTP pattern to accept visitors for children class,
// and dispatch back to the children.
template <class Op>
class ExprNode : public BaseExprNode {
 public:
  using ExprNodeBase = ExprNode<Op>;
  void accept(IRVisitor* visitor) const override {
    visitor->visit(static_cast<const Op*>(this));
  }
  explicit ExprNode(Dtype dtype) : BaseExprNode(dtype) {}
};

template <class Op>
class StmtNode : public BaseStmtNode {
 public:
  using StmtNodeBase = StmtNode<Op>;
  void accept(IRVisitor* visitor) const override {
    visitor->visit(static_cast<const Op*>(this));
  }
  StmtNode() {}
};

// A refcounted pointer to the underlying ExprNode.
// Also serves the primary way to build and operate on other expressions.
class Expr : public RefHandle<BaseExprNode> {
 public:
  using BaseHandle = RefHandle<BaseExprNode>;
  explicit Expr() : BaseHandle(nullptr) {}
  explicit Expr(BaseExprNode* node) : BaseHandle(node) {}

  void accept(IRVisitor* visitor) const {
    // TODO: Consider implement this without using recursion. Otherwise,
    // if the expression tree is degenerate and too long, it could cause a
    // stack overflow.
    node()->accept(visitor);
  }

  Expr(int v);
  Expr(float v);

  template <class Op>
  const Op* AsNode() const {
    return dynamic_cast<const Op*>(this->node());
  }

  Dtype dtype() const {
    return node()->dtype();
  }

  // Handling the math operators.
  Expr operator+(const Expr& other) const;
  Expr operator-(const Expr& other) const;
  Expr operator*(const Expr& other) const;
  Expr operator/(const Expr& other) const;
};

class Stmt : public RefHandle<BaseStmtNode> {
 public:
  using BaseHandle = RefHandle<BaseStmtNode>;
  explicit Stmt(BaseStmtNode* node) : BaseHandle(node) {}

  void accept(IRVisitor* visitor) const {
    node()->accept(visitor);
  }

  template <class Op>
  const Op* AsNode() const {
    return dynamic_cast<const Op*>(this->node());
  }
};

} // namespace compiler
} // namespace jit
} // namespace torch
