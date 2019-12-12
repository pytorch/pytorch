#ifndef NNC_INCLUDE_EXPR_H_INCLUDED_
#define NNC_INCLUDE_EXPR_H_INCLUDED_

#include "ir_visitor.h"
#include "refcount.h"

namespace nnc {

// The common base between all IR expression node.
class BaseExprNode : public RefCounted {
 public:
  virtual void accept(IRVisitor* visitor) const = 0;
};

// A CRTP pattern to accept visitors for children class,
// and dispatch back to the children.
template <class Op>
class ExprNode : public BaseExprNode {
 public:
  void accept(IRVisitor* visitor) const override { visitor->visit(static_cast<const Op*>(this)); }
};

// A refcounted pointer to the underlying ExprNode.
// Also serves the primary way to build and operate on other expressions.
class Expr : public RefHandle<BaseExprNode> {
 public:
  using BaseHandle = RefHandle<BaseExprNode>;
  explicit Expr(BaseExprNode* node) : BaseHandle(node) {}

  void accept(IRVisitor* visitor) const {
    // TODO: Consider implement this without using recursion. Otherwise,
    // if the expression tree is degenerate and too long, it could cause a
    // stack overflow.
    node()->accept(visitor);
  }

  explicit Expr(int v);
  explicit Expr(float v);

  template <class Op>
  Op* AsNode() {
    BaseExprNode* node = this->node();
    return dynamic_cast<Op*>(node);
  }

  template <class Op>
  const Op* AsNode() const {
    Expr* this_non_const = const_cast<Expr*>(this);
    return this_non_const->AsNode<Op>();
  }

  // Handling the math operators.
  Expr operator+(const Expr& other) const;
  Expr operator-(const Expr& other) const;
  Expr operator*(const Expr& other) const;
  Expr operator/(const Expr& other) const;
};

}  // namespace nnc

#endif  // NNC_INCLUDE_EXPR_H_INCLUDED_
