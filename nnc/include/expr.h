#ifndef NNC_INCLUDE_EXPR_H_INCLUDED_
#define NNC_INCLUDE_EXPR_H_INCLUDED_

#include "expr.h"
#include "ir_visitor.h"
#include "refcount.h"
#include "types.h"

namespace nnc {

// The common base between all IR expression node.
class BaseExprNode : public RefCounted {
 public:
  virtual void accept(IRVisitor* visitor) const = 0;
  BaseExprNode() : dtype_(kUninitialized) {}
  explicit BaseExprNode(Dtype dtype) : dtype_(dtype) {}
  Dtype dtype() const { return dtype_; }

 protected:
  void set_dtype(Dtype dtype) {
    CHECK_EQ(this->dtype_, Dtype::kUninitialized) << "can only set uninitialized dtype";
    CHECK_NE(dtype, Dtype::kUninitialized) << "new dtype must not be valid";
    this->dtype_ = dtype;
  }

 private:
  Dtype dtype_;
};

// A CRTP pattern to accept visitors for children class,
// and dispatch back to the children.
template <class Op>
class ExprNode : public BaseExprNode {
 public:
  using ExprNodeBase = ExprNode<Op>;
  void accept(IRVisitor* visitor) const override { visitor->visit(static_cast<const Op*>(this)); }
  explicit ExprNode(Dtype dtype) : BaseExprNode(dtype) {}
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

  Dtype dtype() const { return node()->dtype(); }

  // Handling the math operators.
  Expr operator+(const Expr& other) const;
  Expr operator-(const Expr& other) const;
  Expr operator*(const Expr& other) const;
  Expr operator/(const Expr& other) const;
};

}  // namespace nnc

#endif  // NNC_INCLUDE_EXPR_H_INCLUDED_
