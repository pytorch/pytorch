#ifndef NNC_INCLUDE_EXPR_H_INCLUDED_
#define NNC_INCLUDE_EXPR_H_INCLUDED_

#include "ir_visitor.h"
#include "refcount.h"

namespace nnc {

// The common base between all IR expression node.
class BaseExprNode : public RefCounted {
 public:
  virtual void accept(IRVisitor *visitor) const = 0;
};

// A CRTP pattern to accept visitors for children class,
// and dispatch back to the children.
template <class Op>
class ExprNode : public BaseExprNode {
 public:
  void accept(IRVisitor *visitor) const override {
    visitor->visit(static_cast<const Op*>(this));
  }
};

// A refcounted pointer to the underlying Expr node.
// Also serves the primary way to build and operate on other expressions.
class Expr {
 public:
  explicit Expr(BaseExprNode *node) : node_(node) {}

  ~Expr() { reset(); }

  // Handling refcount of the underlyng BaseExprNode
  Expr(const Expr& other) {
    this->reset();
    node_ = other.node_;
    node_->Ref();
  }

  Expr(Expr&& other) {
    node_ = other.node_;
    other.node_ = nullptr;
  }

  Expr& operator=(const Expr& other) {
    this->reset();
    node_ = other.node_;
    node_->Ref();
  }

  Expr& operator=(Expr&& other) {
    node_ = other.node_;
    other.node_ = nullptr;
  }

  void accept(IRVisitor *visitor) const {
    node_->accept(visitor);
  }

  // Handling the math operators.
  Expr operator+(const Expr& other) const;
  Expr operator-(const Expr& other) const;
  Expr operator*(const Expr& other) const;
  Expr operator/(const Expr& other) const;  

 private:
  void reset() {
    if (node_) {
      node_->Unref();
    }
    node_ = nullptr;
  }

  BaseExprNode *node_ = nullptr;
};

} // namespace nnc

#endif // NNC_INCLUDE_EXPR_H_INCLUDED_
