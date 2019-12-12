#ifndef NNC_INCLUDE_IR_H_INCLUDED_
#define NNC_INCLUDE_IR_H_INCLUDED_

#include <string>

#include "expr.h"

namespace nnc {

enum ExprNodeType {
  kAdd,
  kSub,
  kMul,
  kDiv,
};

// Represent the expression node for binary operators.
// A CRTP pattern to share common code among the operators.
template <typename Op>
class BinaryOpNode : public ExprNode<Op> {
 public:
  Expr& lhs() { return lhs_; }
  Expr& rhs() { return rhs_; }
  const Expr& lhs() const { return lhs_; }
  const Expr& rhs() const { return rhs_; }
  ExprNodeType expr_type() const { return expr_type_; }

  static Expr make(const Expr& lhs, const Expr& rhs) { return Expr(new Op(lhs, rhs)); }

 protected:
  BinaryOpNode(const Expr& lhs, const Expr& rhs, ExprNodeType expr_type)
      : lhs_(lhs), rhs_(rhs), expr_type_(expr_type) {}

 private:
  Expr lhs_;
  Expr rhs_;
  ExprNodeType expr_type_;
};

class Add : public BinaryOpNode<Add> {
 private:
  Add(const Expr& lhs, const Expr& rhs) : BinaryOpNode(lhs, rhs, ExprNodeType::kAdd) {}
  friend class BinaryOpNode<Add>;
};

class Sub : public BinaryOpNode<Sub> {
 private:
  Sub(const Expr& lhs, const Expr& rhs) : BinaryOpNode(lhs, rhs, ExprNodeType::kSub) {}
  friend class BinaryOpNode<Sub>;
};

class Mul : public BinaryOpNode<Mul> {
 private:
  Mul(const Expr& lhs, const Expr& rhs) : BinaryOpNode(lhs, rhs, ExprNodeType::kMul) {}
  friend class BinaryOpNode<Mul>;
};

class Div : public BinaryOpNode<Div> {
 private:
  Div(const Expr& lhs, const Expr& rhs) : BinaryOpNode(lhs, rhs, ExprNodeType::kDiv) {}
  friend class BinaryOpNode<Div>;
};

// Encode an integer immediate value.
class IntImm : public ExprNode<IntImm> {
 public:
  int value() const { return value_; }
  static Expr make(int value) { return Expr(new IntImm(value)); }

 private:
  IntImm(int value) : value_(value) {}
  int value_;
};

// Encode an fp32 immediate value.
class FloatImm : public ExprNode<FloatImm> {
 public:
  float value() const { return value_; }
  static Expr make(float value) { return Expr(new FloatImm(value)); }

 private:
  FloatImm(float value) : value_(value) {}
  float value_;
};

}  // namespace nnc

#endif  // NNC_INCLUDE_IR_H_INCLUDED_
