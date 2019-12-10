#include <stdexcept>

#include <gtest/gtest.h>
#include <ir.h>

namespace nnc {

template <typename T>
class SimpleExprEvaluator : public IRVisitor {
 public:
  void visit(const Add *v) override {
    visit_binary_op(v);
  }

  void visit(const Sub *v) override {
    visit_binary_op(v);
  }

  void visit(const Mul *v) override {
    visit_binary_op(v);
  }

  void visit(const Div *v) override {
    visit_binary_op(v);
  }

  template <typename Op>
  void visit_binary_op(const BinaryOpNode<Op>* v) {
    v->lhs().accept(this);
    T lhs_v = this->value_;
    v->rhs().accept(this);
    T rhs_v = this->value_;
    switch (v->expr_type()) {
    case ExprNodeType::kAdd:
      this->value_ = lhs_v + rhs_v;
      break;
    case ExprNodeType::kSub:
      this->value_ = lhs_v - rhs_v;
      break; 
    case ExprNodeType::kMul:
      this->value_ = lhs_v * rhs_v;
      break;
    case ExprNodeType::kDiv:
      this->value_ = lhs_v / rhs_v;
      break;
    default:
      // TODO: change to a proper error report
      throw std::runtime_error("invalid operator type");
    }
  }

  void visit(const IntImm *v) override {
    value_ = (T)(v->value());
  }

  void visit(const FloatImm *v) override {
    value_ = (T)(v->value());
  }

  T value() const { return value_; }

 private:
  T value_ = T(0);
};

TEST(ExprTest, BasicValueTest) {
  Expr a = IntImm::make(2), b = IntImm::make(3);
  Expr c = Add::make(a, b);
  SimpleExprEvaluator<int> eval;
  c.accept(&eval);
  EXPECT_EQ(eval.value(), 5);
}

TEST(ExprTest, BasicValueTest02) {
  Expr a = FloatImm::make(2);
  Expr b = FloatImm::make(3);
  Expr c = FloatImm::make(4);
  Expr d = FloatImm::make(5);
  Expr f = (a + b) - (c + d);
  SimpleExprEvaluator<float> eval;
  f.accept(&eval);
  EXPECT_EQ(eval.value(), -4.0f);
}

} // namespace nnc
