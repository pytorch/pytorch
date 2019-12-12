#include <stdexcept>

#include <gtest/gtest.h>
#include "test_utils.h"

namespace nnc {

TEST(ExprTest, BasicValueTest) {
  Expr a = IntImm::make(2), b = IntImm::make(3);
  Expr c = Add::make(a, b);
  SimpleExprEvaluator<int> eval;
  c.accept(&eval);
  EXPECT_EQ(eval.value(), 5);
}

TEST(ExprTest, BasicValueTest02) {
  Expr a(2.0f);
  Expr b(3.0f);
  Expr c(4.0f);
  Expr d(5.0f);
  Expr f = (a + b) - (c + d);
  SimpleExprEvaluator<float> eval;
  f.accept(&eval);
  EXPECT_EQ(eval.value(), -4.0f);
}

TEST(ExprTest, LetTest01) {
  Var x("x");
  Expr value = Expr(3.f);
  Expr body = Expr(2.f) + (x * Expr(3.f) + Expr(4.f));
  Expr result = Let::make(x, Expr(3.f), body);
  SimpleExprEvaluator<float> eval;
  result.accept(&eval);
  EXPECT_EQ(eval.value(), 2 + (3 * 3 + 4));
}

TEST(ExprTest, LetTest02) {
  Var x("x");
  Var y("y");
  Expr value = Expr(3.f);
  Expr body = Expr(2.f) + (x * Expr(3.f) + Expr(4.f) * y);
  Expr e1 = Let::make(x, Expr(3.f), body);
  Expr e2 = Let::make(y, Expr(6.f), e1);
  SimpleExprEvaluator<float> eval;
  e2.accept(&eval);
  EXPECT_EQ(eval.value(), 2 + (3 * 3 + 4 * 6));
}

}  // namespace nnc
