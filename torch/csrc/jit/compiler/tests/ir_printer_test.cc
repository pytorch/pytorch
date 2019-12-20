#include <stdexcept>

#include "torch/csrc/jit/compiler/include/expr.h"
#include "torch/csrc/jit/compiler/include/ir.h"
#include "torch/csrc/jit/compiler/include/ir_printer.h"

#include <gtest/gtest.h>
#include "torch/csrc/jit/compiler/tests/test_utils.h"

#include <sstream>

using namespace torch::jit::compiler;

TEST(IRPrinterTest, BasicValueTest) {
  Expr a = IntImm::make(2), b = IntImm::make(3);
  Expr c = Add::make(a, b);

  std::stringstream ss;
  ss << c;
  EXPECT_EQ(ss.str(), "(2 + 3)");
}

TEST(IRPrinterTest, BasicValueTest02) {
  Expr a(2.0f);
  Expr b(3.0f);
  Expr c(4.0f);
  Expr d(5.0f);
  Expr f = (a + b) - (c + d);

  std::stringstream ss;
  ss << f;
  EXPECT_EQ(ss.str(), "((2 + 3) + (4 + 5))");
}

TEST(IRPrinterTest, LetTest01) {
  Var x("x", kFloat32);
  Expr value = Expr(3.f);
  Expr body = Expr(2.f) + (x * Expr(3.f) + Expr(4.f));
  Expr result = Let::make(x, Expr(3.f), body);

  std::stringstream ss;
  ss << result;
  EXPECT_EQ(ss.str(), "(let x = 3 in (2 + ((x + 3) + 4)))");
}

TEST(IRPrinterTest, LetTest02) {
  Var x("x", kFloat32);
  Var y("y", kFloat32);
  Expr value = Expr(3.f);
  Expr body = Expr(2.f) + (x * Expr(3.f) + Expr(4.f) * y);
  Expr e1 = Let::make(x, Expr(3.f), body);
  Expr e2 = Let::make(y, Expr(6.f), e1);

  std::stringstream ss;
  ss << e2;
  EXPECT_EQ(
      ss.str(), "(let y = 6 in (let x = 3 in (2 + ((x + 3) + (4 + y)))))");
}

TEST(IRPrinterTest, CastTest) {
  Var x("x", kFloat32);
  Var y("y", kFloat32);
  Expr value = Expr(3.f);
  Expr body = Expr(2.f) + (x * Expr(3.f) + Expr(4.f) * y);
  Expr e1 = Let::make(x, Cast::make(kInt32, Expr(3.f)), body);
  Expr e2 = Let::make(y, Expr(6.f), e1);

  std::stringstream ss;
  ss << e2;
  EXPECT_EQ(
      ss.str(),
      "(let y = 6 in (let x = int32(3) in (2 + ((x + 3) + (4 + y)))))");
}
