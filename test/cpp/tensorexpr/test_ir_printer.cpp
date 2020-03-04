#include <stdexcept>
#include "test/cpp/tensorexpr/test_base.h"

#include "torch/csrc/jit/tensorexpr/expr.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"

#include <sstream>
namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

void testIRPrinterBasicValueTest() {
  KernelScope kernel_scope;
  ExprHandle a = IntImm::make(2), b = IntImm::make(3);
  ExprHandle c = Add::make(a, b);

  std::stringstream ss;
  ss << c;
  EXPECT_EQ(ss.str(), "2 + 3");
}

void testIRPrinterBasicValueTest02() {
  KernelScope kernel_scope;
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle c(4.0f);
  ExprHandle d(5.0f);
  ExprHandle f = (a + b) - (c + d);

  std::stringstream ss;
  ss << f;
  EXPECT_EQ(ss.str(), "(2.f + 3.f) - (4.f + 5.f)");
}

void testIRPrinterLetTest01() {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  ExprHandle value = ExprHandle(3.f);
  ExprHandle body = ExprHandle(2.f) + (x * ExprHandle(3.f) + ExprHandle(4.f));
  ExprHandle result = Let::make(x, ExprHandle(3.f), body);

  std::stringstream ss;
  ss << result;
  EXPECT_EQ(ss.str(), "let x = 3.f in 2.f + (x * 3.f + 4.f)");
}

void testIRPrinterLetTest02() {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  VarHandle y("y", kFloat);
  ExprHandle value = ExprHandle(3.f);
  ExprHandle body =
      ExprHandle(2.f) + (x * ExprHandle(3.f) + ExprHandle(4.f) * y);
  ExprHandle e1 = Let::make(x, ExprHandle(3.f), body);
  ExprHandle e2 = Let::make(y, ExprHandle(6.f), e1);

  std::stringstream ss;
  ss << e2;
  EXPECT_EQ(
      ss.str(), "let y = 6.f in (let x = 3.f in 2.f + (x * 3.f + 4.f * y))");
}

void testIRPrinterCastTest() {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  VarHandle y("y", kFloat);
  ExprHandle value = ExprHandle(3.f);
  ExprHandle body =
      ExprHandle(2.f) + (x * ExprHandle(3.f) + ExprHandle(4.f) * y);
  ExprHandle e1 = Let::make(x, Cast::make(kInt, ExprHandle(3.f)), body);
  ExprHandle e2 = Let::make(y, ExprHandle(6.f), e1);

  std::stringstream ss;
  ss << e2;
  EXPECT_EQ(
      ss.str(),
      "let y = 6.f in (let x = int(3.f) in 2.f + (x * 3.f + 4.f * y))");
}
} // namespace jit
} // namespace torch
