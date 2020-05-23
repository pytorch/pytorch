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
  ASSERT_EQ(ss.str(), "2 + 3");
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
  ASSERT_EQ(ss.str(), "(2.f + 3.f) - (4.f + 5.f)");
}

void testIRPrinterCastTest() {
  KernelScope kernel_scope;
  VarHandle x("x", kHalf);
  VarHandle y("y", kFloat);
  ExprHandle body = ExprHandle(2.f) +
      (Cast::make(kFloat, x) * ExprHandle(3.f) + ExprHandle(4.f) * y);

  std::stringstream ss;
  ss << body;
  ASSERT_EQ(ss.str(), "2.f + (float(x) * 3.f + 4.f * y)");
}
} // namespace jit
} // namespace torch
