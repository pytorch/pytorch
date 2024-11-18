#include <gtest/gtest.h>

#include <stdexcept>
#include "test/cpp/tensorexpr/test_base.h"

#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/testing/file_check.h>

#include <sstream>
namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

TEST(IRPrinter, BasicValueTest) {
  ExprHandle a = IntImm::make(2), b = IntImm::make(3);
  ExprHandle c = Add::make(a, b);

  std::stringstream ss;
  ss << c;
  ASSERT_EQ(ss.str(), "2 + 3");
}

TEST(IRPrinter, BasicValueTest02) {
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle c(4.0f);
  ExprHandle d(5.0f);
  ExprHandle f = (a + b) - (c + d);

  std::stringstream ss;
  ss << f;
  ASSERT_EQ(ss.str(), "(2.f + 3.f) - (4.f + 5.f)");
}

TEST(IRPrinter, BasicValueTest03) {
  ExprHandle a(3.402823466385289e+38f);
  ExprHandle b(-3.402823466385289e+38f);
  std::stringstream ss;
  ss << a << ", " << b;
  ASSERT_EQ(ss.str(), "3.402823466385289e+38f, -3.402823466385289e+38f");
}

TEST(IRPrinter, CastTest) {
  VarHandle x("x", kHalf);
  VarHandle y("y", kFloat);
  ExprHandle body = ExprHandle(2.f) +
      (Cast::make(kFloat, x) * ExprHandle(3.f) + ExprHandle(4.f) * y);

  std::stringstream ss;
  ss << body;
  ASSERT_EQ(ss.str(), "2.f + (float(x) * 3.f + 4.f * y)");
}

TEST(IRPrinter, FunctionName) {
  int M = 4;
  int N = 20;

  Tensor producer = Compute(
      "producer", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        return m * n;
      });

  Tensor chunk_0 = Compute(
      "chunk_0", {M, N / 2}, [&](const ExprHandle& m, const ExprHandle& n) {
        return producer.load(m, n);
      });

  Tensor chunk_1 = Compute(
      "chunk_1", {M, N / 2}, [&](const ExprHandle& m, const ExprHandle& n) {
        return producer.load(m, n + ExprHandle(N / 2));
      });

  Tensor consumer = Compute(
      "consumer", {M, N / 2}, [&](const ExprHandle& i, const ExprHandle& j) {
        return i * chunk_1.load(i, j);
      });

  LoopNest l({chunk_0, chunk_1, consumer});
  auto body = LoopNest::sanitizeNames(l.root_stmt());

  std::stringstream ss;
  ss << *body;

  const std::string& verification_pattern =
      R"IR(
 # CHECK:   for (int i_2
 # CHECK:    for (int j_2
 # CHECK:     consumer[i_2, j_2] = i_2 * (chunk_1[i_2, j_2])IR";

  torch::jit::testing::FileCheck().run(verification_pattern, ss.str());
}
} // namespace jit
} // namespace torch
