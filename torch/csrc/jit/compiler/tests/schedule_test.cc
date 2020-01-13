#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <gtest/gtest.h>

#include "torch/csrc/jit/compiler/include/ir_printer.h"
#include "torch/csrc/jit/compiler/include/schedule.h"
#include "torch/csrc/jit/compiler/include/tensor.h"
#include "torch/csrc/jit/compiler/tests/test_utils.h"

using namespace torch::jit::compiler;
using namespace torch::jit::compiler::schedule;

TEST(TensorExpr, Simple01) {
  Tensor tensor = Compute(
      "f", {Expr(16), Expr(5)}, {"x", "y"}, [](const Var& x, const Var& y) {
        return Expr(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  Var x = tensor.function().arg(0);
  Var y = tensor.function().arg(1);
  Schedule sch = Schedule::make({tensor});
  Var x_outer;
  Var x_inner;
  Var x_tail;
  TensorOperation tail_op;
  tensor.SplitWithTail(x, 2, true, &x_outer, &x_inner, &x_tail, &tail_op);

  Var x_2;
  Var x_1;
  Var x_tail_2;
  TensorOperation tail_op_2;
  tensor.SplitWithTail(x_outer, 2, true, &x_2, &x_1, &x_tail_2, &tail_op_2);
}

TEST(TensorExpr, Lower01) {
  Tensor tensor = Compute(
      "f", {Expr(16), Expr(5)}, {"x", "y"}, [](const Var& x, const Var& y) {
        return Expr(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  Var x = tensor.function().arg(0);
  Var y = tensor.function().arg(1);
  Schedule sch = Schedule::make({tensor});
  Stmt stmt = sch.Lower();
  std::ostringstream oss;
  oss << stmt;
  ASSERT_GT(oss.str().size(), 20);
  ASSERT_LT(oss.str().size(), 200);
}

TEST(TensorExpr, Simple02) {
  auto func = [](const Expr& x, const Expr& y) {
    return Expr(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
  };
  Tensor tensor = Compute("f", {Expr(26), Expr(5)}, {"x", "y"}, func);
  Var x = tensor.function().arg(0);
  Var y = tensor.function().arg(1);
  Schedule sch = Schedule::make({tensor});
  Var x_outer;
  Var x_inner;
  Var x_tail;
  TensorOperation tail_op;
  tensor.SplitWithTail(x, 4, true, &x_outer, &x_inner, &x_tail, &tail_op);

  Stmt stmt = sch.Lower();
  std::ostringstream oss;
  oss << stmt;
  // TODO: switch to a better check

  ASSERT_GT(oss.str().size(), 200);
  ASSERT_LT(oss.str().size(), 500);

  {
    Var x_outer("x.outer", kInt32);
    Var x_inner("x.inner", kInt32);
    Var y("y", kInt32);
    Var x_tail("x.tail", kInt32);
    Var f("f", kHandle);
    Expr x_1 = x_outer * 4 + x_inner;
    Stmt stmt1 = For::make(
        x_outer,
        0,
        6,
        For::make(
            x_inner,
            0,
            4,
            For::make(
                y, 0, 5, Store::make(f, x_1 * 5 + y * 1, func(x_1, y), 1))));
    Expr x_2 = x_tail + Expr(6) * 4;
    Stmt stmt2 = For::make(
        x_tail,
        0,
        2,
        For::make(y, 0, 5, Store::make(f, x_2 * 5 + y * 1, func(x_2, y), 1)));
    Stmt stmt = Block::make({stmt1, stmt2});

    std::ostringstream oss_ref;
    oss_ref << stmt;
    ASSERT_EQ(oss.str(), oss_ref.str());
  }
}
