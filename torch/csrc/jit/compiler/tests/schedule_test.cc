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
  Tensor tensor =
      Compute("f", {{16, "X"}, {5, "y"}}, [](const Var& x, const Var& y) {
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
  Tensor tensor =
      Compute("f", {{16, "x"}, {5, "y"}}, [](const Var& x, const Var& y) {
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
  Tensor tensor = Compute("f", {{26, "x"}, {5, "y"}}, func);
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
  ASSERT_GT(oss.str().size(), 200);
  ASSERT_LT(oss.str().size(), 500);

  {
    // Compare to a reference loop structure structure.
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

  {
    // Evaluate its execution
    SimpleIREvaluator ir_eval;
    SimpleIREvaluator::BufferMapping buffer_mapping;
    // TODO: make this a standard testing helper.
    PaddedBuffer<float> f_v(26, 5, "f_v");
    PaddedBuffer<float> f_ref(26, 5, "f_res");

    buffer_mapping[tensor.function().func_var().node()] = f_v.data();
    ir_eval.SetBufferMapping(buffer_mapping);
    stmt.accept(&ir_eval);

    for (int x = 0; x < 26; x++) {
      for (int y = 0; y < 5; y++) {
        f_ref(x, y) = 1 + x * x + y * y;
      }
    }

    ExpectAllNear(f_v, f_ref, 1e-5);
  }
}

TEST(TestSchedule, BroadcastAddBuffer) {
  const int M = 4;
  const int N = 5;
  const int K = 6;
  Buffer a_buf("a", kFloat32, {M, N});
  Buffer b_buf("b", kFloat32, {N, K});
  Tensor c = Compute(
      "broadcast_add",
      {{M, "m"}, {N, "n"}, {K, "k"}},
      [&](const Var& m, const Var& n, const Var& k) {
        return a_buf(m, n) + b_buf(n, k);
      });
  Schedule sch({c});
  Stmt stmt = sch.Lower();

  PaddedBuffer<float> a_v(M, N, "a_v");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      a_v(m, n) = 7 * m * n;
    }
  }
  a_v.Backup();

  PaddedBuffer<float> b_v(N, K, "b_v");
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) {
      b_v(n, k) = 11 * n * k;
    }
  }
  b_v.Backup();

  PaddedBuffer<float> c_v(M, N, K, "c_buf");
  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c);
  ir_eval(a_v, b_v, c_v);

  a_v.CheckBackup();
  b_v.CheckBackup();
  PaddedBuffer<float> c_ref(M, N, K, "c_ref");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        c_ref(m, n, k) = 7 * m * n + 11 * n * k;
      }
    }
  }
  ExpectAllNear(c_v, c_ref, 1e-5);
}
