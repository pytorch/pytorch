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
  Tensor tensor = Compute("f", {{16, "X"}, {5, "y"}}, [](const Var& x, const Var& y) {
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
      "f", {{16, "x"}, {5, "y"}}, [](const Var& x, const Var& y) {
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
    const int kPadding = 8;
    float kPaddingValue = 0.1357;
    std::vector<float> f_v(26 * 5 + 2 * kPadding);
    std::vector<float> f_ref(26 * 5 + 2 * kPadding);

    buffer_mapping[tensor.function().func_var().node()] = &f_v[kPadding];
    ir_eval.SetBufferMapping(buffer_mapping);
    stmt.accept(&ir_eval);

    float* f_ref_p = &f_ref[kPadding];
    for (int x = 0; x < 26; x++) {
      for (int y = 0; y < 5; y++) {
        f_ref_p[x * 5 + y] = 1 + x * x + y * y;
      }
    }

    for (int i = 0; i < f_v.size(); i++) {
      ASSERT_NEAR(f_v[i], f_ref[i], 1e-5);
    }
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

  const int kPaddingSize = 8;
  float kPaddingValue = 0.1357;
  std::vector<float> a_vec(M * N + 2 * kPaddingSize, kPaddingValue);
  std::vector<float> b_vec(N * K + 2 * kPaddingSize, kPaddingValue);
  std::vector<float> c_vec(M * N * K + 2 * kPaddingSize, kPaddingValue);

  std::vector<float> c_ref(c_vec);
  float* a_ptr = &a_vec[kPaddingSize];
  float* b_ptr = &b_vec[kPaddingSize];
  float* c_ptr = &c_ref[kPaddingSize];

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      a_ptr[m * N + n] = 7 * m * n;
    }
  }
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) {
      b_ptr[n * K + k] = 11 * n * k;
    }
  }
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        c_ptr[m * N * K + n * K + k] = 7 * m * n + 11 * n * k;
      }
    }
  }
  std::vector<float> a_ref(a_vec);
  std::vector<float> b_ref(b_vec);

  SimpleIREvaluator ir_eval;
  ir_eval.SetBufferMapping({
      {a_buf.data(), a_ptr},
      {b_buf.data(), b_ptr},
      {c.function().func_var(), &c_vec[kPaddingSize]},
  });
  stmt.accept(&ir_eval);

  ExpectAllNear(a_vec, a_ref, 1e-5, "a");
  ExpectAllNear(b_vec, b_ref, 1e-5, "b");
  ExpectAllNear(c_vec, c_ref, 1e-5, "c");
}
