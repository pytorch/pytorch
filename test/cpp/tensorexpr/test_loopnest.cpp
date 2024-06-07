#include <gtest/gtest.h>

#include <test/cpp/tensorexpr/test_base.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <test/cpp/tensorexpr/padded_buffer.h>
#include <test/cpp/tensorexpr/test_utils.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

void checkIR(StmtPtr s, const std::string& pattern) {
  std::ostringstream oss;
  oss << *s;
  torch::jit::testing::FileCheck().run(pattern, oss.str());
}

void checkExprIR(ExprPtr e, const std::string& pattern) {
  std::string prefixed_pattern = "# CHECK: " + pattern + "\n";
  std::ostringstream oss;
  oss << *e << "\n";
  torch::jit::testing::FileCheck().run(prefixed_pattern, oss.str());
}

void checkExprIR(const ExprHandle& e, const std::string& pattern) {
  checkExprIR(e.node(), pattern);
}

TEST(LoopNest, ExprSimple01) {
  Tensor tensor =
      Compute("f", {16, 5}, [](const VarHandle& x, const VarHandle& y) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  LoopNest l({tensor});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);

  LoopNest::splitWithTail(loops[0], 2);
  LoopNest::splitWithTail(loops[0], 2);
}

TEST(LoopNest, ExprLower01) {
  Tensor tensor =
      Compute("f", {16, 5}, [](const VarHandle& x, const VarHandle& y) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  LoopNest l({tensor});
  StmtPtr stmt = l.root_stmt();
  std::ostringstream oss;
  oss << *stmt;
  ASSERT_GT(oss.str().size(), 20);
  ASSERT_LT(oss.str().size(), 200);
}

TEST(LoopNest, ExprSimple02) {
  auto func = [](const ExprHandle& x, const ExprHandle& y) {
    return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
  };
  Tensor tensor = Compute("f", {26, 5}, func);
  LoopNest l({tensor});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);

  LoopNest::splitWithTail(loops[0], 4);

  StmtPtr stmt = l.root_stmt();
  std::ostringstream oss;
  oss << *stmt;
  ASSERT_GT(oss.str().size(), 200);
  ASSERT_LT(oss.str().size(), 600);

  {
    // Compare to a reference loop structure structure.
    VarHandle x_outer("i_outer", kInt);
    VarHandle x_inner("i_inner", kInt);
    VarHandle y("i", kInt);
    VarHandle x_tail("i_tail", kInt);
    BufHandle f("f", {26, 5}, kFloat);
    ExprHandle x_1 = x_outer * 4 + x_inner;
    ExprHandle x_outer_end = (ExprHandle(26) - 0) / 4;
    ForPtr stmt1 = For::make(
        x_outer,
        0,
        x_outer_end,
        For::make(
            x_inner,
            0,
            4,
            For::make(y, 0, 5, Store::make(f, {x_1, y}, func(x_1, y)))));
    ExprHandle x_2 = x_tail + x_outer_end * 4;
    ForPtr stmt2 = For::make(
        x_tail,
        0,
        (ExprHandle(26) - 0) % 4,
        For::make(y, 0, 5, Store::make(f, {x_2, y}, func(x_2, y))));
    StmtPtr stmt = Block::make({stmt1, stmt2});

    std::ostringstream oss_ref;
    oss_ref << *stmt;
    ASSERT_EQ(oss.str(), oss_ref.str());
  }

  {
    PaddedBuffer<float> f_v(26, 5, "f_v");
    PaddedBuffer<float> f_ref(26, 5, "f_res");

    stmt = FlattenIndexes(stmt);
    SimpleIREvaluator ir_eval(stmt, {tensor});
    ir_eval(f_v);

    for (int x = 0; x < 26; x++) {
      for (int y = 0; y < 5; y++) {
        f_ref(x, y) = 1 + x * x + y * y;
      }
    }

    ExpectAllNear(f_v, f_ref, 1e-5);
  }
}

BlockPtr getSimplifiedBody(const LoopNest& l) {
  StmtPtr stmt = l.root_stmt();
  StmtPtr simplified = IRSimplifier::simplify(stmt);
  return to<Block>(simplified);
}

void assertForRange(ForPtr f, int expected_start, int expected_stop) {
  ASSERT_NE(f, nullptr);
  IntImmPtr start = to<IntImm>(f->start());
  ASSERT_NE(start, nullptr);
  ASSERT_EQ(start->value(), expected_start);
  IntImmPtr stop = to<IntImm>(f->stop());
  ASSERT_NE(stop, nullptr);
  ASSERT_EQ(stop->value(), expected_stop);
}

void assertForRanges(
    BlockPtr body,
    const std::vector<std::pair<int, int>>& start_stops) {
  ASSERT_EQ(body->nstmts(), start_stops.size());

  auto it = body->begin();
  for (size_t i = 0; i < start_stops.size(); i++, it++) {
    ForPtr loop = to<For>(*it);
    assertForRange(loop, start_stops[i].first, start_stops[i].second);
  }
}

TEST(LoopNest, ExprSliceHeadWithLoopOptions) {
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor tensor = Compute("f", {10}, func);
  LoopNest l({tensor});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr tail;
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  loops[0]->set_gpu_block_index(LoopOptions::IDX_Y);
  LoopNest::sliceHead(loops[0], 2, &head, &tail);

  BlockPtr body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 2}, {0, 8}});

  ASSERT_TRUE(tail->loop_options().is_gpu_block_index());
  ASSERT_EQ(tail->loop_options().gpu_block_index(), LoopOptions::IDX_Y);

  ASSERT_TRUE(head->loop_options().isDefault());
}

TEST(LoopNest, ExprSliceTailWithLoopOptions) {
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor tensor = Compute("f", {10}, func);
  LoopNest l({tensor});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr tail;
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  LoopNest::sliceTail(loops[0], 4, &head, &tail);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr tail_head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr tail_tail;
  tail->set_gpu_block_index(LoopOptions::IDX_Y);
  LoopNest::sliceTail(tail, 2, &tail_head, &tail_tail);

  BlockPtr body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 6}, {0, 2}, {8, 10}});

  ASSERT_TRUE(tail_head->loop_options().is_gpu_block_index());
  ASSERT_EQ(tail_head->loop_options().gpu_block_index(), LoopOptions::IDX_Y);

  ASSERT_TRUE(head->loop_options().isDefault());
  ASSERT_TRUE(tail_tail->loop_options().isDefault());
}

TEST(LoopNest, ExprSliceHeadWhenFactorEqualsSize) {
  // When factor equals the For loop's original size, keep using the original
  // For loop.
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor tensor = Compute("f", {10}, func);
  LoopNest l({tensor});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr tail;
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  LoopNest::sliceHead(loops[0], 10, &head, &tail);

  ASSERT_EQ(head, loops[0]);
  ASSERT_EQ(tail, nullptr);

  BlockPtr body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 10}});
}

TEST(LoopNest, ExprSliceHeadWhenFactorLargerThanSize) {
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor tensor = Compute("f", {10}, func);
  LoopNest l({tensor});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr tail;
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  LoopNest::sliceHead(loops[0], 100, &head, &tail);

  ASSERT_EQ(head, loops[0]);
  ASSERT_EQ(tail, nullptr);

  BlockPtr body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 10}});
}

TEST(LoopNest, ExprSliceHead) {
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor tensor = Compute("f", {10}, func);
  LoopNest l({tensor});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr tail;
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  LoopNest::sliceHead(loops[0], 4, &head, &tail);

  ASSERT_NE(head, nullptr);
  ASSERT_NE(head, loops[0]);
  ASSERT_NE(tail, nullptr);
  ASSERT_EQ(tail, loops[0]);

  BlockPtr body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 4}, {4, 10}});
}

TEST(LoopNest, ExprSliceHeadWithNonZeroStart) {
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor tensor = Compute("f", {10}, func);
  LoopNest l({tensor});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr tail;
  LoopNest::sliceTail(loops[0], 4, &head, &tail);
  // head: [0, 6)
  // tail: [6, 10)

  LoopNest::sliceHead(tail, 2);
  // tail_head: [6, 8)
  // tail_tail: [8, 10)

  BlockPtr body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 6}, {6, 8}, {8, 10}});
}

TEST(LoopNest, ExprSliceTailWhenFactorEqualsSize) {
  // When factor equals the For loop's original size, keep using the original
  // For loop.
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor tensor = Compute("f", {10}, func);
  LoopNest l({tensor});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr tail;
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  LoopNest::sliceTail(loops[0], 10, &head, &tail);

  ASSERT_EQ(head, nullptr);
  ASSERT_EQ(tail, loops[0]);

  BlockPtr body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 10}});
}

TEST(LoopNest, ExprSliceTailWhenFactorLargerThanSize) {
  // When factor equals the For loop's original size, keep using the original
  // For loop.
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor tensor = Compute("f", {10}, func);
  LoopNest l({tensor});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr tail;
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  LoopNest::sliceTail(loops[0], 100, &head, &tail);

  ASSERT_EQ(head, nullptr);
  ASSERT_EQ(tail, loops[0]);

  BlockPtr body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 10}});
}

TEST(LoopNest, ExprSliceTail) {
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor tensor = Compute("f", {10}, func);
  LoopNest l({tensor});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr tail;
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  LoopNest::sliceTail(loops[0], 4, &head, &tail);

  ASSERT_NE(head, nullptr);
  ASSERT_EQ(head, loops[0]);
  ASSERT_NE(tail, nullptr);
  ASSERT_NE(tail, loops[0]);

  BlockPtr body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 6}, {6, 10}});
}

TEST(LoopNest, ExprSplitAndSlice) {
  // 0: splitWithTail
  // 1: sliceTail on inner loop
  // 2: sliceHead on outer loop
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor tensor = Compute("f", {100}, func);
  LoopNest l({tensor});

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr inner;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr tail;
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // outer: [0, 4)
  // inner: [0, 21)
  // tail:  [84, 100)
  LoopNest::splitWithTail(loops[0], 21, &inner, &tail);
  LoopNest::sliceTail(inner, 2);
  LoopNest::sliceHead(loops[0], 2);

  // for (int x_outer = 0; x_outer < 2; x_outer++) {
  //   for (int x_inner = 0; x_inner < 19; x_inner++) {
  //     f[21 * x_outer + x_inner] = 1.f + float(21 * x_outer + x_inner);
  //   }
  //   for (int x_inner = 19; x_inner < 21; x_inner++) {
  //     f[21 * x_outer + x_inner] = 1.f + float(21 * x_outer + x_inner);
  //   }
  // }
  // for (int x_outer = 2; x_outer < 4; x_outer++) {
  //   for (int x_inner = 0; x_inner < 19; x_inner++) {
  //     f[21 * x_outer + x_inner] = 1.f + float(21 * x_outer + x_inner);
  //   }
  //   for (int x_inner = 19; x_inner < 21; x_inner++) {
  //     f[21 * x_outer + x_inner] = 1.f + float(21 * x_outer + x_inner);
  //   }
  // }
  // for (int x_tail = 0; x_tail < 16; x_tail++) {
  //   f[x_tail + 84] = 1.f + float(x_tail + 84);
  // }
  BlockPtr body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 2}, {2, 4}, {0, 16}});

  auto biter = body->begin();

  ForPtr loop = to<For>(*biter++);
  assertForRanges(loop->body(), {{0, 19}, {19, 21}});

  loop = to<For>(*biter);
  assertForRanges(loop->body(), {{0, 19}, {19, 21}});
}

TEST(LoopNest, ExprSliceAndNormalize) {
  // 0: sliceHead
  // 1: normalize tail
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor tensor = Compute("f", {10}, func);
  LoopNest l({tensor});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr tail;
  LoopNest::sliceHead(loops[0], 2, &head, &tail);
  // head: [0, 2)
  // tail: [2, 10)

  LoopNest::normalize(tail);
  // normalized_tail: [0, 8)

  BlockPtr body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 2}, {0, 8}});
}

template <typename T>
T evalExpr(const ExprHandle& expr, const VarHandle& var, T value) {
  ExprEval<SimpleIREvaluator> eval(expr, {var});
  return eval.value<T>(value);
}

TEST(LoopNest, ExprSliceWithVariableDimension) {
  auto testWithDimension =
      [](int dimension,
         const std::vector<std::pair<int, int>>& expected_for_ranges) {
        VarHandle dim("dim", kInt);
        Tensor tensor =
            Compute("f", {dim}, [](const ExprHandle& x) { return x; });
        LoopNest l({tensor});
        std::vector<ForPtr> loops =
            l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);

        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        ForPtr head;
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        ForPtr tail;
        LoopNest::sliceHead(loops[0], 2, &head, &tail);

        LoopNest::sliceTail(tail, 2);

        BlockPtr body = getSimplifiedBody(l);
        ASSERT_EQ(expected_for_ranges.size(), 3);
        auto it = body->begin();
        for (auto& start_stop : expected_for_ranges) {
          ForPtr loop = to<For>(*it++);
          int start = evalExpr<int>(ExprHandle(loop->start()), dim, dimension);
          int stop = evalExpr<int>(ExprHandle(loop->stop()), dim, dimension);
          ASSERT_EQ(start, start_stop.first);
          ASSERT_EQ(stop, start_stop.second);
        }
      };

  testWithDimension(1, {{0, 1}, {1, 1}, {1, 1}});
  testWithDimension(2, {{0, 2}, {2, 2}, {2, 2}});
  testWithDimension(3, {{0, 2}, {2, 2}, {2, 3}});
  testWithDimension(4, {{0, 2}, {2, 2}, {2, 4}});
  testWithDimension(5, {{0, 2}, {2, 3}, {3, 5}});
  testWithDimension(10, {{0, 2}, {2, 8}, {8, 10}});
}

TEST(LoopNest, ExprSplitWithTail) {
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor tensor = Compute("f", {199}, func);
  LoopNest l({tensor});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  LoopNest::splitWithTail(loops[0], 17);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  LoopNest::splitWithTail(loops[0], 7);

  StmtPtr stmt = l.root_stmt();
  StmtPtr simplified = IRSimplifier::simplify(stmt);
  BlockPtr body = to<Block>(simplified);
  ASSERT_EQ(body->nstmts(), 3);
  auto biter = body->begin();

  // Verify that the split loops are ordered correctly.
  ForPtr loop = to<For>(*biter++);
  assertForRange(loop, 0, 7);

  loop = to<For>(*biter++);
  assertForRange(loop, 0, 4);

  loop = to<For>(*biter);
  assertForRange(loop, 0, 12);
}

TEST(LoopNest, ExprSplitWithTailNone) {
  auto func = [](const ExprHandle& x, const ExprHandle& y) {
    return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
  };
  Tensor tensor = Compute("f", {24, 5}, func);
  LoopNest l({tensor});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  LoopNest::splitWithTail(loops[0], 4);

  StmtPtr stmt = l.root_stmt();
  std::ostringstream oss;
  oss << *stmt;
  ASSERT_GT(oss.str().size(), 200);
  ASSERT_LT(oss.str().size(), 600);

  {
    // Compare to a reference loop structure structure.
    VarHandle x_outer("i_outer", kInt);
    VarHandle x_inner("i_inner", kInt);
    VarHandle y("i", kInt);
    VarHandle x_tail("i_tail", kInt);
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks,cppcoreguidelines-avoid-magic-numbers)
    BufHandle f("f", {24, 5}, kFloat);
    ExprHandle x_1 = x_outer * 4 + x_inner;
    ExprHandle x_outer_end = (ExprHandle(24) - 0) / 4;
    StmtPtr stmt = alloc<Block>(std::vector<StmtPtr>({For::make(
        x_outer,
        0,
        x_outer_end,
        For::make(
            x_inner,
            0,
            4,
            For::make(y, 0, 5, Store::make(f, {x_1, y}, func(x_1, y)))))}));

    std::ostringstream oss_ref;
    oss_ref << *stmt;
    ASSERT_EQ(oss.str(), oss_ref.str());
  }

  {
    PaddedBuffer<float> f_v(24, 5, "f_v");
    PaddedBuffer<float> f_ref(24, 5, "f_res");

    SimpleIREvaluator ir_eval(stmt, {tensor});
    ir_eval(f_v);

    for (int x = 0; x < 24; x++) {
      for (int y = 0; y < 5; y++) {
        f_ref(x, y) = 1 + x * x + y * y;
      }
    }

    ExpectAllNear(f_v, f_ref, 1e-5);
  }
}

TEST(LoopNest, ExprSplitWithMask01) {
  const int M = 26;
  const int N = 5;
  BufHandle a_buf("a", {M, N}, kFloat);
  BufHandle b_buf("b", {M, N}, kFloat);
  Tensor tensor =
      Compute("f", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        return a_buf.load(m, n) + b_buf.load(m, n) + 1.0f;
      });

  LoopNest l({tensor});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  LoopNest::splitWithMask(loops[1], 4);

  StmtPtr stmt = l.root_stmt();

  PaddedBuffer<float> a_v(M, N, "a");
  PaddedBuffer<float> b_v(M, N, "b");
  PaddedBuffer<float> c_v(M, N, "c");
  PaddedBuffer<float> c_ref(M, N, "c_ref");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      a_v(m, n) = 2 * m;
      b_v(m, n) = 3 * n;
      c_ref(m, n) = a_v(m, n) + b_v(m, n) + 1.0f;
    }
  }

  SimpleIREvaluator(stmt, {a_buf, b_buf, tensor})(a_v, b_v, c_v);

  ExpectAllNear(c_v, c_ref, 1e-5);
}

// Tests the case where we split a loop cleanly multiple times, we should not
// insert any masks.
TEST(LoopNest, ExprSplitWithMaskRepeatedNoMask) {
  const int M = 64;
  BufHandle a_buf("a", {M}, kFloat);
  BufHandle b_buf("b", {M}, kFloat);
  Tensor tensor = Compute("f", {M}, [&](const ExprHandle& m) {
    return a_buf.load(m) + b_buf.load(m) + 1.0f;
  });

  LoopNest l({tensor});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  LoopNest::splitWithMask(loops[0], 4);
  LoopNest::splitWithMask(loops[0], 4);

  StmtPtr stmt1 = IRSimplifier::simplify(l.root_stmt());

  // Two splits mean 3 loops, but should need no masks in this case.
  checkIR(stmt1, R"IR(
# CHECK: for (
# CHECK-NOT: if (
# CHECK:   for (
# CHECK-NOT: if (
# CHECK:     for (
# CHECK-NOT: if (
# CHECK:       f[)IR");
}

TEST(LoopNest, getLoopAt) {
  // Input IR:
  //  for (int i = 0; i < 100; i++) {
  //    for (int j = 0; j < 100; j++) {
  //      A[i, j] = sin(i * j);
  //      for (int k1 = 0; k1 < 200; k1++) {
  //        B[i, j, k1] = (A[i, j]) / (k1 + 1);
  //      }
  //      for (int k2 = 0; k2 < 300; k2++) {
  //        C[i, j, k2] = (A[i, j]) * (k2 + 1);
  //      }
  //    }
  //  }
  BufPtr A = alloc<Buf>(
      "A",
      std::vector<ExprPtr>({alloc<IntImm>(100), alloc<IntImm>(100)}),
      kInt);
  BufPtr B = alloc<Buf>(
      "B",
      std::vector<ExprPtr>(
          {alloc<IntImm>(100), alloc<IntImm>(100), alloc<IntImm>(200)}),
      kInt);
  BufPtr C = alloc<Buf>(
      "C",
      std::vector<ExprPtr>(
          {alloc<IntImm>(100), alloc<IntImm>(100), alloc<IntImm>(300)}),
      kInt);
  BufHandle a_buf(A);
  BufHandle b_buf(B);
  BufHandle c_buf(C);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k1("k1", kInt);
  VarHandle k2("k2", kInt);
  auto store1 = Store::make(a_buf, {i, j}, sin(i * j));
  auto store2 = Store::make(
      b_buf, {i, j, k1}, Div::make(Load::make(a_buf, {i, j}), (k1 + 1)));
  auto store3 = Store::make(
      c_buf, {i, j, k2}, Mul::make(Load::make(a_buf, {i, j}), (k2 + 1)));
  auto for_k2 = For::make(k2, 0, 300, Block::make({store3}));
  auto for_k1 = For::make(k1, 0, 200, Block::make({store2}));
  auto for_j = For::make(j, 0, 100, Block::make({store1, for_k1, for_k2}));
  auto for_i = For::make(i, 0, 100, for_j);
  LoopNest l(Block::make({for_i}), {B, C});
  auto ret_k2 = l.getLoopAt(for_i, {0, 2});
  TORCH_CHECK(ret_k2 == for_k2);

  std::ostringstream oss;
  oss << *ret_k2;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int k2
# CHECK-NEXT: C[i, j, k2] =
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(LoopNest, TileSimple) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  const int M = 64, N = 64;
  BufHandle a_buf("a", {M, N}, kFloat);
  BufHandle b_buf("b", {M, N}, kFloat);
  Tensor tensor =
      Compute("f", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        return a_buf.load({m, n}) + b_buf.load({m, n}) + 1.0f;
      });

  LoopNest l({tensor});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  l.tile(loops[0], loops[1], 4, 8);

  // IR check
  StmtPtr stmt = IRSimplifier::simplify(l.root_stmt());
  checkIR(stmt, R"IR(
# CHECK: for (int i_outer
# CHECK:   for (int i_outer_1
# CHECK:     for (int i_inner
# CHECK:       for (int i_inner_1
# CHECK:         f[
# CHECK-NOT:     for (int i_tail
# CHECK-NOT: for (int i_tail)IR");

  // Correctness check
  PaddedBuffer<float> a_v(M, N, "a");
  PaddedBuffer<float> b_v(M, N, "b");
  PaddedBuffer<float> c_v(M, N, "c");
  PaddedBuffer<float> c_ref(M, N, "c_ref");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      a_v(m, n) = 2 * m;
      b_v(m, n) = 3 * n;
      c_ref(m, n) = a_v(m, n) + b_v(m, n) + 1.0f;
    }
  }

  SimpleIREvaluator(stmt, {a_buf, b_buf, tensor})(a_v, b_v, c_v);

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  ExpectAllNear(c_v, c_ref, 1e-5);
}

TEST(LoopNest, TileWithTails) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  const int M = 64, N = 64;
  BufHandle a_buf("a", {M, N}, kFloat);
  BufHandle b_buf("b", {M, N}, kFloat);
  Tensor tensor =
      Compute("f", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        return a_buf.load({m, n}) + b_buf.load({m, n}) + 1.0f;
      });

  LoopNest l({tensor});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  l.tile(loops[0], loops[1], 5, 9);

  // IR check
  StmtPtr stmt = IRSimplifier::simplify(l.root_stmt());
  checkIR(stmt, R"IR(
# CHECK: for (int i_outer
# CHECK:   for (int i_outer_1
# CHECK:     for (int i_inner
# CHECK:       for (int i_inner_1
# CHECK:         f[
# CHECK:   for (int i_inner
# CHECK:     f[
# CHECK: for (int i_tail)IR");

  // Correctness check
  PaddedBuffer<float> a_v(M, N, "a");
  PaddedBuffer<float> b_v(M, N, "b");
  PaddedBuffer<float> c_v(M, N, "c");
  PaddedBuffer<float> c_ref(M, N, "c_ref");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      a_v(m, n) = 2 * m;
      b_v(m, n) = 3 * n;
      c_ref(m, n) = a_v(m, n) + b_v(m, n) + 1.0f;
    }
  }

  SimpleIREvaluator(stmt, {a_buf, b_buf, tensor})(a_v, b_v, c_v);

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  ExpectAllNear(c_v, c_ref, 1e-5);
}

TEST(LoopNest, TileInMiddle) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  const int M = 8, N = 8, L = 8, K = 8;
  BufHandle a_buf("a", {M, N, L, K}, kFloat);
  BufHandle b_buf("b", {M, N, L, K}, kFloat);
  Tensor tensor = Compute(
      "f",
      {M, N, L, K},
      [&](const ExprHandle& m,
          const ExprHandle& n,
          const ExprHandle& l,
          const ExprHandle& k) {
        return a_buf.load({m, n, l, k}) + b_buf.load({m, n, l, k}) + 1.0f;
      });

  LoopNest nest({tensor});
  std::vector<ForPtr> loops =
      nest.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  nest.tile(loops[1], loops[2], 3, 3);

  // IR check
  StmtPtr stmt = IRSimplifier::simplify(nest.root_stmt());
  checkIR(stmt, R"IR(
# CHECK: for (int i
# CHECK:   for (int i_outer
# CHECK:     for (int i_outer_1
# CHECK:       for (int i_inner
# CHECK:         for (int i_inner_1
# CHECK:           for (int i_1
# CHECK:             f[
# CHECK:     for (int i_tail_1
# CHECK:       for (int i_inner_1
# CHECK:         for (int i_1
# CHECK:           f[
# CHECK:   for (int i_tail)IR");

  // Correctness check
  PaddedBuffer<float> a_v(M, N, L, K, "a");
  PaddedBuffer<float> b_v(M, N, L, K, "b");
  PaddedBuffer<float> c_v(M, N, L, K, "c");
  PaddedBuffer<float> c_ref(M, N, L, K, "c_ref");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int l = 0; l < L; l++) {
        for (int k = 0; k < K; k++) {
          a_v(m, n, l, k) = 2 * (m + l);
          b_v(m, n, l, k) = 3 * (n + k);
          c_ref(m, n, l, k) = a_v(m, n, l, k) + b_v(m, n, l, k) + 1.0f;
        }
      }
    }
  }

  SimpleIREvaluator(stmt, {a_buf, b_buf, tensor})(a_v, b_v, c_v);

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  ExpectAllNear(c_v, c_ref, 1e-5);
}

TEST(LoopNest, SplitWithTailWithLoopOptions) {
  const int M = 21;
  BufHandle a_buf("a", {M}, kFloat);
  BufHandle b_buf("b", {M}, kFloat);
  Tensor tensor = Compute("f", {M}, [&](const ExprHandle& m) {
    return a_buf.load(m) + b_buf.load(m) + 1.0f;
  });
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr inner, tail;

  LoopNest l({tensor});
  auto loops = NodeFinder<For>::find(l.root_stmt());
  ASSERT_GT(loops.size(), 0);
  loops[0]->set_gpu_block_index(LoopOptions::IDX_Y);
  LoopNest::splitWithTail(loops[0], 4, &inner, &tail);
  ASSERT_NE(inner, nullptr);
  ASSERT_NE(tail, nullptr);
  ForPtr outer = loops[0];

  // Outer loop carries loop axis bindings.
  ASSERT_TRUE(outer->loop_options().is_gpu_block_index());
  ASSERT_EQ(outer->loop_options().gpu_block_index(), LoopOptions::IDX_Y);

  // Inner loop has none.
  ASSERT_TRUE(inner->loop_options().isDefault());

  // Tail loop has none.
  ASSERT_TRUE(tail->loop_options().isDefault());
}

TEST(LoopNest, SplitWithMaskWithLoopOptions) {
  const int M = 21;
  BufHandle a_buf("a", {M}, kFloat);
  BufHandle b_buf("b", {M}, kFloat);
  Tensor tensor = Compute("f", {M}, [&](const ExprHandle& m) {
    return a_buf.load(m) + b_buf.load(m) + 1.0f;
  });
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr inner;

  LoopNest l({tensor});
  auto loops = NodeFinder<For>::find(l.root_stmt());
  loops[0]->set_gpu_block_index(LoopOptions::IDX_Y);
  LoopNest::splitWithMask(loops[0], 4, &inner);
  ForPtr outer = loops[0];

  // Outer loop carries loop axis bindings.
  ASSERT_TRUE(outer->loop_options().is_gpu_block_index());
  ASSERT_EQ(outer->loop_options().gpu_block_index(), LoopOptions::IDX_Y);

  // Inner loop has none.
  ASSERT_TRUE(inner->loop_options().isDefault());
}

TEST(LoopNest, ScheduleBroadcastAddBuffer) {
  const int M = 4;
  const int N = 5;
  const int K = 6;
  BufHandle a_buf("a", {M, N}, kFloat);
  BufHandle b_buf("b", {N, K}, kFloat);
  Tensor c = Compute(
      "broadcast_add",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) + b_buf.load(n, k);
      });
  LoopNest l({c});
  StmtPtr stmt = l.root_stmt();

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
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c});
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

TEST(LoopNest, ScheduleFunctionCall01) {
  const int M = 4;
  const int N = 5;
  const int K = 6;
  BufHandle a_buf("a", {M, N}, kFloat);
  BufHandle b_buf("b", {N, K}, kFloat);
  Tensor c = Compute(
      "broadcast_add",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) + b_buf.load(n, k);
      });
  Tensor d = Compute(
      "d",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return c.load(m, n, k) + 1;
      });

  LoopNest l({d}, {c, d});
  l.prepareForCodegen();
  StmtPtr stmt = l.root_stmt();
  std::ostringstream oss;
  oss << *stmt;
  ASSERT_GT(oss.str().size(), 100);

  PaddedBuffer<float> a_v(M, N);
  PaddedBuffer<float> b_v(N, K);
  PaddedBuffer<float> c_v(M, N, K);
  PaddedBuffer<float> d_v(M, N, K);
  PaddedBuffer<float> d_ref(M, N, K);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      a_v(i, j) = i * i;
    }
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      b_v(i, j) = j * j;
    }
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        d_ref(i, j, k) = a_v(i, j) + b_v(j, k) + 1;
      }
    }
  }

  SimpleIREvaluator eval(stmt, {a_buf, b_buf, d});
  eval(a_v, b_v, d_v);

  ExpectAllNear(d_v, d_ref, 1e-5);
}

TEST(LoopNest, ScheduleInlineSimple) {
  const int M = 4;
  const int N = 5;
  const int K = 6;
  BufHandle a_buf("a", {M, N}, kFloat);
  BufHandle b_buf("b", {N, K}, kFloat);
  BufHandle c_buf("c", {M, N}, kFloat);
  BufHandle d_buf("d", {M, K}, kFloat);

  Tensor x = Compute(
      "x",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) * b_buf.load(n, k);
      });
  Tensor y = Compute(
      "y",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return c_buf.load(m, n) * d_buf.load(m, k) + x.load(m, n, k);
      });

  LoopNest l1({y}, {x, y});
  LoopNest l2(l1);
  l2.computeInline(x.buf());

  l1.prepareForCodegen();
  l2.prepareForCodegen();

  StmtPtr stmt1 = IRSimplifier::simplify(l1.root_stmt());
  StmtPtr stmt2 = IRSimplifier::simplify(l2.root_stmt());

  SimpleIREvaluator eval1(stmt1, {a_buf, b_buf, c_buf, d_buf, y});
  SimpleIREvaluator eval2(stmt2, {a_buf, b_buf, c_buf, d_buf, y});

  PaddedBuffer<float> a_v(M, N);
  PaddedBuffer<float> b_v(N, K);
  PaddedBuffer<float> c_v(M, N);
  PaddedBuffer<float> d_v(M, K);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      a_v(i, j) = i * i;
    }
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      b_v(i, j) = j * j;
    }
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      c_v(i, j) = i + j;
    }
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      d_v(i, j) = i * j;
    }
  }

  PaddedBuffer<float> y_1(M, N, K);
  PaddedBuffer<float> y_2(M, N, K);

  eval1(a_v, b_v, c_v, d_v, y_1);
  eval2(a_v, b_v, c_v, d_v, y_2);
  ExpectAllNear(y_1, y_2, 1e-5);
  std::ostringstream oss1, oss2;
  oss1 << *stmt1;
  oss2 << *stmt2;
  ASSERT_GT(oss1.str().size(), oss2.str().size());
}

static std::string remove_space(const std::string& str) {
  std::string str_new = str;
  str_new.erase(
      remove_if(str_new.begin(), str_new.end(), isspace), str_new.end());
  return str_new;
}

void InlineFunc01Helper(const std::vector<std::string>& inline_order) {
  const int M = 4;
  const int N = 5;
  const int K = 6;
  BufHandle a_buf("a", {M, N}, kFloat);
  BufHandle b_buf("b", {N, K}, kFloat);
  BufHandle c_buf("c", {M, N}, kFloat);
  BufHandle d_buf("d", {M, K}, kFloat);

  Tensor x = Compute(
      "x",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) * b_buf.load(n, k);
      });
  Tensor y = Compute(
      "y",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return c_buf.load(m, n) * d_buf.load(m, k) + x.load(m, n, k);
      });
  Tensor z = Compute(
      "z",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return x.load(m, n, k) + y.load(m, n, k);
      });

  LoopNest l({z}, {x, y, z});
  for (const std::string& order : inline_order) {
    if (order == "x") {
      l.computeInline(x.buf());
    } else if (order == "y") {
      l.computeInline(y.buf());
    } else {
      throw std::runtime_error("Invalid order: " + order);
    }
  }
  l.prepareForCodegen();
  StmtPtr stmt = l.root_stmt();

  std::ostringstream oss;
  oss << *stmt;
  std::string str1 = remove_space(oss.str());

  {
    PaddedBuffer<float> a_v(M, N);
    PaddedBuffer<float> b_v(N, K);
    PaddedBuffer<float> c_v(M, N);
    PaddedBuffer<float> d_v(M, K);

    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        a_v(i, j) = i * i;
      }
    }
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < K; j++) {
        b_v(i, j) = j * j;
      }
    }
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        c_v(i, j) = i + j;
      }
    }
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < K; j++) {
        d_v(i, j) = i * j;
      }
    }

    PaddedBuffer<float> z_v(M, N, K);
    PaddedBuffer<float> z_ref(M, N, K);
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
          z_ref(m, n, k) = a_v(m, n) * b_v(n, k) * 2 + c_v(m, n) * d_v(m, k);
        }
      }
    }

    SimpleIREvaluator eval(stmt, {a_buf, b_buf, c_buf, d_buf, z});
    eval(a_v, b_v, c_v, d_v, z_v);
    ExpectAllNear(z_v, z_ref, 1e-5);
  }

  if (inline_order.size() == 2) {
    Tensor z2 = Compute(
        "z",
        {M, N, K},
        [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
          return a_buf.load(m, n) * b_buf.load(n, k) +
              (c_buf.load(m, n) * d_buf.load(m, k) +
               a_buf.load(m, n) * b_buf.load(n, k));
        });
    LoopNest l2({z2});
    l2.prepareForCodegen();
    StmtPtr stmt2 = l2.root_stmt();

    std::ostringstream oss2;
    oss2 << *stmt2;
    std::string str2 = remove_space(oss2.str());

    ASSERT_EQ(str1, str2);
    ASSERT_GT(str1.size(), 100);
  }
}

TEST(LoopNest, ScheduleInlineFunc01) {
  InlineFunc01Helper({"x", "y"});
  InlineFunc01Helper({"y", "x"});
  InlineFunc01Helper({"x"});
  InlineFunc01Helper({"y"});
  InlineFunc01Helper({});
}

// Make sure we cache random vars if we should.
TEST(LoopNest, ScheduleInlineRandom) {
  const int M = 4;
  const int N = 5;
  const int K = 6;

  Tensor x = Compute(
      "x",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return Mod::make(Intrinsics::make(kRand, kInt), 5);
      });
  Tensor y = Compute(
      "y",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return x.load(m, n, k) + x.load(m, n, k);
      });

  LoopNest l1({y}, {x, y});
  l1.computeInline(x.buf());

  // would normally compare results but Rand isn't implemented in the
  // SimpleIREvaluator, even if we could seed it.
  StmtPtr stmt1 = IRSimplifier::simplify(l1.root_stmt());

  // Check the IR we produced
  checkIR(stmt1, R"IR(
# CHECK: for (int i = 0; i < 4; i++)
# CHECK:   for (int i_1 = 0; i_1 < 5; i_1++)
# CHECK:     for (int i_2 = 0; i_2 < 6; i_2++)
# CHECK:       int x = rand();
# CHECK:       y[i, i_1, i_2] = 2 * (x % 5);)IR");
}

// Make sure we don't cache random vars that are not being inlined.
TEST(LoopNest, ScheduleInlineRandomUnrelated) {
  const int M = 4;
  const int N = 5;
  const int K = 6;

  Tensor x = Compute(
      "x",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return m * n * k;
      });
  Tensor y = Compute(
      "y",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return x.load(m, n, k) + Intrinsics::make(kRand, kInt) +
            Intrinsics::make(kRand, kInt);
      });

  LoopNest l1({y}, {x, y});
  l1.computeInline(x.buf());

  // would normally compare results but Rand isn't implemented in the
  // SimpleIREvaluator, even if we could seed it.
  StmtPtr stmt1 = IRSimplifier::simplify(l1.root_stmt());

  // Check the IR we produced
  checkIR(stmt1, R"IR(
# CHECK: for (int i = 0; i < 4; i++)
# CHECK:   for (int i_1 = 0; i_1 < 5; i_1++)
# CHECK:     for (int i_2 = 0; i_2 < 6; i_2++)
# CHECK:       y[i, i_1, i_2] = ((i * i_1) * i_2 + (rand())) + (rand());)IR");
}

// Make sure we generate the right number of random values == the dimensionality
// of the production tensor.
TEST(LoopNest, ScheduleInlineRandomLowerDimensions) {
  const int M = 4;
  const int N = 5;
  const int K = 6;

  Tensor x = Compute("x", {M}, [&](const VarHandle& m) {
    return Mod::make(Intrinsics::make(kRand, kInt), 5);
  });
  Tensor y = Compute(
      "y",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return x.load(m) + x.load(m);
      });

  LoopNest l1({y}, {x, y});
  l1.computeInline(x.buf());

  // would normally compare results but Rand isn't implemented in the
  // SimpleIREvaluator, even if we could seed it.
  StmtPtr stmt1 = IRSimplifier::simplify(l1.root_stmt());

  // Check the IR we produced
  checkIR(stmt1, R"IR(
# CHECK: for (int i = 0; i < 4; i++)
# CHECK:   int x = rand();
# CHECK:   for (int i_1 = 0; i_1 < 5; i_1++)
# CHECK:     for (int i_2 = 0; i_2 < 6; i_2++)
# CHECK:       y[i, i_1, i_2] = 2 * (x % 5);)IR");
}

// Make sure we don't screw up intrinsics thinking they're rand.
TEST(LoopNest, ScheduleInlineIntrinsics) {
  const int M = 4;
  const int N = 5;
  const int K = 6;
  BufHandle a_buf("a", {M, N}, kFloat);
  BufHandle b_buf("b", {N, K}, kFloat);

  Tensor x = Compute(
      "x",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) * b_buf.load(n, k);
      });
  Tensor y = Compute(
      "y",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return Intrinsics::make(kSqrt, x.load(m, n, k));
      });

  PaddedBuffer<float> a_v(M, N);
  PaddedBuffer<float> b_v(N, K);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      a_v(i, j) = i * i;
    }
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      b_v(i, j) = j * j;
    }
  }

  LoopNest l1({y}, {x, y});
  LoopNest l2(l1);
  l2.computeInline(x.buf());

  l1.prepareForCodegen();
  l2.prepareForCodegen();

  StmtPtr stmt1 = IRSimplifier::simplify(l1.root_stmt());
  StmtPtr stmt2 = IRSimplifier::simplify(l2.root_stmt());

  SimpleIREvaluator eval1(stmt1, {a_buf, b_buf, y});
  SimpleIREvaluator eval2(stmt2, {a_buf, b_buf, y});

  PaddedBuffer<float> y_1(M, N, K);
  PaddedBuffer<float> y_2(M, N, K);

  eval1(a_v, b_v, y_1);
  eval2(a_v, b_v, y_2);
  ExpectAllNear(y_1, y_2, 1e-5);
  std::ostringstream oss1, oss2;
  oss1 << *stmt1;
  oss2 << *stmt2;
  ASSERT_GT(oss1.str().size(), oss2.str().size());
}

// Make sure we can handle rand and non-rand intrinsics.
TEST(LoopNest, ScheduleInlineRandWithIntrinsics) {
  const int M = 4;
  const int N = 5;
  const int K = 6;

  Tensor x = Compute(
      "x",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return Intrinsics::make(kRand, kFloat);
      });
  Tensor y = Compute(
      "y",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return Intrinsics::make(kSqrt, x.load(m, n, k));
      });

  LoopNest l1({y}, {x, y});
  l1.computeInline(x.buf());

  StmtPtr stmt1 = IRSimplifier::simplify(l1.root_stmt());

  // Check the IR we produced
  checkIR(stmt1, R"IR(
# CHECK: for (int i = 0; i < 4; i++)
# CHECK:   for (int i_1 = 0; i_1 < 5; i_1++)
# CHECK:     for (int i_2 = 0; i_2 < 6; i_2++)
# CHECK:       float x = rand();
# CHECK:       y[i, i_1, i_2] = sqrt(x);)IR");
}

// Split a Compute then inline it into another compute.
TEST(LoopNest, ScheduleSplitAThenInline) {
  Tensor a = Compute("a", {18}, [&](const VarHandle& i) { return i * i; });
  Tensor b = Compute(
      "b", {2}, [&](const VarHandle& j) { return a.load(j + ExprHandle(8)); });

  LoopNest l({b}, {a, b});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(a.buf()).at(0);
  LoopNest::splitWithMask(loops[0], 4);
  ASSERT_FALSE(l.computeInline(a.buf()));
}

// Split a Compute then inline another Compute into it.
TEST(LoopNest, ScheduleSplitBThenInline) {
  Tensor a = Compute("a", {18}, [&](const VarHandle& i) { return i * i; });
  Tensor b = Compute(
      "b", {6}, [&](const VarHandle& j) { return a.load(j + ExprHandle(8)); });

  LoopNest l({b}, {a, b});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(b.buf()).at(0);
  LoopNest::splitWithMask(loops[0], 3);
  l.computeInline(a.buf());
  l.prepareForCodegen();
  StmtPtr s = IRSimplifier::simplify(l.root_stmt());

  std::vector<int> output(6, 0);
  SimpleIREvaluator eval(s, {b});
  eval(output);

  for (int i = 0; i < 6; ++i) {
    ASSERT_EQ(output[i], (i + 8) * (i + 8));
  }
}

// Split a Compute twice then inline it.
TEST(LoopNest, ScheduleSplitTwiceThenInline) {
  Tensor a = Compute("a", {18}, [&](const VarHandle& i) { return i * i; });
  Tensor b = Compute(
      "b", {2}, [&](const VarHandle& j) { return a.load(j + ExprHandle(8)); });
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr i_inner;

  LoopNest l({b}, {a, b});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(a.buf()).at(0);
  LoopNest::splitWithMask(loops[0], 4, &i_inner);
  LoopNest::splitWithMask(i_inner, 2);
  ASSERT_FALSE(l.computeInline(a.buf()));
}

// Inline a Compute, then split.
TEST(LoopNest, ScheduleInlineThenSplit) {
  Tensor a = Compute("a", {18}, [&](const VarHandle& i) { return i * i; });
  Tensor b = Compute(
      "b", {6}, [&](const VarHandle& j) { return a.load(j + ExprHandle(8)); });

  LoopNest l({b}, {a, b});
  l.computeInline(a.buf());

  std::vector<ForPtr> loops = NodeFinder<For>::find(l.root_stmt());
  LoopNest::splitWithMask(loops.back(), 3);
  l.prepareForCodegen();
  StmtPtr s = IRSimplifier::simplify(l.root_stmt());
  std::vector<int> output(6, 0);
  SimpleIREvaluator eval(s, {b});
  eval(output);

  for (int i = 0; i < 6; ++i) {
    ASSERT_EQ(output[i], (i + 8) * (i + 8));
  }
}

// Split a Compute, inline it, then split the result.
TEST(LoopNest, ScheduleSplitInlineThenSplit) {
  Tensor a = Compute("a", {18}, [&](const VarHandle& i) { return i * i; });
  Tensor b = Compute(
      "b", {16}, [&](const VarHandle& j) { return a.load(j + ExprHandle(8)); });

  LoopNest l({b}, {a, b});
  auto loops = NodeFinder<For>::find(l.root_stmt());
  LoopNest::splitWithMask(loops.back(), 2);
  l.computeInline(a.buf());

  loops = NodeFinder<For>::find(l.root_stmt());
  LoopNest::splitWithMask(loops.front(), 2);
  l.prepareForCodegen();
  StmtPtr s = IRSimplifier::simplify(l.root_stmt());
  std::vector<int> output(16, 0);
  SimpleIREvaluator eval(s, {b});
  eval(output);

  for (int i = 0; i < 16; ++i) {
    ASSERT_EQ(output[i], (i + 8) * (i + 8));
  }
}

// Oversplit a loop that is simplified out after inlining.
TEST(LoopNest, ScheduleSplitInlineSimplify) {
  Tensor a = Compute("a", {18}, [&](const VarHandle& i) {
    return ExprHandle(4) * i - ExprHandle(2) * i;
  });
  Tensor b = Compute(
      "b", {2}, [&](const VarHandle& j) { return a.load(j) - ExprHandle(1); });

  LoopNest l({b}, {a, b});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(a.buf()).at(0);
  LoopNest::splitWithMask(loops[0], 4);
  ASSERT_FALSE(l.computeInline(a.buf()));
}

// Inline a Compute with two consumers.
TEST(LoopNest, ScheduleInlineThreeMixedOnce) {
  Tensor a = Compute("a", {18}, [&](const VarHandle& i) { return i * i; });
  Tensor b = Compute(
      "b", {6}, [&](const VarHandle& j) { return a.load(j + ExprHandle(8)); });
  Tensor c = Compute("c", {4, 3}, [&](const VarHandle& k, const VarHandle& l) {
    return a.load(k) * b.load(l);
  });

  LoopNest l({c}, {a, b, c});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(a.buf()).at(0);
  l.computeInline(a.buf());
  l.prepareForCodegen();

  StmtPtr s = IRSimplifier::simplify(l.root_stmt());
  std::vector<int> output(4 * 3, 0);
  SimpleIREvaluator eval(s, {c});
  eval(output);

  for (int k = 0; k < 4; ++k) {
    for (int l = 0; l < 3; ++l) {
      ASSERT_EQ(output[k * 3 + l], (k) * (k) * (l + 8) * (l + 8));
    }
  }
}

// Inline Compute A into B, then inline B into C.
TEST(LoopNest, ScheduleInlineThreeMixedTwice) {
  Tensor a = Compute("a", {18}, [&](const VarHandle& i) { return i * i; });
  Tensor b = Compute(
      "b", {6}, [&](const VarHandle& j) { return a.load(j + ExprHandle(8)); });
  Tensor c = Compute("c", {4, 3}, [&](const VarHandle& k, const VarHandle& l) {
    return a.load(k) * b.load(l);
  });

  LoopNest l({c}, {a, b, c});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(a.buf()).at(0);
  l.computeInline(a.buf());
  l.computeInline(b.buf());
  l.prepareForCodegen();

  StmtPtr s = IRSimplifier::simplify(l.root_stmt());
  std::vector<int> output(4 * 3, 0);
  SimpleIREvaluator eval(s, {c});
  eval(output);

  for (int k = 0; k < 4; ++k) {
    for (int l = 0; l < 3; ++l) {
      ASSERT_EQ(output[k * 3 + l], (k) * (k) * (l + 8) * (l + 8));
    }
  }
}

// Inline a Compute that is both a producer and consumer.
TEST(LoopNest, ScheduleInlineThreeMixedInner) {
  Tensor a = Compute("a", {18}, [&](const VarHandle& i) { return i * i; });
  Tensor b = Compute(
      "b", {6}, [&](const VarHandle& j) { return a.load(j + ExprHandle(8)); });
  Tensor c = Compute("c", {4, 3}, [&](const VarHandle& k, const VarHandle& l) {
    return a.load(k) * b.load(l);
  });

  LoopNest l({c}, {a, b, c});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(a.buf()).at(0);
  l.computeInline(b.buf());
  l.prepareForCodegen();

  StmtPtr s = IRSimplifier::simplify(l.root_stmt());
  std::vector<int> output(4 * 3, 0);
  SimpleIREvaluator eval(s, {c});
  eval(output);

  for (int k = 0; k < 4; ++k) {
    for (int l = 0; l < 3; ++l) {
      ASSERT_EQ(output[k * 3 + l], (k) * (k) * (l + 8) * (l + 8));
    }
  }
}

// Split 3 Computes, then inline the first two into the last.
TEST(LoopNest, ScheduleInlineThreeMixedSplit) {
  Tensor a = Compute("a", {18}, [&](const VarHandle& i) { return i * i; });
  Tensor b = Compute(
      "b", {6}, [&](const VarHandle& j) { return a.load(j + ExprHandle(8)); });
  Tensor c = Compute("c", {4, 3}, [&](const VarHandle& k, const VarHandle& l) {
    return a.load(k) * b.load(l);
  });

  LoopNest l({c}, {a, b, c});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(a.buf()).at(0);
  LoopNest::splitWithMask(loops[0], 4);
  loops = l.getAllLoopNestsWritingToBuf(b.buf()).at(0);
  LoopNest::splitWithMask(loops[0], 3);
  loops = l.getAllLoopNestsWritingToBuf(c.buf()).at(0);
  LoopNest::splitWithMask(loops[0], 2);

  ASSERT_FALSE(l.computeInline(a.buf()));
}

// Check that inlining works for output tensors too
TEST(LoopNest, ScheduleInlineOutputTensors) {
  const int M = 4;
  const int N = 5;
  const int K = 6;

  Tensor x = Compute(
      "x",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return m * n * k;
      });
  Tensor y = Compute(
      "y",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return x.load(m, n, k) + m;
      });

  LoopNest l1({x, y});
  l1.computeInline(x.buf());

  // would normally compare results but Rand isn't implemented in the
  // SimpleIREvaluator, even if we could seed it.
  StmtPtr stmt1 = IRSimplifier::simplify(l1.root_stmt());

  // Check the IR we produced
  checkIR(stmt1, R"IR(
# CHECK: for (int i = 0; i < 4; i++)
# CHECK:   for (int i_1 = 0; i_1 < 5; i_1++)
# CHECK:     for (int i_2 = 0; i_2 < 6; i_2++)
# CHECK:       x[i, i_1, i_2] = (i * i_1) * i_2;
# CHECK: for (int i_3 = 0; i_3 < 4; i_3++)
# CHECK:   for (int i_4 = 0; i_4 < 5; i_4++)
# CHECK:     for (int i_5 = 0; i_5 < 6; i_5++)
# CHECK:       y[i_3, i_4, i_5] = i_3 + (i_3 * i_4) * i_5;)IR");
}

TEST(LoopNest, ScheduleInlineWithCompoundIndices) {
  // Input IR:
  //     for (int64_t i = 0; i < 100; i++) {
  //       A[i*2,i] = i * 500ll;
  //     }
  //     for (int64_t j = 0; j < 100; j++) {
  //       B[0ll,j] = A[0, j] + j * 100ll;
  //     }
  BufHandle a_buf("A", {20, 100}, kLong);
  BufHandle b_buf("B", {20, 100}, kLong);
  VarHandle i("i", kLong);
  VarHandle j("j", kLong);
  auto forI = For::make(
      i,
      0,
      100,
      Store::make(a_buf, {i * 2, i}, Mul::make(i, static_cast<int64_t>(500))));
  auto forJ = For::make(
      j,
      0,
      100,
      Store::make(
          b_buf,
          {static_cast<int64_t>(0), j},
          Add::make(
              Load::make(a_buf, {static_cast<int64_t>(0), j}),
              Mul::make(j, static_cast<int64_t>(100)))));
  auto par = Block::make({forI, forJ});

  LoopNest l(par, {b_buf.node()});
  // Inlining should fail since the producer has compound expr as index.
  ASSERT_FALSE(l.computeInline(a_buf.node()));

  // The input statement must remain as is.
  checkIR(l.root_stmt(), R"IR(
    # CHECK: for (int64_t i = 0;
    # CHECK-NEXT:   A[
    # CHECK: for (int64_t j = 0;
    # CHECK-NEXT:   B[)IR");
}

TEST(LoopNest, ScheduleInlineConsumerIndicesWithCast) {
  // Input IR:
  //     for (int64_t i = 0; i < 100; i++) {
  //       A[0ll,i] = i * 500ll;
  //     }
  //     for (int64_t j = 0; j < 100; j++) {
  //       B[0ll,j] = A[(int64_t)0, j] + j * 100ll;
  //     }
  BufHandle a_buf("A", {20, 100}, kLong);
  BufHandle b_buf("B", {20, 100}, kLong);
  VarHandle i("i", kLong);
  VarHandle j("j", kLong);
  auto forI = For::make(
      i,
      0,
      100,
      Store::make(
          a_buf,
          {static_cast<int64_t>(0), i},
          Mul::make(i, static_cast<int64_t>(500))));
  auto forJ = For::make(
      j,
      0,
      100,
      Store::make(
          b_buf,
          {static_cast<int64_t>(0), j},
          Add::make(
              Load::make(a_buf, {0, j}),
              Mul::make(j, static_cast<int64_t>(100)))));
  auto par = Block::make({forI, forJ});

  LoopNest l(par, {b_buf.node()});
  ASSERT_TRUE(l.computeInline(a_buf.node()));

  checkIR(l.root_stmt(), R"IR(
    # CHECK: for (int64_t j = 0; j < 100; j++) {
    # CHECK:   B[0ll, j] = j * 500ll + j * 100ll;
    # CHECK: })IR");
}

TEST(LoopNest, ScheduleInlineProducerIndicesWithCast) {
  // Input IR:
  //     for (int64_t i = 0; i < 100; i++) {
  //       A[(int64_t)0,i] = i * 500ll;
  //     }
  //     for (int64_t j = 0; j < 100; j++) {
  //       B[0ll,j] = A[0ll, j] + j * 100ll;
  //     }
  BufHandle a_buf("A", {20, 100}, kLong);
  BufHandle b_buf("B", {20, 100}, kLong);
  VarHandle i("i", kLong);
  VarHandle j("j", kLong);
  auto forI = For::make(
      i,
      0,
      100,
      Store::make(a_buf, {0, i}, Mul::make(i, static_cast<int64_t>(500))));
  auto forJ = For::make(
      j,
      0,
      100,
      Store::make(
          b_buf,
          {static_cast<int64_t>(0), j},
          Add::make(
              Load::make(a_buf, {static_cast<int64_t>(0), j}),
              Mul::make(j, static_cast<int64_t>(100)))));
  auto par = Block::make({forI, forJ});

  LoopNest l(par, {b_buf.node()});
  ASSERT_TRUE(l.computeInline(a_buf.node()));

  checkIR(l.root_stmt(), R"IR(
    # CHECK: for (int64_t j = 0; j < 100; j++) {
    # CHECK:   B[0ll, j] = j * 500ll + j * 100ll;
    # CHECK: })IR");
}

TEST(LoopNest, ScheduleFuserStyle) {
  const int kVectorSize = 8;
  const int kVectorCount = 128;
  const int kTotalSize = kVectorSize * kVectorCount;

  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);

  Tensor b =
      Compute("f", {kTotalSize}, [&](const std::vector<VarHandle>& axes) {
        return a_buf.load(axes[0]) + 11.0f;
      });

  Tensor c =
      Compute("g", {kTotalSize}, [&](const std::vector<VarHandle>& axes) {
        return b.load(axes[0]) + 1.0f;
      });

  LoopNest l({b, c});
  l.prepareForCodegen();
  StmtPtr s = l.root_stmt();

  std::vector<float> a_data(kTotalSize, 7.0f);
  std::vector<float> b_data(kTotalSize, 0.0f);
  std::vector<float> c_data(kTotalSize, 0.0f);
  SimpleIREvaluator(s, {a_buf, b, c})(a_data, b_data, c_data);

  for (int i = 0; i < kTotalSize; i++) {
    ASSERT_EQ(b_data[i], 18.0f);
    ASSERT_EQ(c_data[i], 19.0f);
  }
}

TEST(LoopNest, ScheduleFuserThreeArg) {
  const int kVectorSize = 8;
  const int kVectorCount = 128;
  const int kTotalSize = kVectorSize * kVectorCount;

  BufHandle a("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b("B", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle c("C", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle d("D", {ExprHandle(kTotalSize)}, kFloat);

  Tensor e = Compute("e", {kTotalSize}, [&](const VarHandle& i) {
    return a.load(i) + b.load(i);
  });
  Tensor f = Compute("f", {kTotalSize}, [&](const VarHandle& i) {
    return e.load(i) + c.load(i);
  });
  Tensor g = Compute("g", {kTotalSize}, [&](const VarHandle& i) {
    return f.load(i) + d.load(i);
  });

  LoopNest l({g}, {e, f, g});
  l.computeInline(l.getLoopBodyFor(e));
  l.computeInline(l.getLoopBodyFor(f));
  l.prepareForCodegen();
  StmtPtr s = l.root_stmt();

  std::vector<float> a_data(kTotalSize, 1.0f);
  std::vector<float> b_data(kTotalSize, 2.0f);
  std::vector<float> c_data(kTotalSize, 3.0f);
  std::vector<float> d_data(kTotalSize, 4.0f);
  std::vector<float> g_data(kTotalSize, 0.0f);
  SimpleIREvaluator(s, {a, b, c, d, g})(a_data, b_data, c_data, d_data, g_data);

  for (int i = 0; i < kTotalSize; i++) {
    ASSERT_EQ(g_data[i], 10.0f);
  }
}

TEST(LoopNest, ScheduleDynamicShape2D) {
  auto testWithSize = [](int32_t M, int32_t N) {
    VarHandle m("m", kInt);
    VarHandle n("n", kInt);
    BufHandle a("a", {m, n}, kFloat);
    BufHandle b("b", {m, n}, kFloat);
    Tensor c =
        Compute("c", {m, n}, [&](const VarHandle& i, const VarHandle& j) {
          return a.load(i, j) + b.load(i, j);
        });
    LoopNest l({c});
    StmtPtr s = l.root_stmt();
    SimpleIREvaluator cg(s, {a, b, c, m, n});
    std::vector<float> aData(M * N, 1.0f);
    std::vector<float> bData(M * N, 2.0f);
    std::vector<float> cData(M * N, 0.0f);
    cg.call({aData, bData, cData, M, N});
    ExpectAllNear(cData, std::vector<float>(M * N, 3.0f), 1e-7);
  };
  testWithSize(1, 8);
  testWithSize(16, 32);
  testWithSize(37, 11);
}

TEST(LoopNest, LoopNestComputeAt_1) {
  // Verify that compute_at works on the following example:
  //
  // for (int i_a = 0; i_a < N; i_a++) {
  //   A[i_a] = i_a * i_a
  // }
  // for (int i_b = 0; i_b < N; i_b++) {
  //   B[i_b] = A[i_b]
  // }
  //
  // After the transformation the i_b loop should have an allocation for a temp
  // buffer and that buffer should be used in computation of B. No use of A
  // should be in that loop after the transformation. Also, computation of A
  // should not be inlined into B. Instead, it should be computed into the temp,
  // and the temp should be used in B.
  VarHandle N("N", kInt);
  Tensor A = Compute("A", {N}, [&](const VarHandle& i_a) { return i_a * i_a; });
  Tensor B =
      Compute("B", {N}, [&](const VarHandle& i_b) { return A.load(i_b); });
  LoopNest l({B}, {A, B});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(B.buf()).at(0);
  LoopNest::computeAt(l.getLoopBodyFor(A), loops[0]);
  l.prepareForCodegen();
  SimpleIREvaluator cg(l.root_stmt(), {B, N});
  StmtPtr s = cg.stmt();

  checkIR(s, R"IR(
# CHECK: Allocate(temp); // dtype=int, dims=[1]
# CHECK: for (int i = 0; i < N; i++)
# CHECK:   temp[
# CHECK-NOT: A[
# CHECK:   B[i_1] = temp[0]
# CHECK:   Free(temp))IR");

  // Now check that the loop still produces the correct result.
  std::vector<int> b_data(100, 0);
  cg.call({b_data, 100});

  std::vector<int> b_ref(100, 0);
  for (int i = 0; i < 100; i++) {
    b_ref[i] = i * i;
  }
  assertAllEqual(b_data, b_ref);
}

TEST(LoopNest, LoopNestComputeAt_2) {
  // Verify that compute_at works on the following example:
  //
  // for (int py = 0; py < H+1; py++) {
  //   for (int px = 0; px < W+1; px++) {
  //     p[py, px] = py*px
  //   }
  // }
  // for (int cy = 0; cy < H; cy++) {
  //   for (int cx = 0; cx < W; cx++) {
  //     c[py, px] = p[cy,cx]   + p[cy+1,cx] +
  //                 p[cy,cx+1] + p[cy+1,cx+1]
  //   }
  // }

  const int kW = 16, kH = 16;
  VarHandle W("W", kInt);
  VarHandle H("H", kInt);
  Tensor p = Compute(
      "prod", {H + 1, W + 1}, [&](const VarHandle& py, const VarHandle& px) {
        return px * py;
      });
  Tensor c =
      Compute("cons", {H, W}, [&](const VarHandle& y, const VarHandle& x) {
        return p.load(y, x) + p.load(y + 1, x) + p.load(y, x + 1) +
            p.load(y + 1, x + 1);
      });

  std::vector<int> c_ref(kW * kH, 0);
  for (int y = 0; y < kH; y++) {
    for (int x = 0; x < kW; x++) {
      c_ref[y * kW + x] = y * x + (y + 1) * x + y * (x + 1) + (y + 1) * (x + 1);
    }
  }
  LoopNest orig_loopnest({c}, {p, c});

  {
    // First let's try to compute P at axis cy (the outer loop)
    LoopNest l(orig_loopnest);
    std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(c.buf()).at(0);
    LoopNest::computeAt(l.getLoopBodyFor(p), loops[0]);
    l.prepareForCodegen();
    SimpleIREvaluator cg(l.root_stmt(), {c, W, H});
    StmtPtr s = cg.stmt();

    // Check the IR we produced
    checkIR(s, R"IR(
# CHECK: Allocate(temp); // dtype=int, dims=[2, W + 1]
# CHECK: for (int i_2 = 0; i_2 < H; i_2++)
# CHECK:   for
# CHECK:     for
# CHECK:   for (int i_3 = 0; i_3 < W; i_3++)
# CHECK-NOT: prod[
# CHECK:     cons[
# CHECK: Free(temp))IR");

    // Now check that the loop still produces the correct result.
    std::vector<int> c_data(kW * kH, 0);
    cg.call({c_data, kW, kH});

    assertAllEqual(c_data, c_ref);
  }
  {
    // Now let's try to compute P at axis cx (the inner loop)
    LoopNest l(orig_loopnest);
    std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(c.buf()).at(0);
    LoopNest::computeAt(l.getLoopBodyFor(p), loops[1]);
    l.prepareForCodegen();
    SimpleIREvaluator cg(l.root_stmt(), {c, W, H});
    StmtPtr s = cg.stmt();

    // Check the IR we produced
    checkIR(s, R"IR(
# CHECK: Allocate(temp); // dtype=int, dims=[2, 2]
# CHECK: for (int i_2 = 0; i_2 < H; i_2++)
# CHECK:   for (int i_3 = 0; i_3 < W; i_3++)
# CHECK:     for
# CHECK:       for
# CHECK-NOT: prod[
# CHECK:     cons[
# CHECK: Free(temp))IR");

    // Now check that the loop still produces the correct result.
    std::vector<int> c_data(kW * kH, 0);
    cg.call({c_data, kW, kH});

    assertAllEqual(c_data, c_ref);
  }
}

TEST(LoopNest, LoopNestComputeAt_3) {
  // Verify that compute_at works on the following example:
  //
  // A(x,y) = x*y
  // B(x,y) = A(x, y)
  // C(x,y) = B(x+1, y)
  // D(x,y) = A(x, y+1) + C(x, y)
  //
  // i.e. when 'A' comes to 'D' directly and indirectly through 'C'.

  const int kW = 16, kH = 16;
  VarHandle W("W", kInt);
  VarHandle H("H", kInt);
  Tensor A = Compute(
      "A", {H + 1, W + 1}, [&](const VarHandle& ay, const VarHandle& ax) {
        return ax * ay;
      });
  Tensor B = Compute(
      "B", {H + 1, W + 1}, [&](const VarHandle& by, const VarHandle& bx) {
        return A.load(by, bx);
      });
  Tensor C =
      Compute("C", {H, W}, [&](const VarHandle& cy, const VarHandle& cx) {
        return B.load(cy, cx + 1);
      });
  Tensor D =
      Compute("D", {H, W}, [&](const VarHandle& dy, const VarHandle& dx) {
        return A.load(dy + 1, dx) + C.load(dy, dx);
      });

  std::vector<int> c_ref(kW * kH, 0);
  for (int y = 0; y < kH; y++) {
    for (int x = 0; x < kW; x++) {
      c_ref[y * kW + x] = (y + 1) * x + y * (x + 1);
    }
  }

  LoopNest orig_loopnest({D}, {A, B, C, D});
  {
    // First let's try to compute A at axis dy (the outer loop)
    LoopNest l(orig_loopnest);
    std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(D.buf()).at(0);
    LoopNest::computeAt(l.getLoopBodyFor(A), loops[0]);
    l.prepareForCodegen();
    SimpleIREvaluator cg(l.root_stmt(), {D, W, H});
    StmtPtr s = cg.stmt();

    // Check the IR we produced
    checkIR(s, R"IR(
# CHECK: Allocate(temp); // dtype=int, dims=[1, W]
# CHECK: for (int i = 0; i < H + 1; i++)
# CHECK:   for (int i_1 = 0; i_1 < W + 1; i_1++)
# CHECK:     A[
# CHECK: for (int i_2 = 0; i_2 < H + 1; i_2++)
# CHECK:   for (int i_3 = 0; i_3 < W + 1; i_3++)
# CHECK:     B[
# CHECK: for (int i_4 = 0; i_4 < H; i_4++)
# CHECK:   for (int i_5 = 0; i_5 < W; i_5++)
# CHECK:     C[
# CHECK: for (int i_6 = 0; i_6 < H; i_6++)
# CHECK:   for (int i_7 = 0; i_7 < W; i_7++)
# CHECK-NOT: A[)IR");

    // Now check that the loop still produces the correct result.
    std::vector<int> c_data(kW * kH, 0);
    cg.call({c_data, kW, kH});

    assertAllEqual(c_data, c_ref);
  }
  {
    // Now let's try to compute A at axis dx (the inner loop)
    LoopNest l(orig_loopnest);
    std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(D.buf()).at(0);
    LoopNest::computeAt(l.getLoopBodyFor(A), loops[1]);
    l.prepareForCodegen();
    SimpleIREvaluator cg(l.root_stmt(), {D, W, H});
    StmtPtr s = cg.stmt();

    // Check the IR we produced
    checkIR(s, R"IR(
# CHECK: Allocate(temp); // dtype=int, dims=[1, 1]
# CHECK: for (int i = 0; i < H + 1; i++)
# CHECK:   for (int i_1 = 0; i_1 < W + 1; i_1++)
# CHECK:     A[
# CHECK: for (int i_2 = 0; i_2 < H + 1; i_2++)
# CHECK:   for (int i_3 = 0; i_3 < W + 1; i_3++)
# CHECK:     B[
# CHECK: for (int i_4 = 0; i_4 < H; i_4++)
# CHECK:   for (int i_5 = 0; i_5 < W; i_5++)
# CHECK:     C[
# CHECK: for (int i_6 = 0; i_6 < H; i_6++)
# CHECK:   for (int i_7 = 0; i_7 < W; i_7++)
# CHECK-NOT: A[)IR");

    // Now check that the loop still produces the correct result.
    std::vector<int> c_data(kW * kH, 0);
    cg.call({c_data, kW, kH});

    assertAllEqual(c_data, c_ref);
  }
}

using Axis = const VarHandle&;

TEST(LoopNest, Reduce2dComputeAt) {
  const int kW = 16, kH = 16;
  VarHandle W("W", kInt);
  VarHandle H("H", kInt);

  Tensor p = Compute(
      "prod", {H + 1, W + 1}, [&](Axis py, Axis px) { return px * py; });
  Tensor c = Reduce(
      "cons",
      {H, W},
      Sum(),
      [&](Axis y, Axis x, Axis r, Axis s) { return p.load(y + r, x + s); },
      {2, 2});

  std::vector<int> c_ref(kW * kH, 0);
  for (int y = 0; y < kH; y++) {
    for (int x = 0; x < kW; x++) {
      c_ref[y * kW + x] = y * x + (y + 1) * x + y * (x + 1) + (y + 1) * (x + 1);
    }
  }
  LoopNest orig_loopnest({c}, {p, c});
  checkIR(orig_loopnest.root_stmt(), R"IR(
# CHECK: for (int i = 0; i < H + 1; i++) {
# CHECK:   for (int i_1 = 0; i_1 < W + 1; i_1++) {
# CHECK:     prod[i, i_1] = i_1 * i;
# CHECK:   }
# CHECK: }
# CHECK: for (int i_2 = 0; i_2 < H; i_2++) {
# CHECK:   for (int i_3 = 0; i_3 < W; i_3++) {
# CHECK:     cons[i_2, i_3] = int(0);
# CHECK:     for (int i_4 = 0; i_4 < 2; i_4++) {
# CHECK:       for (int i_5 = 0; i_5 < 2; i_5++) {
# CHECK:         cons[i_2, i_3] = ReduceOp((cons[i_2, i_3]) + (prod[i_2 + i_4, i_3 + i_5]), reduce_args={i_4, i_5});
# CHECK:       }
# CHECK:     }
# CHECK:   }
# CHECK: }
)IR");

  {
    // First let's try to compute P at axis cy (the outer loop)
    LoopNest l(orig_loopnest);
    auto loops = l.getAllLoopNestsWritingToBuf(c.buf()).at(0);
    LoopNest::computeAt(l.getLoopBodyFor(p), loops[0]);
    // FIXME: Calling simplify here breaks the IR:
    // MALFORMED INPUT: could not find base node in Load - temp[...]
    // l.simplify();
    l.eliminateDeadStores();
    l.prepareForCodegen();
    SimpleIREvaluator cg(l.root_stmt(), {c, W, H});
    checkIR(cg.stmt(), R"IR(
# CHECK: Allocate(temp); // dtype=int, dims=[2, W + 1]
# CHECK: for (int i = 0; i < H; i++) {
# CHECK:   for (int idx0 = 0; idx0 < 2; idx0++) {
# CHECK:     for (int idx1 = 0; idx1 < W + 1; idx1++) {
# CHECK:       temp[(0 + idx0 * (1 * (W + 1))) + idx1 * 1] = (idx0 + i) * (idx1 + 0);
# CHECK:     }
# CHECK:   }
# CHECK:   for (int i_1 = 0; i_1 < W; i_1++) {
# CHECK:     cons[(0 + i * (1 * W)) + i_1 * 1] = int(0);
# CHECK:     for (int i_2 = 0; i_2 < 2; i_2++) {
# CHECK:       for (int i_3 = 0; i_3 < 2; i_3++) {
# CHECK:         cons[(0 + i * (1 * W)) + i_1 * 1] = (cons[(0 + i * (1 * W)) + i_1 * 1]) + (temp[(0 + i_2 * (1 * (W + 1))) + (i_1 + i_3) * 1]);
# CHECK:       }
# CHECK:     }
# CHECK:   }
# CHECK: }
# CHECK: Free(temp);
)IR");

    // Now check that the loop still produces the correct result.
    std::vector<int> c_data(kW * kH, 0);
    cg.call({c_data, kW, kH});
    assertAllEqual(c_data, c_ref);
  }
  {
    // Now let's try to compute P at axis cx (the inner loop)
    LoopNest l(orig_loopnest);
    std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(c.buf()).at(0);
    LoopNest::computeAt(l.getLoopBodyFor(p), loops[1]);
    l.simplify();
    l.eliminateDeadStores();
    l.prepareForCodegen();
    SimpleIREvaluator cg(l.root_stmt(), {c, W, H});
    checkIR(cg.stmt(), R"IR(
# CHECK: Allocate(temp); // dtype=int, dims=[2, 2]
# CHECK: for (int i = 0; i < H; i++) {
# CHECK:   for (int i_1 = 0; i_1 < W; i_1++) {
# CHECK:     for (int idx0 = 0; idx0 < 2; idx0++) {
# CHECK:       for (int idx1 = 0; idx1 < 2; idx1++) {
# CHECK:         temp[(0 + idx0 * (1 * 2)) + idx1 * 1] = (i + idx0) * (i_1 + idx1);
# CHECK:       }
# CHECK:     }
# CHECK:     cons[(0 + i * (1 * W)) + i_1 * 1] = 0;
# CHECK:     for (int i_2 = 0; i_2 < 2; i_2++) {
# CHECK:       for (int i_3 = 0; i_3 < 2; i_3++) {
# CHECK:         cons[(0 + i * (1 * W)) + i_1 * 1] = (cons[(0 + i * (1 * W)) + i_1 * 1]) + (temp[(0 + i_2 * (1 * 2)) + i_3 * 1]);
# CHECK:       }
# CHECK:     }
# CHECK:   }
# CHECK: }
# CHECK: Free(temp);
)IR");

    // Now check that the loop still produces the correct result.
    std::vector<int> c_data(kW * kH, 0);
    cg.call({c_data, kW, kH});
    assertAllEqual(c_data, c_ref);
  }
}

TEST(LoopNest, DISABLED_Conv1d_NH) {
  // Lots of stuff is broken here.  The computeAt swaps the axes for some odd
  // reason.  Even without that, the index flattener fails due to "dimensions
  // mismatch in flatten index".

  int N = 4;
  int H = 256;
  int R = 3;
  int Pad = 1;
  BufHandle IP("input", {H}, kFloat);

  Tensor A = Compute("A", {N, H + 2 * Pad}, [&](Axis n, Axis h) {
    auto cond = CompareSelect::make(h, Pad, 1, 0, kLT);
    cond = CompareSelect::make(h, H + Pad, 1, cond, kGE);
    return ifThenElse(cond, 0.f, IP.load(n, h - Pad));
  });
  Tensor B = Reduce(
      "B",
      {N, H},
      Sum(),
      [&](Axis n, Axis h, Axis r) { return A.load(n, h + r); },
      {R});
  LoopNest l({B});
  checkIR(l.root_stmt(), R"IR(
# CHECK: for (int np = 0; np < 4; np++) {
# CHECK:   for (int hp = 0; hp < 258; hp++) {
# CHECK:     A[np, hp] = IfThenElse(hp>=257 ? 1 : (hp<1 ? 1 : 0), 0.f, input[np, hp - 1]);
# CHECK:   }
# CHECK: }
# CHECK: for (int n = 0; n < 4; n++) {
# CHECK:   for (int h = 0; h < 256; h++) {
# CHECK:     B[n, h] = float(0);
# CHECK:     for (int r = 0; r < 3; r++) {
# CHECK:       B[n, h] = ReduceOp((B[n, h]) + (A(n, h + r)), reduce_args={r});
# CHECK:     }
# CHECK:   }
# CHECK: }
)IR");
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(B.buf()).at(0);
  LoopNest::computeAt(l.getLoopBodyFor(A), loops[0]);
  // FIXME: The current IR is totally broken.  The body of the inlined loop is:

  // temp[idx0, idx1] = IfThenElse(idx0 + n>=257 ? 1 : (idx0 + n<1 ? 1 : 0),
  // 0.f, input[idx1 + 0, (idx0 + n) - 1]);

  // Which seems to mix up the axes.  The CHECK below is my best guess at what
  // the input "should" look like

  checkIR(l.root_stmt(), R"IR(
# CHECK: for (int n = 0; n < 4; n++) {
# CHECK:   for (int idx0 = 0; idx0 < 1; idx0++) {
# CHECK:     for (int idx1 = 0; idx1 < 258; idx1++) {
        temp[idx0, idx1] = IfThenElse(idx1>=257 ? 1 : (idx1<1 ? 1 : 0), 0.f, input[n, idx1 - 1]);
# CHECK:     }
# CHECK:   }
# CHECK:   for (int h = 0; h < 256; h++) {
# CHECK:     B[n, h] = float(0);
# CHECK:     for (int r = 0; r < 3; r++) {
# CHECK:       B[n, h] = ReduceOp((B[n, h]) + (temp[0, r + h]), reduce_args={r});
# CHECK:     }
# CHECK:   }
# CHECK: }
)IR");

  l.simplify();
  l.prepareForCodegen();
  StmtPtr s = l.root_stmt();

  SimpleIREvaluator cg(s, {IP, B});
  // auto At = at::ones({N, H}, at::kFloat);
  auto At = at::arange(N * H, at::kFloat).reshape({N, H});
  auto Rt = at::conv1d(
      At, at::ones({1, 1, 3}), at::Tensor(), /*stride=*/1, /*padding=*/3);
  auto Bt = at::empty_like(Rt);
  cg.call({At.data_ptr<float>(), Bt.data_ptr<float>()});
  ASSERT_TRUE(at::allclose(Rt, Bt));
}

class LoopOrderHelper : public IRVisitor {
  std::stringstream ordering;

 public:
  std::string getOrder(StmtPtr s) {
    ordering.str("");
    s->accept(this);
    return ordering.str();
  }

  void visit(ForPtr v) final {
    ordering << v->var()->name_hint() << ",";
    IRVisitor::visit(v);
  }
};

TEST(LoopNest, LoopNestReorderAxis1) {
  Tensor tensor =
      Compute("f", {2, 3}, [](const VarHandle& x, const VarHandle& y) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  LoopNest l({tensor});
  StmtPtr stmt1 = LoopNest::sanitizeNames(Stmt::clone(l.root_stmt()));

  std::vector<int> stmt1_output(6, 0);
  SimpleIREvaluator cg(stmt1, {tensor});
  cg.call({stmt1_output});

  auto loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  LoopNest::reorderAxis(loops[0], loops[1]);
  StmtPtr stmt2 = LoopNest::sanitizeNames(Stmt::clone(l.root_stmt()));

  ASSERT_NE(stmt1, stmt2);
  LoopOrderHelper loopOrderHelper;
  std::string order1 = loopOrderHelper.getOrder(stmt1);
  std::string order2 = loopOrderHelper.getOrder(stmt2);

  ASSERT_EQ(order1, "j,i,");
  ASSERT_EQ(order2, "i,j,");

  std::vector<int> stmt2_output(6, 0);
  SimpleIREvaluator cg2(stmt2, {tensor});
  cg.call({stmt2_output});

  for (int i = 0; i < 6; ++i) {
    ASSERT_EQ(stmt1_output[i], stmt2_output[i]);
  }

  // Reorder them back.
  loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  LoopNest::reorderAxis(loops[0], loops[1]);
  StmtPtr stmt3 = l.root_stmt();

  std::string order3 = loopOrderHelper.getOrder(stmt3);
  ASSERT_EQ(order3, order1);

  std::ostringstream oss1, oss2;
  oss1 << *stmt1;
  oss2 << *stmt3;

  // Should be identical to the unreordered statement.
  ASSERT_EQ(oss1.str(), oss2.str());
}

TEST(LoopNest, LoopNestReorderPartialAxes) {
  Tensor tensor = Compute(
      "f",
      {2, 3, 4},
      [](const VarHandle& x, const VarHandle& y, const VarHandle& z) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y +
            cast<float>(z) * z;
      });
  LoopNest l({tensor});

  LoopOrderHelper loopOrderHelper;
  StmtPtr stmt1 = LoopNest::sanitizeNames(Stmt::clone(l.root_stmt()));
  ASSERT_EQ(loopOrderHelper.getOrder(stmt1), "i,j,k,");

  std::vector<int> stmt1_output(24, 0);
  SimpleIREvaluator cg(stmt1, {tensor});
  cg.call({stmt1_output});

  auto loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  LoopNest::reorderAxis(loops[0], loops[1]);
  ASSERT_EQ(loopOrderHelper.getOrder(l.root_stmt()), "j,i,k,");

  StmtPtr stmt2 = Stmt::clone(l.root_stmt());

  std::vector<int> stmt2_output(24, 0);
  SimpleIREvaluator cg2(stmt2, {tensor});
  cg2.call({stmt2_output});

  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(stmt1_output[i], stmt2_output[i]);
  }

  loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  LoopNest::reorderAxis(loops[1], loops[2]);
  ASSERT_EQ(loopOrderHelper.getOrder(l.root_stmt()), "j,k,i,");

  StmtPtr stmt3 = Stmt::clone(l.root_stmt());

  std::vector<int> stmt3_output(24, 0);
  SimpleIREvaluator cg3(stmt3, {tensor});
  cg3.call({stmt3_output});

  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(stmt1_output[i], stmt3_output[i]);
  }
}

TEST(LoopNest, LoopNestReorderInternalAxis) {
  Tensor tensor = Compute(
      "f",
      {1, 2, 3, 4},
      [](const VarHandle& w,
         const VarHandle& x,
         const VarHandle& y,
         const VarHandle& z) {
        return ExprHandle(1.0f) + w + cast<float>(x) * x + cast<float>(y) * y +
            cast<float>(z) * z;
      });
  LoopNest l({tensor});

  LoopOrderHelper loopOrderHelper;
  StmtPtr stmt1 = LoopNest::sanitizeNames(Stmt::clone(l.root_stmt()));
  ASSERT_EQ(loopOrderHelper.getOrder(stmt1), "i,j,k,l,");

  std::vector<int> stmt1_output(24, 0);
  SimpleIREvaluator cg(stmt1, {tensor});
  cg.call({stmt1_output});

  auto loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  LoopNest::reorderAxis(loops[2], loops[1]);
  ASSERT_EQ(loopOrderHelper.getOrder(l.root_stmt()), "i,k,j,l,");

  StmtPtr stmt2 = l.root_stmt();

  std::vector<int> stmt2_output(24, 0);
  SimpleIREvaluator cg2(stmt2, {tensor});
  cg2.call({stmt2_output});

  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(stmt1_output[i], stmt2_output[i]);
  }
}

TEST(LoopNest, LoopNestReorderEnclosingAxis) {
  Tensor tensor = Compute(
      "f",
      {1, 2, 3, 4},
      [](const VarHandle& w,
         const VarHandle& x,
         const VarHandle& y,
         const VarHandle& z) {
        return ExprHandle(1.0f) + w + cast<float>(x) * x + cast<float>(y) * y +
            cast<float>(z) * z;
      });
  LoopNest l({tensor});

  LoopOrderHelper loopOrderHelper;
  StmtPtr stmt1 = LoopNest::sanitizeNames(Stmt::clone(l.root_stmt()));

  std::vector<int> stmt1_output(24, 0);
  SimpleIREvaluator cg(stmt1, {tensor});
  cg.call({stmt1_output});

  auto loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  LoopNest::reorderAxis(loops[0], loops[3]);
  ASSERT_EQ(loopOrderHelper.getOrder(l.root_stmt()), "l,j,k,i,");

  StmtPtr stmt2 = l.root_stmt();

  std::vector<int> stmt2_output(24, 0);
  SimpleIREvaluator cg2(stmt2, {tensor});
  cg2.call({stmt2_output});

  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(stmt1_output[i], stmt2_output[i]);
  }
}

TEST(LoopNest, LoopNestReorderSameAxis) {
  Tensor tensor =
      Compute("f", {2, 3}, [](const VarHandle& x, const VarHandle& y) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  LoopNest l({tensor});
  StmtPtr stmt1 = Stmt::clone(l.root_stmt());

  auto loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  LoopNest::reorderAxis(loops[1], loops[1]);
  StmtPtr stmt2 = Stmt::clone(l.root_stmt());

  std::ostringstream oss, oss2;
  oss << *stmt1;
  oss2 << *stmt2;
  ASSERT_EQ(oss.str(), oss2.str());
}

TEST(LoopNest, LoopNestReorderExtraStatements) {
  /* We're going for a structure like this:
   * for i in ...
   *   Stmt 1
   *   for j in ...
   *     Stmt 2
   *     for k in ...
   *       Stmt 3
   *     Stmt 4
   */

  Tensor tensor = Compute(
      "f",
      {2, 3, 4},
      [](const VarHandle& x, const VarHandle& y, const VarHandle& z) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y +
            cast<float>(z) * z;
      });
  LoopNest l({tensor});

  BufHandle extra("res", {6, 3}, kFloat);

  auto loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);

  VarHandle i = VarHandle(loops[0]->var());

  StmtPtr store_1 = Store::make(extra, {i, 0}, 1.f);
  StmtPtr store_2 = Store::make(extra, {i, 1}, 2.f);
  // stmt 3 is the Function body.
  StmtPtr store_3 = Store::make(extra, {i, 2}, 4.f);

  loops[0]->body()->prepend_stmt(store_1);
  loops[1]->body()->prepend_stmt(store_2);
  loops[1]->body()->append_stmt(store_3);
  StmtPtr stmt1 = LoopNest::sanitizeNames(Stmt::clone(l.root_stmt()));

  std::vector<int> extra1(6, 0);
  std::vector<int> res1(24, 0);
  SimpleIREvaluator cg(stmt1, {tensor, extra});
  cg.call({res1, extra1});

  /* Then we reorder loop y and z, we want it to look like:
   *
   * for i in ...
   *   Stmt 1
   *   for j in ...
   *     Stmt 2
   *   for j_1 in ...
   *    for k in ...
   *       Stmt 3
   *   for j_2 in ...
   *     Stmt 4
   *
   * We need extra loops because we don't have dependency info about stmt 3
   * and 4.
   *
   */

  LoopNest::reorderAxis(loops[1], loops[2]);
  StmtPtr stmt2 = LoopNest::sanitizeNames(Stmt::clone(l.root_stmt()));

  // Check the IR we produced
  checkIR(stmt2, R"IR(
# CHECK: for
# CHECK:   res[i, 0] = 1
# CHECK:   for
# CHECK:     res[i, 1] = 2
# CHECK:   for
# CHECK:     for
# CHECK:       f[
# CHECK:   for
# CHECK:     res[i, 2] = 4
)IR");

  std::vector<int> extra2(6, 0);
  std::vector<int> res2(24, 0);
  SimpleIREvaluator cg2(stmt2, {tensor, extra});
  cg2.call({res2, extra2});

  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(res1[i], res2[i]);
  }
  for (int i = 0; i < 6; ++i) {
    ASSERT_EQ(extra1[i], extra2[i]);
  }

  /* Now reorder x and the y above stmt 3:
   *
   *
   * for x in ...
   *   Stmt 1
   *   for y in ...
   *     Stmt 2
   *
   * for y in ...
   *   for z in ...
   *    for x in ...
   *       Stmt 3
   *
   * for x in ...
   *   for y in ...
   *     Stmt 4
   *
   *
   */
  loops = l.getAllLoopNestsWritingToBuf(tensor.buf()).at(0);
  LoopNest::reorderAxis(loops[0], loops[2]);
  StmtPtr stmt3 = LoopNest::sanitizeNames(Stmt::clone(l.root_stmt()));

  // Check the IR we produced
  checkIR(stmt3, R"IR(
# CHECK: for
# CHECK:   res[i, 0] = 1
# CHECK:   for
# CHECK:     res[i, 1] = 2
# CHECK: for
# CHECK:   for
# CHECK:     for
# CHECK:       f[
# CHECK: for
# CHECK:   for
# CHECK:     res[i_2, 2] = 4
)IR");

  std::vector<int> extra3(6, 0);
  std::vector<int> res3(24, 0);
  SimpleIREvaluator cg3(stmt3, {tensor, extra});
  cg3.call({res3, extra3});

  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(res1[i], res3[i]);
  }
  for (int i = 0; i < 6; ++i) {
    ASSERT_EQ(extra1[i], extra3[i]);
  }
}

void LoopNestReorderTestHelper(
    bool prepend,
    bool append,
    int index1,
    int index2) {
  Tensor c = Compute(
      "5d", {2, 3, 2, 3, 2}, [](const std::vector<VarHandle>&) { return -1; });
  LoopNest l({c});

  BufHandle extra("extra", {5}, kInt);

  auto loops = l.getAllLoopNestsWritingToBuf(c.buf()).at(0);
  int j = 0;
  for (auto l : loops) {
    // Add an increment at each layer of the loop which counts the number of
    // times the loop executes.
    LoadPtr load =
        alloc<Load>(extra.node(), std::vector<ExprPtr>({alloc<IntImm>(j)}));
    AddPtr add = alloc<Add>(load, alloc<IntImm>(1));
    StmtPtr store = alloc<Store>(
        extra.node(), std::vector<ExprPtr>({alloc<IntImm>(j)}), add);
    if (prepend) {
      l->body()->prepend_stmt(store);
    }
    if (append) {
      l->body()->append_stmt(Stmt::clone(store));
    }

    j++;
  }

  StmtPtr stmt1 = Stmt::clone(l.root_stmt());

  std::vector<int> extra1(5, 0);
  std::vector<int> res1(2 * 3 * 2 * 3 * 2, 0);
  SimpleIREvaluator cg(stmt1, {c, extra});
  cg.call({res1, extra1});

  std::vector<int> loopExtents = {2, 3, 2, 3, 2};

  int expected_loops = 0;
  if (prepend) {
    expected_loops++;
  }
  if (append) {
    expected_loops++;
  }
  for (int i = 0; i < 5; ++i) {
    expected_loops *= loopExtents[i];
    ASSERT_EQ(extra1[i], expected_loops);
  }

  loops = l.getAllLoopNestsWritingToBuf(c.buf()).at(0);
  LoopNest::reorderAxis(loops[index1], loops[index2]);
  StmtPtr stmt2 = Stmt::clone(l.root_stmt());

  std::ostringstream oss, oss2;
  oss << *stmt1;
  oss2 << *stmt2;
  ASSERT_NE(oss.str(), oss2.str());

  std::vector<int> extra2(5, 0);
  std::vector<int> res2(2 * 3 * 2 * 3 * 2, 0);
  SimpleIREvaluator cg2(stmt2, {c, extra});
  cg2.call({res2, extra2});

  expected_loops = 0;
  if (prepend) {
    expected_loops++;
  }
  if (append) {
    expected_loops++;
  }

  for (int i = 0; i < 5; ++i) {
    expected_loops *= loopExtents[i];
    ASSERT_EQ(extra2[i], expected_loops);
  }

  for (int i = 0; i < 2 * 3 * 2 * 3 * 2; ++i) {
    ASSERT_EQ(res2[i], res1[i]);
  }
}

TEST(LoopNest, LoopNestReorderLongStringOfPreOrphans) {
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      // skip noops, since we check the loop isn't the same after reordering.
      if (i != j) {
        LoopNestReorderTestHelper(true, false, i, j);
      }
    }
  }
}

TEST(LoopNest, LoopNestReorderLongStringOfPostOrphans) {
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      // skip noops, since we check the loop isn't the same after reordering.
      if (i != j) {
        LoopNestReorderTestHelper(false, true, i, j);
      }
    }
  }
}

TEST(LoopNest, LoopNestReorderLongStringFull) {
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      // skip noops, since we check the loop isn't the same after reordering.
      if (i != j) {
        LoopNestReorderTestHelper(true, true, i, j);
      }
    }
  }
}

TEST(LoopNest, LoopNestReorderInternalLoopNest) {
  const int M = 4;
  const int N = 5;
  const int K = 6;
  BufHandle a_buf("a", {M, N}, kFloat);
  BufHandle b_buf("b", {N, K}, kFloat);
  BufHandle c_buf("c", {M, N}, kFloat);
  BufHandle d_buf("d", {M, K}, kFloat);

  Tensor x = Compute(
      "x",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) * b_buf.load(n, k);
      });
  Tensor y = Compute(
      "y",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return c_buf.load(m, n) * d_buf.load(m, k) + x.load(m, n, k);
      });
  Tensor z = Compute(
      "z",
      {M, N, K},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return x.load(m, n, k) + y.load(m, n, k);
      });

  LoopNest l({z}, {x, y, z});
  ForPtr a = l.getAllLoopNestsWritingToBuf(y.buf())[0][2];
  ForPtr b = l.getAllLoopNestsWritingToBuf(y.buf())[0][0];
  LoopNest::reorderAxis(a, b);

  l.prepareForCodegen();
  StmtPtr stmt = IRSimplifier::simplify(l.root_stmt());

  // Check the IR we produced has the 3 nests in the right order, but k and m
  // swapped in the middle.
  checkIR(stmt, R"IR(
# CHECK: < 4
# CHECK: < 5
# CHECK: < 6
# CHECK: < 6
# CHECK: < 5
# CHECK: < 4
# CHECK: < 4
# CHECK: < 5
# CHECK: < 6)IR");

  {
    PaddedBuffer<float> a_v(M, N);
    PaddedBuffer<float> b_v(N, K);
    PaddedBuffer<float> c_v(M, N);
    PaddedBuffer<float> d_v(M, K);

    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        a_v(i, j) = i * i;
      }
    }
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < K; j++) {
        b_v(i, j) = j * j;
      }
    }
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        c_v(i, j) = i + j;
      }
    }
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < K; j++) {
        d_v(i, j) = i * j;
      }
    }

    PaddedBuffer<float> z_v(M, N, K);
    PaddedBuffer<float> z_ref(M, N, K);
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
          z_ref(m, n, k) = a_v(m, n) * b_v(n, k) * 2 + c_v(m, n) * d_v(m, k);
        }
      }
    }

    SimpleIREvaluator eval(stmt, {a_buf, b_buf, c_buf, d_buf, z});
    eval(a_v, b_v, c_v, d_v, z_v);
    ExpectAllNear(z_v, z_ref, 1e-5);
  }
}

TEST(LoopNest, OuterLoopVectorization) {
  Tensor tensor =
      Compute("f", {8, 8}, [](const VarHandle& x, const VarHandle& y) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  LoopNest l({tensor});

  ASSERT_TRUE(
      LoopNest::vectorize(l.getAllLoopNestsWritingToBuf(tensor.buf())[0][0]));

  StmtPtr root_stmt = l.root_stmt();
  BlockPtr outer_block = to<Block>(root_stmt);
  ASSERT_NE(outer_block, nullptr);
  while (BlockPtr inner_block = to<Block>(outer_block->front())) {
    outer_block = inner_block;
  }

  // Verify that we have only a single loop level remaining after
  // vectorization.
  ASSERT_EQ(outer_block->nstmts(), 1);
  ForPtr for_loop = to<For>(outer_block->front());
  ASSERT_NE(for_loop, nullptr);
  BlockPtr for_body = for_loop->body();
  ASSERT_EQ(for_body->nstmts(), 1);
  ASSERT_EQ(to<For>(for_body->front()), nullptr);
}

TEST(LoopNest, VectorizeLoopNotNormalized) {
  // Input IR:
  //   for (int i = 0; i < 10; i++) {
  //     for (int j = 1; j < 5; j++) {
  //       A[i,j] = i * j;
  //     }
  //   }
  BufHandle a_buf("A", {10, 5}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto for_body = Block::make({Store::make(a_buf, {i, j}, i * j)});
  auto inner_for = For::make(j, 1, 5, for_body);
  auto outer_for = For::make(i, 0, 10, inner_for);
  auto block = Block::make({outer_for});
  LoopNest l(block, {a_buf.node()});

  ASSERT_TRUE(LoopNest::vectorize(inner_for));
  ASSERT_EQ(outer_for->body()->nstmts(), 1);
  ASSERT_EQ(to<For>(outer_for->body()->front()), nullptr);
}

namespace {

std::string constantUpperBoundLoopIR(int upper_bound_val) {
  ExprHandle upper_bound(upper_bound_val);
  Tensor A =
      Compute("A", {upper_bound}, [&](const VarHandle& x) { return x * 2; });
  LoopNest l({A});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(A.buf())[0];
  StmtPtr unrolled = nullptr;
  LoopNest::fullUnroll(loops[0], &unrolled);
  std::ostringstream oss;
  oss << *unrolled;
  return oss.str();
}

} // namespace

TEST(LoopNest, Unroll) {
  const std::string actual = constantUpperBoundLoopIR(3);
  const std::string& verification_pattern =
      R"IR(
# CHECK: A[0] = 0;
# CHECK: A[1] = 2;
# CHECK: A[2] = 4)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, actual);
}

TEST(LoopNest, UnrollOuter) {
  ExprHandle outer_bound(3);
  ExprHandle inner_bound(4);
  Tensor A = Compute(
      "A",
      {outer_bound, inner_bound},
      [&](const VarHandle& x, const VarHandle& y) { return x + y; });
  LoopNest l({A});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(A.buf())[0];
  StmtPtr unrolled = nullptr;
  LoopNest::fullUnroll(loops[0], &unrolled);
  checkIR(unrolled, R"IR(
# CHECK: for (int i = 0; i < 4; i++) {
# CHECK: A[0, i] = i;
# CHECK: }
# CHECK: for (int i = 0; i < 4; i++) {
# CHECK: A[1, i] = i + 1;
# CHECK: }
# CHECK: for (int i = 0; i < 4; i++) {
# CHECK: A[2, i] = i + 2;
# CHECK: })IR");
}

TEST(LoopNest, UnrollInner) {
  ExprHandle outer_bound(3);
  ExprHandle inner_bound(4);
  Tensor A = Compute(
      "A",
      {outer_bound, inner_bound},
      [&](const VarHandle& x, const VarHandle& y) { return x + y; });
  LoopNest l({A});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(A.buf())[0];
  StmtPtr unrolled = nullptr;
  LoopNest::fullUnroll(
      static_to<For>(loops[0]->body()->stmts().front()), &unrolled);
  checkIR(loops[0], R"IR(
# CHECK: for (int i = 0; i < 3; i++) {
# CHECK: A[i, 0] = i;
# CHECK: A[i, 1] = i + 1;
# CHECK: A[i, 2] = i + 2;
# CHECK: A[i, 3] = i + 3;
# CHECK: })IR");
}

TEST(LoopNest, UnrollMultipleStatements) {
  const int kTotalSize = 3;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);

  VarHandle x("x", kInt);
  auto f = For::make(
      x,
      0,
      kTotalSize,
      Block::make(
          {Store::make(a_buf, {x}, x * 2),
           Store::make(b_buf, {x}, Load::make(a_buf, {x}))}));
  auto parent_block = Block::make({f});
  StmtPtr unrolled = nullptr;
  LoopNest::fullUnroll(f, &unrolled);
  checkIR(unrolled, R"IR(
# CHECK: A[0] = 0;
# CHECK: B[0] = A[0];
# CHECK: A[1] = 2;
# CHECK: B[1] = A[1];
# CHECK: A[2] = 4
# CHECK: B[2] = A[2];)IR");
}

TEST(LoopNest, UnrollNonLiteralConstantBounds) {
  // Input IR:
  //   for (int i = 2 - 1; i < 12 / 3; i++) {
  //     for (int j = 0; j < 4; j++) {
  //       A[i,j] = i * j;
  //     }
  //   }
  BufHandle a_buf("A", {3, 4}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto for_body = Block::make({Store::make(a_buf, {i, j}, i * j)});
  auto inner_for = For::make(j, 0, 4, for_body);
  auto outer_for = For::make(
      i,
      IntImm::make(2) - IntImm::make(1),
      IntImm::make(12) / IntImm::make(3),
      inner_for);
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  auto b = Block::make({outer_for});

  std::vector<ForPtr> loops = {outer_for, inner_for};
  StmtPtr unrolled = nullptr;
  LoopNest::fullUnroll(loops[0], &unrolled);
  checkIR(unrolled, R"IR(
# CHECK: for (int j = 0; j < 4; j++) {
# CHECK:   A[1, j] = j;
# CHECK: }
# CHECK: for (int j = 0; j < 4; j++) {
# CHECK:   A[2, j] = 2 * j;
# CHECK: }
# CHECK: for (int j = 0; j < 4; j++) {
# CHECK:   A[3, j] = 3 * j;
# CHECK: })IR");
}

TEST(LoopNest, UnrollNonConstantBounds) {
  // Input IR:
  //   for (int i = 0; i < M; i++) {
  //     for (int j = 0; j < N; j++) {
  //       A[i, j] = i * j;
  //     }
  //   }
  VarHandle M("M", kInt);
  VarHandle N("N", kInt);
  BufHandle a_buf("A", {M, N}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto for_body = Block::make({Store::make(a_buf, {i, j}, i * j)});
  auto inner_for = For::make(j, 0, N, for_body);
  auto outer_for = For::make(i, 0, M, inner_for);
  auto block = Block::make({outer_for});
  LoopNest l(block, {a_buf.node()});

  LoopNest::unroll(inner_for, 8);
  l.simplify();
  checkIR(l.root_stmt(), R"IR(
    # CHECK: for (int i = 0; i < M; i++) {
    # CHECK:   for (int j_outer = 0; j_outer < N / 8; j_outer++) {
    # CHECK:     A[i, 8 * j_outer] =
    # CHECK:     A[i, 8 * j_outer + 1] =
    # CHECK:     A[i, 2 * (4 * j_outer + 1)] =
    # CHECK:     A[i, 8 * j_outer + 3] =
    # CHECK:     A[i, 4 * (2 * j_outer + 1)] =
    # CHECK:     A[i, 8 * j_outer + 5] =
    # CHECK:     A[i, 8 * j_outer + 6] =
    # CHECK:     A[i, 8 * j_outer + 7] =
    # CHECK:   }
    # CHECK:   for (int j_tail = 0; j_tail < N % 8; j_tail++) {
    # CHECK:     A[i, 8 * (N / 8) + j_tail] =
    # CHECK:   }
    # CHECK: }
  )IR");
}

TEST(LoopNest, UnrollByFactorsLessThan2) {
  // Input IR:
  //   for (int i = 0; i < M; i++) {
  //     for (int j = 0; j < N; j++) {
  //       A[i, j] = i * j;
  //     }
  //   }
  VarHandle M("M", kInt);
  VarHandle N("N", kInt);
  BufHandle a_buf("A", {M, N}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto for_body = Block::make({Store::make(a_buf, {i, j}, i * j)});
  auto inner_for = For::make(j, 0, N, for_body);
  auto outer_for = For::make(i, 0, M, inner_for);
  auto block = Block::make({outer_for});
  LoopNest l(block, {a_buf.node()});

  // Unrolling by factor = 1 should do nothing.
  LoopNest::unroll(inner_for, 1);
  checkIR(l.root_stmt(), R"IR(
    # CHECK: for (int i = 0; i < M; i++) {
    # CHECK:   for (int j = 0; j < N; j++) {
    # CHECK:     A[i, j] =
    # CHECK:   }
    # CHECK: }
  )IR");

  // Unrolling by factor = 0 should do nothing.
  LoopNest::unroll(inner_for, 0);
  checkIR(l.root_stmt(), R"IR(
    # CHECK: for (int i = 0; i < M; i++) {
    # CHECK:   for (int j = 0; j < N; j++) {
    # CHECK:     A[i, j] =
    # CHECK:   }
    # CHECK: }
  )IR");

  // Unrolling by negative factor should do nothing.
  LoopNest::unroll(inner_for, -2);
  checkIR(l.root_stmt(), R"IR(
    # CHECK: for (int i = 0; i < M; i++) {
    # CHECK:   for (int j = 0; j < N; j++) {
    # CHECK:     A[i, j] =
    # CHECK:   }
    # CHECK: }
  )IR");
}

TEST(LoopNest, UnrollByFactorEqualToIters) {
  // Input IR:
  //   for (int i = 0; i < 5; i++) {
  //     A[i] = i * i;
  //   }
  BufHandle a_buf("A", {5}, kInt);
  VarHandle i("i", kInt);
  auto for_body = Block::make({Store::make(a_buf, {i}, i * i)});
  auto for_loop = For::make(i, 0, 5, for_body);
  auto block = Block::make({for_loop});
  LoopNest l(block, {a_buf.node()});

  LoopNest::unroll(for_loop, 5);
  checkIR(l.root_stmt(), R"IR(
    # CHECK: for (int i_outer = 0; i_outer < (5 - 0) / 5; i_outer++)
    # CHECK:   A[5 * i_outer]
    # CHECK:   A[5 * i_outer + 1]
    # CHECK:   A[5 * i_outer + 2]
    # CHECK:   A[5 * i_outer + 3]
    # CHECK:   A[5 * i_outer + 4]
  )IR");
}

TEST(LoopNest, UnrollEmpty) {
  const std::string actual = constantUpperBoundLoopIR(0);
  const std::string& verification_pattern = R"IR(
# CHECK-NOT: A[
  )IR";

  torch::jit::testing::FileCheck().run(verification_pattern, actual);
}

TEST(LoopNest, NoUnroll) {
  VarHandle upper_bound("N", kInt);
  Tensor A =
      Compute("A", {upper_bound}, [&](const VarHandle& x) { return x * 2; });
  LoopNest l({A});
  std::vector<ForPtr> loops = l.getAllLoopNestsWritingToBuf(A.buf())[0];
  StmtPtr unrolled = nullptr;
  ASSERT_THROWS_WITH(
      LoopNest::fullUnroll(loops[0], &unrolled), "non-constant loop");
}

TEST(LoopNest, UnrollWithLet) {
  const int kTotalSize = 3;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);

  VarHandle e("e", kInt);
  VarHandle x("x", kInt);
  auto f = For::make(
      x,
      0,
      kTotalSize,
      Block::make(
          {Let::make(e, 7),
           Store::make(a_buf, {x}, e),
           Store::make(b_buf, {x}, e + 1)}));
  auto parent_block = Block::make({f});
  StmtPtr unrolled = nullptr;
  LoopNest::fullUnroll(f, &unrolled);
  std::ostringstream oss;
  oss << *unrolled;
  const std::string& verification_pattern =
      R"IR(
# CHECK: int e = 7;
# CHECK: A[0] = e;
# CHECK: B[0] = e + 1;
# CHECK: A[1] = e;
# CHECK: B[1] = e + 1;
# CHECK: A[2] = e;
# CHECK: B[2] = e + 1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  std::vector<int> a_v(kTotalSize, 0);
  std::vector<int> b_v(kTotalSize, 0);
  SimpleIREvaluator eval(unrolled, {a_buf, b_buf});
  eval(a_v, b_v);
  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v[i], 7);
    ASSERT_EQ(b_v[i], 8);
  }
}

TEST(LoopNest, IsNormalized) {
  // Input IR:
  //   for (int i = 50; i < 100; i++) {
  //     A[i] = B[i];
  //   }
  BufHandle a_buf("A", {ExprHandle(100)}, kInt);
  BufHandle b_buf("B", {ExprHandle(100)}, kInt);
  VarHandle i("i", kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto for_stmt =
      For::make(i, 50, 100, Store::make(a_buf, {i}, Load::make(b_buf, {i})));
  Block::make({for_stmt});
  ASSERT_FALSE(LoopNest::isNormalized(for_stmt));

  for_stmt->set_start(alloc<IntImm>(0));
  ASSERT_TRUE(LoopNest::isNormalized(for_stmt));

  VarHandle N("N", kInt);
  for_stmt->set_start(N.node());
  ASSERT_FALSE(LoopNest::isNormalized(for_stmt));
}

TEST(LoopNest, NormalizeStartPositive) {
  // Input IR:
  //   for (int x = 50; x < 100; x++) {
  //     A[x] = B[x];
  //     B[x] = x * 2;
  //   }
  const int kTotalSize = 50;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  VarHandle x("x", kInt);
  auto for_body = Block::make(
      {Store::make(a_buf, {x}, Load::make(kInt, b_buf, {x})),
       Store::make(b_buf, {x}, x * 2)});
  auto for_stmt = For::make(x, 50, 100, for_body);
  Block::make({for_stmt});

  LoopNest::normalize(for_stmt);

  auto result = IRSimplifier::simplify(for_stmt);
  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int x = 0; x < 50; x++) {
        # CHECK:   A[x + 50] = B[x + 50];
        # CHECK:   B[x + 50] = 2 * (x + 50);
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

TEST(LoopNest, NormalizeStartNegative) {
  // Input IR:
  //   for (int x = -50; x < 100; x++) {
  //     A[x + 50] = B[x + 50];
  //     B[x + 50] = x * 2;
  //   }
  const int kTotalSize = 150;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  VarHandle x("x", kInt);
  auto for_body = Block::make(
      {Store::make(a_buf, {x + 50}, Load::make(kInt, b_buf, {x + 50})),
       Store::make(b_buf, {x + 50}, x * 2)});
  auto for_stmt = For::make(x, -50, 100, for_body);
  Block::make({for_stmt});

  LoopNest::normalize(for_stmt);

  auto result = IRSimplifier::simplify(for_stmt);
  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int x = 0; x < 150; x++) {
        # CHECK:   A[x] = B[x];
        # CHECK:   B[x] = 2 * (x - 50);
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

TEST(LoopNest, NormalizeStartZero) {
  // Input IR:
  //   for (int x = 0; x < 100; x++) {
  //     A[x] = B[x];
  //     B[x] = x * 2;
  //   }
  // Should not be modified.

  const int kTotalSize = 100;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  VarHandle x("x", kInt);
  auto for_body = Block::make(
      {Store::make(a_buf, {x}, Load::make(kInt, b_buf, {x})),
       Store::make(b_buf, {x}, x * 2)});
  auto for_stmt = For::make(x, 0, 100, for_body);
  Block::make({for_stmt});

  LoopNest::normalize(for_stmt);

  auto result = IRSimplifier::simplify(for_stmt);
  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int x = 0; x < 100; x++) {
        # CHECK:   A[x] = B[x];
        # CHECK:   B[x] = 2 * x;
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

TEST(LoopNest, NormalizeStartVariable) {
  // Input IR:
  //   for (int x = y; x < 100; x++) {
  //     A[x] = B[x];
  //     B[x] = x * 2;
  //   }

  const int kTotalSize = 100;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  auto for_body = Block::make(
      {Store::make(a_buf, {x}, Load::make(kInt, b_buf, {x})),
       Store::make(b_buf, {x}, x * 2)});
  auto for_stmt = For::make(x, y, 100, for_body);
  auto parent_block = Block::make({for_stmt});

  LoopNest::normalize(for_stmt);

  auto result = IRSimplifier::simplify(for_stmt);
  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int x = 0; x < 100 - y; x++) {
        # CHECK:   A[x + y] = B[x + y];
        # CHECK:   B[x + y] = 2 * (x + y);
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

TEST(LoopNest, NormalizeOnNestedOuterLoop) {
  // Input IR:
  //   for (int x = 50; x < 100; x++) {
  //     for (int y = 10; y < 100; y++) {
  //       A[x] = A[x] + B[y] + y * 2;
  //     }
  //   }

  BufHandle a_buf("A", {ExprHandle(50)}, kInt);
  BufHandle b_buf("B", {ExprHandle(100)}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  auto inner_for_body = Store::make(
      a_buf, {x}, Load::make(a_buf, {x}) + Load::make(b_buf, {y}) + y * 2);
  auto inner_for = For::make(y, 10, 100, inner_for_body);
  auto for_stmt = For::make(x, 50, 100, inner_for);
  Block::make({for_stmt});

  LoopNest::normalize(for_stmt);

  auto result = IRSimplifier::simplify(for_stmt);
  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int x = 0; x < 50; x++) {
        # CHECK:   for (int y = 10; y < 100; y++) {
        # CHECK:     A[x + 50] = ((A[x + 50]) + (B[y])) + 2 * y;
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

TEST(LoopNest, NormalizeOnNestedInnerLoop) {
  // Input IR:
  //   for (int x = 50; x < 100; x++) {
  //     for (int y = 10; y < 100; y++) {
  //       A[x] = A[x] + B[y] + y * 2;
  //     }
  //   }

  BufHandle a_buf("A", {ExprHandle(50)}, kInt);
  BufHandle b_buf("B", {ExprHandle(100)}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  auto inner_for_body = Store::make(
      a_buf, {x}, Load::make(a_buf, {x}) + Load::make(b_buf, {y}) + y * 2);
  auto inner_for = For::make(y, 10, 100, inner_for_body);
  auto for_stmt = For::make(x, 50, 100, inner_for);
  Block::make({for_stmt});

  LoopNest::normalize(inner_for);

  auto result = IRSimplifier::simplify(for_stmt);
  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int x = 50; x < 100; x++) {
        # CHECK:   for (int y = 0; y < 90; y++) {
        # CHECK:     A[x] = (((A[x]) + (B[y + 10])) + 2 * y) + 20;
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

TEST(LoopNest, NormalizeAndSplitWithTail) {
  // Create a dummy tensor to construct LoopNest.
  ExprHandle n(100);
  BufHandle a("a", {n}, kFloat);
  Tensor b = Compute("b", {n}, [&](const VarHandle& i) { return a.load(i); });
  LoopNest l({b});

  // Input IR:
  //   for (int x = 5; x < 10; x++) {
  //     A[x] = x * 2;
  //   }
  const int kTotalSize = 5;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  VarHandle x("x", kInt);
  auto for_stmt = For::make(x, 5, 10, Store::make(a_buf, {x}, x * 2));
  auto parent_block = Block::make({for_stmt});

  LoopNest::normalize(for_stmt);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr x_inner;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr x_tail;
  LoopNest::splitWithTail(for_stmt, 10, &x_inner, &x_tail);

  auto x_outer_result = IRSimplifier::simplify(for_stmt);
  std::ostringstream oss_outer;
  oss_outer << *x_outer_result;
  const std::string& expected_outer_ir =
      R"IR(
        # CHECK: {
        # CHECK: }
      )IR";
  torch::jit::testing::FileCheck().run(expected_outer_ir, oss_outer.str());

  auto x_tail_result = IRSimplifier::simplify(x_tail);
  std::ostringstream oss_tail;
  oss_tail << *x_tail_result;
  const std::string& expected_tail_ir =
      R"IR(
        # CHECK: for (int x_tail = 0; x_tail < 5; x_tail++) {
        # CHECK:   A[x_tail + 5] = 2 * (x_tail + 5);
      )IR";
  torch::jit::testing::FileCheck().run(expected_tail_ir, oss_tail.str());
}

TEST(LoopNest, NotNormalizeAndSplitWithTail) {
  // Create a dummy tensor to construct LoopNest.
  ExprHandle n(100);
  BufHandle a("a", {n}, kFloat);
  Tensor b = Compute("b", {n}, [&](const VarHandle& i) { return a.load(i); });
  LoopNest l({b});

  // Input IR:
  //   for (int x = 5; x < 15; x++) {
  //     A[x] = x * 2;
  //   }
  const int kTotalSize = 10;
  BufHandle a_buf("A", {kTotalSize}, kInt);
  VarHandle x("x", kInt);
  auto for_stmt = For::make(x, 5, 15, Store::make(a_buf, {x}, x * 2));
  auto parent_block = Block::make({for_stmt});

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr x_inner;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr x_tail;
  LoopNest::splitWithTail(for_stmt, 8, &x_inner, &x_tail);

  auto x_outer_result = IRSimplifier::simplify(for_stmt);
  std::ostringstream oss_outer;
  oss_outer << *x_outer_result;
  const std::string& expected_outer_ir =
      R"IR(
        # CHECK: {
        # CHECK: }
      )IR";
  torch::jit::testing::FileCheck().run(expected_outer_ir, oss_outer.str());

  auto x_tail_result = IRSimplifier::simplify(x_tail);
  std::ostringstream oss_tail;
  oss_tail << *x_tail_result;
  const std::string& expected_tail_ir =
      R"IR(
        # CHECK: for (int x_tail = 0; x_tail < 2; x_tail++) {
        # CHECK:   A[x_tail + 13] = 2 * (x_tail + 13);
      )IR";
  torch::jit::testing::FileCheck().run(expected_tail_ir, oss_tail.str());
}

TEST(LoopNest, FlattenSimpleLoopNest2D) {
  // Input IR:
  //   for (int i = 0; i < 10; i++) {
  //     for (int j = 0; j < 5; j++) {
  //       A[i,j] = i * j;
  //     }
  //   }
  BufHandle a_buf("A", {10, 5}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto for_body = Block::make({Store::make(a_buf, {i, j}, i * j)});
  auto inner_for = For::make(j, 0, 5, for_body);
  auto outer_for = For::make(i, 0, 10, inner_for);
  auto parent_block = Block::make({outer_for});

  std::vector<ForPtr> loops = {outer_for, inner_for};
  ForPtr flattened = nullptr;
  ASSERT_TRUE(LoopNest::flatten(loops, &flattened));
  ASSERT_EQ(flattened, loops.front());

  auto result = IRSimplifier::simplify(flattened);
  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int i_flat = 0; i_flat < 50; i_flat++) {
        # CHECK:   A[i_flat / 5, i_flat % 5] =
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());

  {
    SimpleIREvaluator eval1(loops[0], {a_buf});
    PaddedBuffer<int> inp1(10, 5);
    eval1(inp1);
    SimpleIREvaluator eval2(flattened, {a_buf});
    PaddedBuffer<int> inp2(10, 5);
    eval2(inp2);
    ExpectAllNear(inp1, inp2, 1e-5);
  }
}

TEST(LoopNest, FlattenSimpleLoopNest3D) {
  // Input IR:
  //   for (int i = 0; i < 10; i++) {
  //     for (int j = 0; j < 5; j++) {
  //       for (int k = 0; k < 7; k++) {
  //         A[i,j,k] = i + j * k;
  //       }
  //     }
  //   }
  BufHandle a_buf("A", {10, 5, 7}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto for_body = Block::make({Store::make(a_buf, {i, j, k}, i + j * k)});
  auto for1 = For::make(k, 0, 7, for_body);
  auto for2 = For::make(j, 0, 5, for1);
  auto for3 = For::make(i, 0, 10, for2);
  auto parent_block = Block::make({for3});

  std::vector<ForPtr> loops = {for3, for2, for1};
  ForPtr flattened = nullptr;
  ASSERT_TRUE(LoopNest::flatten(loops, &flattened));
  ASSERT_EQ(flattened, loops.front());

  auto result = IRSimplifier::simplify(flattened);
  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int i_flat = 0; i_flat < 350; i_flat++) {
        # CHECK:   A[i_flat / 35, (i_flat / 7) % 5, i_flat % 7] =
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());

  {
    SimpleIREvaluator eval1(loops[0], {a_buf});
    PaddedBuffer<int> inp1(10, 5, 7);
    eval1(inp1);
    SimpleIREvaluator eval2(flattened, {a_buf});
    PaddedBuffer<int> inp2(10, 5, 7);
    eval2(inp2);
    ExpectAllNear(inp1, inp2, 1e-5);
  }
}

TEST(LoopNest, FlattenLoopNestAfterNormalize) {
  // Input IR:
  //   for (int i = 2; i < 10; i++) {
  //     for (int j = 3; j < 15; j++) {
  //       A[i - 2,j - 3] = i * j;
  //     }
  //   }
  BufHandle a_buf("A", {8, 12}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto for_body = Block::make({Store::make(a_buf, {i - 2, j - 3}, i * j)});
  auto inner_for = For::make(j, 3, 15, for_body);
  auto outer_for = For::make(i, 2, 10, inner_for);
  auto parent_block = Block::make({outer_for});

  std::vector<ForPtr> loops = {outer_for, inner_for};
  ForPtr flattened = nullptr;
  ASSERT_TRUE(LoopNest::flatten(loops, &flattened));
  ASSERT_EQ(flattened, loops.front());

  auto result = IRSimplifier::simplify(flattened);
  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int i_flat = 0; i_flat < 96; i_flat++) {
        # CHECK:   A[i_flat / 12, i_flat % 12] =
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());

  {
    SimpleIREvaluator eval1(loops[0], {a_buf});
    PaddedBuffer<int> inp1(8, 12);
    eval1(inp1);
    SimpleIREvaluator eval2(flattened, {a_buf});
    PaddedBuffer<int> inp2(8, 12);
    eval2(inp2);
    ExpectAllNear(inp1, inp2, 1e-5);
  }
}

TEST(LoopNest, FlattenLoopNestWithNonLiteralConstantBounds) {
  // Input IR:
  //   for (int i = 0; i < 15-5; i++) {
  //     for (int j = 0; j < 20/4; j++) {
  //       A[i,j] = i * j;
  //     }
  //   }
  BufHandle a_buf("A", {10, 5}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto for_body = Block::make({Store::make(a_buf, {i, j}, i * j)});
  auto inner_for =
      For::make(j, 0, IntImm::make(20) / IntImm::make(4), for_body);
  auto outer_for =
      For::make(i, 0, IntImm::make(15) - IntImm::make(5), inner_for);
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  auto b = Block::make({outer_for});

  std::vector<ForPtr> loops = {outer_for, inner_for};
  ForPtr flattened = nullptr;
  ASSERT_TRUE(LoopNest::flatten(loops, &flattened));
  ASSERT_EQ(flattened, loops.front());

  auto result = IRSimplifier::simplify(flattened);
  checkIR(result, R"IR(
        # CHECK: for (int i_flat = 0; i_flat < 50; i_flat++) {
        # CHECK:   A[i_flat / 5, i_flat % 5] =
      )IR");

  {
    SimpleIREvaluator eval1(loops[0], {a_buf});
    PaddedBuffer<int> inp1(10, 5);
    eval1(inp1);
    SimpleIREvaluator eval2(flattened, {a_buf});
    PaddedBuffer<int> inp2(10, 5);
    eval2(inp2);
    ExpectAllNear(inp1, inp2, 1e-5);
  }
}

TEST(LoopNest, FlattenImperfectLoopNest) {
  // Input IR:
  //   for (int i = 0; i < 10; i++) {
  //     A[i, i] = 0;
  //     for (int j = 0; j < 15; j++) {
  //       A[i,j] = i * j;
  //     }
  //   }
  // Do not flatten.

  BufHandle a_buf("A", {10, 15}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto for_body = Block::make({Store::make(a_buf, {i, j}, i * j)});
  auto inner_for = For::make(j, 0, 15, for_body);
  auto outer_for = For::make(
      i, 0, 10, Block::make({Store::make(a_buf, {i, i}, 0), inner_for}));
  auto par = Block::make({outer_for});
  HashProvider hasher;
  auto hash_before = hasher.hash(par);

  std::vector<ForPtr> loops = {outer_for, inner_for};
  ForPtr flattened = nullptr;
  ASSERT_FALSE(LoopNest::flatten(loops, &flattened));
  ASSERT_EQ(flattened, nullptr);
  auto hash_after = hasher.hash(par);
  ASSERT_EQ(hash_before, hash_after);
}

TEST(LoopNest, FlattenReductionLoopNest) {
  // Input IR:
  //   for (int i = 0; i < 10; i++) {
  //     S[i] = 0;
  //     for (int j = 0; j < 15; j++) {
  //       S[i] = S[i] + A[i,j];
  //     }
  //   }
  // Do not flatten.

  BufHandle a_buf("A", {10, 15}, kInt);
  BufHandle s_buf("S", {10}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto for_body = Block::make({Store::make(
      s_buf, {i}, Load::make(s_buf, {i}) + Load::make(a_buf, {i, j}))});
  auto inner_for = For::make(j, 0, 15, for_body);
  auto outer_for =
      For::make(i, 0, 10, Block::make({Store::make(s_buf, {i}, 0), inner_for}));
  auto par = Block::make({outer_for});
  HashProvider hasher;
  auto hash_before = hasher.hash(par);

  std::vector<ForPtr> loops = {outer_for, inner_for};
  ForPtr flattened = nullptr;
  ASSERT_FALSE(LoopNest::flatten(loops, &flattened));
  ASSERT_EQ(flattened, nullptr);
  auto hash_after = hasher.hash(par);
  ASSERT_EQ(hash_before, hash_after);
}

TEST(LoopNest, FlattenReductionLoopNestFromTensor) {
  const int M = 3;
  const int N = 7;
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  BufHandle b("b", {m, n}, kFloat);
  Tensor c = Reduce("sum", {M}, Sum(), b, {N});
  LoopNest loop({c});
  HashProvider hasher;
  auto hash_before = hasher.hash(loop.root_stmt());

  auto loops = loop.getAllLoopNestsWritingToBuf(c.buf())[1];
  ForPtr flattened = nullptr;
  ASSERT_FALSE(LoopNest::flatten(loops, &flattened));
  ASSERT_EQ(flattened, nullptr);
  auto hash_after = hasher.hash(loop.root_stmt());
  ASSERT_EQ(hash_before, hash_after);
}

TEST(LoopNest, FlattenIncorrectLoopsAsInput) {
  // Input IR:
  //   for (int i = 0; i < 10; i++) {
  //     for (int j = 0; j < 5; j++) {
  //       A[i,j] = i * j;
  //     }
  //   }
  //   for (int x = 0; x < 10; x++) {
  //     for (int y = 0; y < 5; y++) {
  //       A[x,y] = A[x,y] + x + y;
  //     }
  //   }
  // Flatten({For_i, For_y}) => should not succeed

  BufHandle a_buf("A", {10, 5}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  auto for_body1 = Block::make({Store::make(a_buf, {i, j}, i * j)});
  auto inner_for1 = For::make(j, 0, 5, for_body1);
  auto outer_for1 = For::make(i, 0, 10, inner_for1);
  auto for_body2 = Block::make(
      {Store::make(a_buf, {x, y}, Load::make(a_buf, {x, y}) + x + y)});
  auto inner_for2 = For::make(y, 0, 5, for_body2);
  auto outer_for2 = For::make(x, 0, 10, inner_for2);
  auto par = Block::make({outer_for1, outer_for2});
  HashProvider hasher;
  auto hash_before = hasher.hash(par);

  std::vector<ForPtr> loops = {outer_for1, inner_for2};
  ForPtr flattened = nullptr;
  ASSERT_FALSE(LoopNest::flatten(loops, &flattened));
  ASSERT_EQ(flattened, nullptr);
  auto hash_after = hasher.hash(par);
  ASSERT_EQ(hash_before, hash_after);
}

TEST(LoopNest, DetectInlineRankMismatch) {
  const int kTotalSize = 8;

  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  Tensor a = Compute(
      "a", {kTotalSize}, [&](const VarHandle& i) { return a_buf.load(i); });
  Tensor reshape = Compute(
      "reshape",
      {kTotalSize / 2, 2},
      [&](const VarHandle& i, const VarHandle& j) { return a.load(i, j); });
  LoopNest l({reshape}, {a, reshape});
  ASSERT_FALSE(l.computeInline(l.getLoopBodyFor(a)));
}

TEST(LoopNest, CacheReadsSimple) {
  Tensor A = Compute("A", {64, 64}, [](const VarHandle& i, const VarHandle& j) {
    return i * j;
  });
  Tensor B =
      Compute("B", {20, 10}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i + 30, j + 3);
      });
  Tensor C =
      Compute("C", {20, 10}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i + 10, j + 20) + A.load(i + 30, j + 40);
      });

  LoopNest l({B, C}, {A, B, C});
  StmtPtr j_loop = l.getAllLoopNestsWritingToBuf(B.buf())[0][1];
  LoopNest::cacheAccesses(A.buf(), "A_local", j_loop);

  l.prepareForCodegen();
  StmtPtr result =
      LoopNest::sanitizeNames(IRSimplifier::simplify(l.root_stmt()));
  SimpleIREvaluator cg(result, {B, C});
  result = cg.stmt();

  // just this once: verify the whole thing.
  checkIR(result, R"IR(
#CHECK: Allocate(A); // dtype=int, dims=[64, 64]
#CHECK: Allocate(A_local); // dtype=int, dims=[1, 10]
#CHECK: for (int i
#CHECK:  for (int j
#CHECK:   A[
#CHECK:  }
#CHECK: }
#CHECK: for (int i_1
#CHECK:  for (int j_1
#CHECK:   A_local[j_1] = A[
#CHECK:  }
#CHECK:  for (int j_2
#CHECK:   B[j_2 + 10 * i_1] = A_local[j_2];
#CHECK:  }
#CHECK: }
#CHECK: for (int i_2
#CHECK:  for (int j_3
#CHECK:   C[
#CHECK:  }
#CHECK: }
#CHECK: Free(A_local);
#CHECK: Free(A);
      )IR");

  std::vector<int> b_data(200, 0);
  std::vector<int> c_data(200, 0);
  cg.call({b_data, c_data});

  std::vector<int> b_ref(200, 0);
  std::vector<int> c_ref(200, 0);

  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 10; ++j) {
      b_ref[i * 10 + j] = (i + 30) * (j + 3);
      c_ref[i * 10 + j] = (i + 10) * (j + 20) + (i + 30) * (j + 40);
    }
  }

  assertAllEqual(b_data, b_ref);
  assertAllEqual(c_data, c_ref);
}

TEST(LoopNest, CacheReadsOuter) {
  Tensor A = Compute("A", {64, 64}, [](const VarHandle& i, const VarHandle& j) {
    return i * j;
  });
  Tensor B =
      Compute("B", {20, 10}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i + 30, j + 40) + A.load(i + 31, j + 41);
      });
  Tensor C =
      Compute("C", {20, 10}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i + 10, j + 20) + A.load(i + 30, j + 40);
      });

  LoopNest l({B, C}, {A, B, C});
  StmtPtr i_loop = l.getAllLoopNestsWritingToBuf(B.buf())[0][0];
  LoopNest::cacheAccesses(A.buf(), "A_local", i_loop);

  l.prepareForCodegen();
  StmtPtr result =
      LoopNest::sanitizeNames(IRSimplifier::simplify(l.root_stmt()));
  SimpleIREvaluator cg(result, {B, C});
  result = cg.stmt();

  checkIR(result, R"IR(
#CHECK: Allocate(A_local); // dtype=int, dims=[21, 11]
#CHECK: A_local[j_1 + 11 * i_1] =
#CHECK: B[j_2 + 10 * i_2] = (A_local[j_2 + 11 * i_2]) + (A_local[(j_2 + 11 * i_2) + 12]);
      )IR");

  std::vector<int> b_data(200, 0);
  std::vector<int> c_data(200, 0);
  cg.call({b_data, c_data});

  std::vector<int> b_ref(200, 0);
  std::vector<int> c_ref(200, 0);

  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 10; ++j) {
      b_ref[i * 10 + j] = (i + 30) * (j + 40) + (i + 31) * (j + 41);
      c_ref[i * 10 + j] = (i + 10) * (j + 20) + (i + 30) * (j + 40);
    }
  }

  assertAllEqual(b_data, b_ref);
  assertAllEqual(c_data, c_ref);
}

TEST(LoopNest, CacheReadsInternal) {
  Tensor A = Compute("A", {64, 64}, [](const VarHandle& i, const VarHandle& j) {
    return i * j;
  });
  Tensor B =
      Compute("B", {20, 10}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i + 30, j + 40) + A.load(i + 31, j + 41);
      });
  Tensor C =
      Compute("C", {20, 10}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i + 10, j + 20) + A.load(i + 30, j + 40);
      });

  LoopNest l({B, C}, {A, B, C});
  StmtPtr j_loop = l.getAllLoopNestsWritingToBuf(B.buf())[0][1];
  LoopNest::cacheAccesses(A.buf(), "A_local", j_loop);
  l.prepareForCodegen();
  StmtPtr result =
      LoopNest::sanitizeNames(IRSimplifier::simplify(l.root_stmt()));
  SimpleIREvaluator cg(result, {B, C});
  result = cg.stmt();

  checkIR(result, R"IR(
#CHECK: Allocate(A_local); // dtype=int, dims=[2, 11]
#CHECK: A_local[k + 11 * j_1] =
#CHECK: B[j_2 + 10 * i_1] = (A_local[j_2 + 12]) + (A_local[j_2]);
      )IR");

  std::vector<int> b_data(200, 0);
  std::vector<int> c_data(200, 0);
  cg.call({b_data, c_data});

  std::vector<int> b_ref(200, 0);
  std::vector<int> c_ref(200, 0);

  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 10; ++j) {
      b_ref[i * 10 + j] = (i + 30) * (j + 40) + (i + 31) * (j + 41);
      c_ref[i * 10 + j] = (i + 10) * (j + 20) + (i + 30) * (j + 40);
    }
  }

  assertAllEqual(b_data, b_ref);
  assertAllEqual(c_data, c_ref);
}

TEST(LoopNest, CacheReadsInner) {
  Tensor A = Compute("A", {64, 64}, [](const VarHandle& i, const VarHandle& j) {
    return i * j;
  });
  // note im changing the offset of the first arg of the first call to A.
  Tensor B =
      Compute("B", {20, 10}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i + 34, j + 40) + A.load(i + 30, j + 41);
      });
  Tensor C =
      Compute("C", {20, 10}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i + 10, j + 20) + A.load(i + 30, j + 40);
      });

  LoopNest l({B, C}, {A, B, C});
  StmtPtr body = l.getLoopBodyFor(B);
  LoopNest::cacheAccesses(A.buf(), "A_local", body);
  l.prepareForCodegen();
  StmtPtr result =
      LoopNest::sanitizeNames(IRSimplifier::simplify(l.root_stmt()));
  SimpleIREvaluator cg(result, {B, C});
  result = cg.stmt();

  checkIR(result, R"IR(
#CHECK: Allocate(A_local); // dtype=int, dims=[5, 2]
#CHECK: A_local[l + 2 * k] =
#CHECK: B[j_1 + 10 * i_1] = (A_local[1]) + (A_local[8]);
      )IR");

  std::vector<int> b_data(200, 0);
  std::vector<int> c_data(200, 0);
  cg.call({b_data, c_data});

  std::vector<int> b_ref(200, 0);
  std::vector<int> c_ref(200, 0);

  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 10; ++j) {
      b_ref[i * 10 + j] = (i + 34) * (j + 40) + (i + 30) * (j + 41);
      c_ref[i * 10 + j] = (i + 10) * (j + 20) + (i + 30) * (j + 40);
    }
  }

  assertAllEqual(b_data, b_ref);
  assertAllEqual(c_data, c_ref);
}

TEST(LoopNest, CacheWritesSimple) {
  Tensor A = Compute("A", {64, 64}, [](const VarHandle& i, const VarHandle& j) {
    return i * j;
  });
  Tensor B =
      Compute("B", {20, 10}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i + 30, j + 40) + A.load(i + 31, j + 41);
      });
  Tensor C =
      Compute("C", {20, 10}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i + 10, j + 20) + A.load(i + 30, j + 40);
      });

  LoopNest l({B, C}, {A, B, C});
  StmtPtr a_loop = l.getAllLoopNestsWritingToBuf(A.buf())[0][1];
  LoopNest::cacheAccesses(A.buf(), "A_local", a_loop);

  l.prepareForCodegen();
  StmtPtr result =
      LoopNest::sanitizeNames(IRSimplifier::simplify(l.root_stmt()));
  SimpleIREvaluator cg(result, {B, C});
  result = cg.stmt();

  checkIR(result, R"IR(
#CHECK: Allocate(A_local); // dtype=int, dims=[1, 64]
#CHECK: for (int j = 0; j < 64
#CHECK:   A_local[j] = i * j;
#CHECK: for (int j_1 = 0; j_1 < 64
#CHECK:   A[j_1 + 64 * i] = A_local[
#CHECK: Free(A_local);
#CHECK-NOT: A_local
      )IR");

  std::vector<int> b_data(200, 0);
  std::vector<int> c_data(200, 0);
  cg.call({b_data, c_data});

  std::vector<int> b_ref(200, 0);
  std::vector<int> c_ref(200, 0);

  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 10; ++j) {
      b_ref[i * 10 + j] = (i + 30) * (j + 40) + (i + 31) * (j + 41);
      c_ref[i * 10 + j] = (i + 10) * (j + 20) + (i + 30) * (j + 40);
    }
  }

  assertAllEqual(b_data, b_ref);
  assertAllEqual(c_data, c_ref);
}

TEST(LoopNest, DeadStoreElimination) {
  VarHandle y("y", kInt);
  VarHandle x("x_tail", kInt);
  BufHandle f("f", {26, 5}, kInt);
  BufHandle g("g", {26, 5}, kInt);
  ExprHandle x_outer_end = 5;
  ExprHandle x_2 = x + x_outer_end * 4;
  ForPtr stmt1 = For::make(
      x,
      0,
      5,
      For::make(
          y,
          0,
          5,
          Block::make({
              Store::make(f, {x_2, y}, (x_2 + y)),
              Store::make(g, {x_2, y}, (x_2 * y)),
          })));
  StmtPtr stmt = Block::make({stmt1});

  // Will eliminate if not used by an output.
  LoopNest loop(Stmt::clone(stmt), {f.node()});
  loop.eliminateDeadStores();

  checkIR(loop.root_stmt(), R"IR(
#CHECK:     f[x_tail + 5 * 4, y]
#CHECK-NOT: g[x_tail + 5 * 4, y]
      )IR");

  // But won't eliminate if used by different outputs.
  LoopNest loop2(stmt, {f.node(), g.node()});
  loop2.eliminateDeadStores();

  checkIR(loop2.root_stmt(), R"IR(
#CHECK:     f[x_tail + 5 * 4, y]
#CHECK:     g[x_tail + 5 * 4, y]
      )IR");
}

TEST(LoopNest, DeadStoreEliminationWithIntermediates) {
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);
  BufHandle f("f", {26 * 5}, kInt);
  BufHandle g("g", {26 * 5}, kInt);
  BufHandle h("h", {26, 5}, kInt);
  ExprHandle x_outer_end = 5;
  ExprHandle x_2 = x + x_outer_end * 4;
  ForPtr stmt1 = For::make(x, 0, 26 * 5, Store::make(f, {x}, x));
  ForPtr stmt2 = For::make(z, 0, 26 * 5, Store::make(g, {z}, z + 1));
  ForPtr stmt3 = For::make(
      x,
      0,
      5,
      For::make(
          y,
          0,
          5,
          Block::make({
              Store::make(h, {x, y}, Load::make(f, {x * y})),
          })));
  StmtPtr stmt = Block::make({stmt1, stmt2, stmt3});

  // Will eliminate the write to g, but not f since it used by the producer of
  // h.
  LoopNest loop(Stmt::clone(stmt), {h.node()});
  loop.eliminateDeadStores();

  checkIR(loop.root_stmt(), R"IR(
  #CHECK:     f[x] = x;
  #CHECK-NOT: g[z] =
  #CHECK:     h[x, y] = f[x * y];
      )IR");

  // Sanity check won't eliminate if g is an output.
  LoopNest loop2(stmt, {h.node(), g.node()});
  loop2.eliminateDeadStores();

  checkIR(loop2.root_stmt(), R"IR(
  #CHECK:     f[x] = x;
  #CHECK:     g[z] = z + 1;
  #CHECK:     h[x, y] = f[x * y];
      )IR");
}

TEST(LoopNest, CompoundTensorSimple) {
  BufHandle a_buf("A", {10, 5}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  auto for_body1 = Block::make({Store::make(a_buf, {i, j}, i * j)});
  auto inner_for1 = For::make(j, 0, 5, for_body1);
  auto outer_for1 = For::make(i, 0, 10, inner_for1);
  auto for_body2 = Block::make(
      {Store::make(a_buf, {x, y}, Load::make(a_buf, {x, y}) + x + y)});
  auto inner_for2 = For::make(y, 0, 5, for_body2);
  auto outer_for2 = For::make(x, 0, 10, inner_for2);
  BlockPtr body = Block::make({outer_for1, outer_for2});

  Tensor A = Tensor(a_buf.node(), body);

  LoopNest l({A});
  l.prepareForCodegen();

  std::vector<int> a_data(50, 0);

  StmtPtr s = IRSimplifier::simplify(l.root_stmt());
  SimpleIREvaluator cg(s, {A});

  std::vector<int> a_ref(50, 0);

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 5; ++j) {
      a_ref[i * 5 + j] = (i * j) + i + j;
    }
  }
  cg.call({a_data});

  assertAllEqual(a_data, a_ref);
}

TEST(LoopNest, InlineConstantIndex) {
  const int N = 10;
  BufHandle x_buf("a", {1, N, 1}, kFloat);
  Tensor y = Compute(
      "f",
      {1, N, 1},
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& o) {
        return x_buf.load(m, n, o);
      });
  Tensor z = Compute(
      "f",
      {1, N, 1},
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& o) {
        return y.load(m, n, o);
      });

  LoopNest l({z}, {y, z});
  l.simplify();
  ASSERT_TRUE(l.computeInline(y.buf()));
}

TEST(LoopNest, CompoundTensorUsed) {
  BufHandle a_buf("A", {10, 5}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  auto for_body1 = Block::make({Store::make(a_buf, {i, j}, i * j)});
  auto inner_for1 = For::make(j, 0, 5, for_body1);
  auto outer_for1 = For::make(i, 0, 10, inner_for1);
  auto for_body2 = Block::make(
      {Store::make(a_buf, {x, y}, Load::make(a_buf, {x, y}) + x + y)});
  auto inner_for2 = For::make(y, 0, 5, for_body2);
  auto outer_for2 = For::make(x, 0, 10, inner_for2);
  BlockPtr body = Block::make({outer_for1, outer_for2});

  Tensor A = Tensor(a_buf.node(), body);
  Tensor B = Compute("B", {10, 3}, [&](const VarHandle& i, const VarHandle& j) {
    return A.load(i, j + 1) + A.load(i, j + 2);
  });

  LoopNest l({B}, {A, B});
  ASSERT_FALSE(l.computeInline(A.buf()));
  l.prepareForCodegen();

  std::vector<int> a_data(50, 0);
  std::vector<int> b_data(50, 0);

  StmtPtr s = IRSimplifier::simplify(l.root_stmt());
  SimpleIREvaluator cg(s, {B});

  std::vector<int> b_ref(50, 0);

  auto AT = [](int i, int j) { return i * j + i + j; };
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 3; ++j) {
      b_ref[i * 3 + j] = AT(i, j + 1) + AT(i, j + 2);
    }
  }
  cg.call({b_data});

  assertAllEqual(b_data, b_ref);
}

TEST(LoopNest, InlineFromLoad) {
  constexpr int N = 1024;
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto store_a = For::make(i, 0, N, Store::make(a, {i}, i));
  auto store_b = For::make(j, 0, N, Store::make(b, {j}, Load::make(a, {j})));
  LoopNest l(Block::make({store_a, store_b}), {b.node()});

  l.computeInline(a.node());

  // Check that A[j] is replaced with j after inlining
  std::ostringstream oss;
  oss << *l.root_stmt();
  torch::jit::testing::FileCheck().run(
      R"IR(
# CHECK: for (int j
# CHECK-NOT: B[j] = A[j]
# CHECK-NEXT: B[j] = j
)IR",
      oss.str());
}

TEST(LoopNest, OptimizeConditionalsSimple) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = IfThenElse(i<5 ? 1 : 0, B[i], C[i-5])
  //   }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle a_buf("A", {20}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle b_buf("B", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle c_buf("C", {15}, kInt);
  VarHandle i("i", kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto store = Store::make(
      a_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(i, 5, kLT),
          Load::make(b_buf, {i}),
          Load::make(c_buf, {i - 5})));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto forI = For::make(i, 0, 20, store);
  auto par = Block::make({forI});

  LoopNest nest(par, {a_buf.node()});
  nest.optimizeConditionals();

  std::ostringstream oss;
  oss << *nest.root_stmt();
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i = 0; i < 5
# CHECK-NEXT: A[i] = B[i]
# CHECK: for (int i = 0; i < 15
# CHECK-NEXT: A[i + 5] = C[i]
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(LoopNest, OptimizeConditionalsNestedConditions) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = IfThenElse(i<10, IfThenElse(i<5, B[i], C[i-5]), D[i-10])
  //   }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle a_buf("A", {20}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle b_buf("B", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle c_buf("C", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle d_buf("D", {10}, kInt);
  VarHandle i("i", kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto store = Store::make(
      a_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(i, 10, kLT),
          IfThenElse::make(
              CompareSelect::make(i, 5, kLT),
              Load::make(b_buf, {i}),
              Load::make(c_buf, {i - 5})),
          Load::make(d_buf, {i - 10})));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto forI = For::make(i, 0, 20, store);
  auto par = Block::make({forI});

  LoopNest nest(par, {a_buf.node()});
  nest.optimizeConditionals();

  std::ostringstream oss;
  oss << *nest.root_stmt();
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i = 0; i < 5
# CHECK-NEXT: A[i] = B[i]
# CHECK: for (int i = 0; i < 5
# CHECK-NEXT: A[i + 5] = C[i]
# CHECK: for (int i = 0; i < 10
# CHECK-NEXT: A[i + 10] = D[i]
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(LoopNest, OptimizeConditionalsMultipleStores) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = IfThenElse(i<5 ? 1 : 0, B[i], C[i-5])
  //   }
  //   for (int j = 0; j < 100; j++) {
  //     B[j] = IfThenElse(j<30 ? 1 : 0, C[j], D[j])
  //   }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle a_buf("A", {20}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle b_buf("B", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle c_buf("C", {100}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle d_buf("D", {100}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto storeA = Store::make(
      a_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(i, 5, kLT),
          Load::make(b_buf, {i}),
          Load::make(c_buf, {i - 5})));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto forI = For::make(i, 0, 20, storeA);
  auto storeB = Store::make(
      b_buf,
      {j},
      IfThenElse::make(
          CompareSelect::make(j, 30, kLT),
          Load::make(c_buf, {j}),
          Load::make(d_buf, {j})));
  auto forJ = For::make(j, 0, 100, storeB);
  auto par = Block::make({forI, forJ});

  LoopNest nest(par, {a_buf.node()});
  nest.optimizeConditionals();

  std::ostringstream oss;
  oss << *nest.root_stmt();
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i = 0; i < 5
# CHECK-NEXT: A[i] = B[i]
# CHECK: for (int i = 0; i < 15
# CHECK-NEXT: A[i + 5] = C[i]
# CHECK: for (int j = 0; j < 30
# CHECK-NEXT: B[j] = C[j]
# CHECK: for (int j = 0; j < 70
# CHECK-NEXT: B[j + 30] = D[j + 30]
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(LoopNest, OptimizeConditionalsMultipleStoresInOneLoop) {
  // Input IR:
  //   for (int i = 0; i < 50; i++) {
  //     A[i] = IfThenElse(i<5 ? 1 : 0, B[i], C[i-5])
  //     B[j] = IfThenElse(j<30 ? 1 : 0, C[j], D[j])
  //   }
  // Only the first conditional, in the write to A, will be optimized.

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle a_buf("A", {100}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle b_buf("B", {100}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle c_buf("C", {100}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle d_buf("D", {100}, kInt);
  VarHandle i("i", kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto storeA = Store::make(
      a_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(i, 5, kLT),
          Load::make(b_buf, {i}),
          Load::make(c_buf, {i - 5})));
  auto storeB = Store::make(
      b_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(i, 30, kLT),
          Load::make(c_buf, {i}),
          Load::make(d_buf, {i})));
  auto forI = For::make(i, 0, 50, Block::make({storeA, storeB}));
  auto par = Block::make({forI});

  LoopNest nest(par, {a_buf.node()});
  nest.optimizeConditionals();

  std::ostringstream oss;
  oss << *nest.root_stmt();
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i = 0; i < 5
# CHECK-NEXT: A[i] = B[i]
# CHECK-NEXT: B[i] = C[i]
# CHECK: for (int i = 0; i < 45
# CHECK-NEXT: A[i + 5] = C[i]
# CHECK-NEXT: B[i + 5] = IfThenElse(i + 5<30 ? 1 : 0, C[i + 5], D[i + 5])
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(LoopNest, OptimizeConditionalsOuterLoopVar) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 100; j++) {
  //       A[i] = IfThenElse(i<10, IfThenElse(i<5, B[i], C[i-5]), D[i-10])
  //     }
  //   }
  // Currently, this case where the condition variable `i` is not the
  // inner-most loop variable, is not optimized.

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle a_buf("A", {20}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle b_buf("B", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle c_buf("C", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle d_buf("D", {10}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto store = Store::make(
      a_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(i, 10, kLT),
          IfThenElse::make(
              CompareSelect::make(i, 5, kLT),
              Load::make(b_buf, {i}),
              Load::make(c_buf, {i - 5})),
          Load::make(d_buf, {i - 10})));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto forI = For::make(i, 0, 20, For::make(j, 0, 100, store));
  auto par = Block::make({forI});
  LoopNest nest(par, {a_buf.node()});

  HashProvider hasher;
  auto hash_before = hasher.hash(nest.root_stmt());
  nest.optimizeConditionals();
  auto hash_after = hasher.hash(nest.root_stmt());
  ASSERT_EQ(hash_before, hash_after);
}

TEST(LoopNest, OptimizeConditionalsCompValuesNotOrdered) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = IfThenElse(i<5, IfThenElse(i<10, B[i], C[i-5]), D[i-10])
  //   }
  // No optimization should be done here because one of the conditions use '>'.

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle a_buf("A", {20}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle b_buf("B", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle c_buf("C", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle d_buf("D", {10}, kInt);
  VarHandle i("i", kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto store = Store::make(
      a_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(i, 5, kLT),
          IfThenElse::make(
              CompareSelect::make(i, 10, kLT),
              Load::make(b_buf, {i}),
              Load::make(c_buf, {i - 5})),
          Load::make(d_buf, {i - 10})));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto forI = For::make(i, 0, 20, store);
  auto par = Block::make({forI});
  LoopNest nest(par, {a_buf.node()});

  HashProvider hasher;
  auto hash_before = hasher.hash(nest.root_stmt());
  nest.optimizeConditionals();
  auto hash_after = hasher.hash(nest.root_stmt());
  ASSERT_EQ(hash_before, hash_after);
}

TEST(LoopNest, OptimizeConditionalsCompValuesNotConstants) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = IfThenElse(i<N, IfThenElse(i<5, B[i], C[i-5]), D[i-10])
  //   }
  // No optimization should be done here because one of the conditions use '>'.

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle a_buf("A", {20}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle b_buf("B", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle c_buf("C", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle d_buf("D", {10}, kInt);
  VarHandle i("i", kInt);
  VarHandle N("N", kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto store = Store::make(
      a_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(i, N, kLT),
          IfThenElse::make(
              CompareSelect::make(i, 5, kLT),
              Load::make(b_buf, {i}),
              Load::make(c_buf, {i - 5})),
          Load::make(d_buf, {i - 10})));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto forI = For::make(i, 0, 20, store);
  auto par = Block::make({forI});
  LoopNest nest(par, {a_buf.node()});

  HashProvider hasher;
  auto hash_before = hasher.hash(nest.root_stmt());
  nest.optimizeConditionals();
  auto hash_after = hasher.hash(nest.root_stmt());
  ASSERT_EQ(hash_before, hash_after);
}

TEST(LoopNest, OptimizeConditionalsInvalidCondition) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = IfThenElse(i<10, IfThenElse(i>5, B[i], C[i-5]), D[i-10])
  //   }
  // No optimization should be done here because one of the conditions use '>'.

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle a_buf("A", {20}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle b_buf("B", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle c_buf("C", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle d_buf("D", {10}, kInt);
  VarHandle i("i", kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto store = Store::make(
      a_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(i, 10, kLT),
          IfThenElse::make(
              CompareSelect::make(i, 5, kGT),
              Load::make(b_buf, {i}),
              Load::make(c_buf, {i - 5})),
          Load::make(d_buf, {i - 10})));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto forI = For::make(i, 0, 20, store);
  auto par = Block::make({forI});
  LoopNest nest(par, {a_buf.node()});

  HashProvider hasher;
  auto hash_before = hasher.hash(nest.root_stmt());
  nest.optimizeConditionals();
  auto hash_after = hasher.hash(nest.root_stmt());
  ASSERT_EQ(hash_before, hash_after);
}

TEST(LoopNest, OptimizeConditionalsInvalidCondition2) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = IfThenElse(10<i, IfThenElse(i<5, B[i], C[i-5]), D[i-10])
  //   }
  // No optimization should be done here because of the invalid condition:
  //    "10 < i".

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle a_buf("A", {20}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle b_buf("B", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle c_buf("C", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle d_buf("D", {10}, kInt);
  VarHandle i("i", kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto store = Store::make(
      a_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(10, i, kLT),
          IfThenElse::make(
              CompareSelect::make(i, 5, kLT),
              Load::make(b_buf, {i}),
              Load::make(c_buf, {i - 5})),
          Load::make(d_buf, {i - 10})));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto forI = For::make(i, 0, 20, store);
  auto par = Block::make({forI});
  LoopNest nest(par, {a_buf.node()});

  HashProvider hasher;
  auto hash_before = hasher.hash(nest.root_stmt());
  nest.optimizeConditionals();
  auto hash_after = hasher.hash(nest.root_stmt());
  ASSERT_EQ(hash_before, hash_after);
}

TEST(LoopNest, OptimizeConditionalsInvalidCondition3) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = IfThenElse(i<10, IfThenElse(k<5, B[i], C[i-5]), D[i-10])
  //   }
  // No optimization should be done here because the conditions use different
  // variables: "i < 10" and "k < 5"

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle a_buf("A", {20}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle b_buf("B", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle c_buf("C", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle d_buf("D", {10}, kInt);
  VarHandle i("i", kInt);
  VarHandle k("k", kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto store = Store::make(
      a_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(i, 10, kLT),
          IfThenElse::make(
              CompareSelect::make(k, 5, kLT),
              Load::make(b_buf, {i}),
              Load::make(c_buf, {i - 5})),
          Load::make(d_buf, {i - 10})));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto forI = For::make(i, 0, 20, store);
  auto par = Block::make({forI});
  LoopNest nest(par, {a_buf.node()});

  HashProvider hasher;
  auto hash_before = hasher.hash(nest.root_stmt());
  nest.optimizeConditionals();
  auto hash_after = hasher.hash(nest.root_stmt());
  ASSERT_EQ(hash_before, hash_after);
}

TEST(LoopNest, OptimizeConditionalsInvalidCondition4) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = IfThenElse(k<10, IfThenElse(k<5, B[i], C[i-5]), D[i-10])
  //   }
  // No optimization should be done here because the conditions use the
  // variable 'k' which is not a loop variable.

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle a_buf("A", {20}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle b_buf("B", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle c_buf("C", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle d_buf("D", {10}, kInt);
  VarHandle i("i", kInt);
  VarHandle k("k", kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto store = Store::make(
      a_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(k, 10, kLT),
          IfThenElse::make(
              CompareSelect::make(k, 5, kLT),
              Load::make(b_buf, {i}),
              Load::make(c_buf, {i - 5})),
          Load::make(d_buf, {i - 10})));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto forI = For::make(i, 0, 20, store);
  auto par = Block::make({forI});
  LoopNest nest(par, {a_buf.node()});

  HashProvider hasher;
  auto hash_before = hasher.hash(nest.root_stmt());
  nest.optimizeConditionals();
  auto hash_after = hasher.hash(nest.root_stmt());
  ASSERT_EQ(hash_before, hash_after);
}

TEST(LoopNest, OptimizeConditionalsNotNormalized) {
  // Input IR:
  //   for (int i = 2; i < 20; i++) {
  //     A[i] = IfThenElse(i<5 ? 1 : 0, B[i], C[i-5])
  //   }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle a_buf("A", {20}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle b_buf("B", {5}, kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  BufHandle c_buf("C", {15}, kInt);
  VarHandle i("i", kInt);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto store = Store::make(
      a_buf,
      {i},
      IfThenElse::make(
          CompareSelect::make(i, 5, kLT),
          Load::make(b_buf, {i}),
          Load::make(c_buf, {i - 5})));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto forI = For::make(i, 2, 20, store);
  auto par = Block::make({forI});
  LoopNest nest(par, {a_buf.node()});

  HashProvider hasher;
  auto hash_before = hasher.hash(nest.root_stmt());
  nest.optimizeConditionals();
  auto hash_after = hasher.hash(nest.root_stmt());
  ASSERT_EQ(hash_before, hash_after);
}

static std::pair<BufHandle, Tensor> colReduce(int M, int N) {
  BufHandle a("a", {M, N}, kFloat);
  Tensor t = Reduce(
      "b",
      {N},
      Sum(),
      [&](const VarHandle& n, const VarHandle& m) { return a.load(m, n); },
      {M});
  return {a, Tensor(t.buf(), LoopNest::sanitizeNames(t.stmt()))};
}

static StmtPtr splitTailReorder(Tensor b) {
  constexpr int kVectorWidth = 8;
  LoopNest nest({b});
  auto loops = nest.getAllLoopNestsWritingToBuf(b.buf())[0];
  nest.splitWithTail(loops[0], kVectorWidth);
  // Now the loopnests will look like:
  //
  // for (int i_outer = 0; ...
  //   for (int i_inner = 0; ...
  //     b[i_outer * 8 + i_inner] = float(0);
  //     for (int j = 0; ...
  //       b[i_outer * 8 + i_inner] = ReduceOp(...);
  //
  // for (int i_tail = 0; ...
  //   b[i_tail + ((100 - 0) / 8) * 8] = float(0);
  //   for (int j = 0; ...
  //     b[i_tail + ((100 - 0) / 8) * 8] = ReduceOp(...);
  //
  // Since there are 4 writes to b, we will get 4 loopnests from the
  // call to `getAllLoopNestsWritingToBuf` below.
  //
  // Write #2: "b[i_outer * 8 + i_inner] = ReduceOp(...)"
  // Loopnest #2: {i_outer, i_inner, j};
  // We will have to reorder i_inner and j.
  auto loopnests = nest.getAllLoopNestsWritingToBuf(b.buf());
  LoopNest::reorderAxis(loopnests[1][1], loopnests[1][2]);
  nest.prepareForCodegen();
  return nest.root_stmt();
}

static StmtPtr splitMaskReorder(Tensor b) {
  constexpr int kVectorWidth = 8;
  LoopNest nest({b});
  auto loops = nest.getAllLoopNestsWritingToBuf(b.buf())[1];
  nest.splitWithMask(loops[0], kVectorWidth);
  loops = nest.getAllLoopNestsWritingToBuf(b.buf())[1];
  LoopNest::reorderAxis(loops[1], loops[2]);
  nest.prepareForCodegen();
  return nest.root_stmt();
}

static void checkColReduce(StmtPtr s, BufHandle p, Tensor t) {
  int M = immediateAs<int>(p.dim(0));
  int N = immediateAs<int>(p.dim(1));
  PaddedBuffer<float> a(M, N);
  PaddedBuffer<float> b(N);
  PaddedBuffer<float> ref(N);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      a(i, j) = 1.0f;
    }
  }
  for (int i = 0; i < N; i++) {
    b(i) = 0.0f;
  }
  for (int i = 0; i < N; i++) {
    ref(i) = 76.0f;
  }
  SimpleIREvaluator(s, {p, t}).call({a, b});
  ExpectAllNear(b, ref, 1e-5);
}

TEST(LoopNest, ColReduceSplitTailEvenReorder) {
  constexpr int M = 76, N = 128;
  auto p = colReduce(M, N);
  StmtPtr s = splitTailReorder(p.second);

  std::ostringstream oss;
  oss << *s;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i_outer
# CHECK-NEXT: for (int i_inner
# CHECK-NEXT: b[
# CHECK: for (int j
# CHECK-NEXT: for (int i_inner
# CHECK-NEXT: b[
# CHECK-NOT: for (
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  checkColReduce(s, p.first, p.second);
}

TEST(LoopNest, ColReduceSplitTailUnevenReorder) {
  constexpr int M = 76, N = 100;
  auto p = colReduce(M, N);
  StmtPtr s = splitTailReorder(p.second);

  std::ostringstream oss;
  oss << *s;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i_outer
# CHECK-NEXT: for (int i_inner
# CHECK-NEXT: b[
# CHECK: for (int j
# CHECK-NEXT: for (int i_inner
# CHECK-NEXT: b[
# CHECK: for (int i_tail
# CHECK-NEXT: b[
# CHECK-NEXT: for (int j
# CHECK-NEXT: b[
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  checkColReduce(s, p.first, p.second);
}

TEST(LoopNest, ColReduceSplitMaskEvenReorder) {
  constexpr int M = 76, N = 128;
  auto p = colReduce(M, N);
  StmtPtr s = splitMaskReorder(p.second);
  checkColReduce(s, p.first, p.second);
}

TEST(LoopNest, ColReduceSplitMaskUnevenReorder) {
  constexpr int M = 76, N = 100;
  auto p = colReduce(M, N);
  StmtPtr s = splitMaskReorder(p.second);
  checkColReduce(s, p.first, p.second);
}

TEST(LoopNest, ReorderAxisWithMultipleConds) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     if i > 5 {
  //       if i < 10 {
  //         for (int j = 0; j < 100; j++) {
  //           A[i] = i * j;
  //         }
  //       }
  //     }
  //   }
  BufHandle a_buf("A", {20}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {i}, Mul::make(i, j)));
  auto inner_cond = Cond::make(CompareSelect::make(i, 10, kLT), forJ, nullptr);
  auto outer_cond =
      Cond::make(CompareSelect::make(i, 5, kGT), inner_cond, nullptr);
  auto forI = For::make(i, 0, 20, outer_cond);
  StmtPtr par = Block::make({forI});
  LoopNest l(par, {a_buf.node()});
  LoopNest::reorderAxis(forI, forJ);
  ASSERT_EQ(par, l.root_stmt());
  par = IRSimplifier::simplify(par);

  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int j
# CHECK-NEXT: for (int i
# CHECK-NEXT: if (i>5
# CHECK-NEXT: if (i<10
# CHECK-NEXT: A[i] = i * j
# CHECK-NOT: for (
      )IR";
  std::ostringstream oss;
  oss << *par;
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(LoopNest, VectorizeUse) {
  constexpr int N = 8;
  BufHandle a("a", {N}, kFloat);
  Tensor b =
      Compute("b", {N}, [&](const VarHandle& n) { return a.load(n) + 1.0f; });
  Tensor c =
      Compute("c", {N}, [&](const VarHandle& n) { return b.load(n) + 2.0f; });
  LoopNest nest({c}, {b, c});
  auto loops = nest.getAllLoopNestsWritingToBuf(b.buf())[0];
  ASSERT_TRUE(LoopNest::vectorize(loops[0]));
  loops = nest.getAllLoopNestsWritingToBuf(c.buf())[0];
  ASSERT_TRUE(LoopNest::vectorize(loops[0]));
  nest.prepareForCodegen();
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  StmtPtr s = nest.root_stmt();
  std::ostringstream oss;
  oss << *nest.root_stmt();
  torch::jit::testing::FileCheck().run(
      R"IR(
# CHECK: c[Ramp
)IR",
      oss.str());
}

const char* int64Loop = R"IR(
# CHECK: for (int64_t i = 0ll; i < 12ll; i++) {
# CHECK:   b[i] = (a[i]) + 1ll;
# CHECK: }
)IR";

TEST(LoopNest, Int64Direct) {
  constexpr int64_t N = 12;
  BufHandle a("a", {N}, kLong);
  BufHandle b("b", {N}, kLong);
  VarHandle n("i", kLong);
  StmtPtr s = For::make(
      n, LongImm::make(0l), N, b.store({n}, a.load({n}) + LongImm::make(1l)));
  s = IRSimplifier::simplify(s);
  std::ostringstream oss;
  oss << *s;
  torch::jit::testing::FileCheck().run(int64Loop, oss.str());
}

TEST(LoopNest, Int64Compute) {
  constexpr int64_t N = 12;
  BufHandle a("a", {N}, kLong);
  Tensor b = Compute("b", {N}, [&](const VarHandle& n) {
    return a.load(n) + LongImm::make(1l);
  });
  LoopNest nest({b});
  nest.prepareForCodegen();
  nest.simplify();
  std::ostringstream oss;
  oss << *nest.root_stmt();
  torch::jit::testing::FileCheck().run(int64Loop, oss.str());
}

TEST(LoopNest, DistributeLoopWithAllStmtsAsPivots) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = 0;
  //     for (int j = 0; j < 100; j++) {
  //       A[i] = A[i] + i * j;
  //     }
  //     B[i] = A[i];
  //     for (int k = 0; k < 50; k++) {
  //       B[i] = B[i] + i * k;
  //     }
  //   }
  BufHandle a_buf("A", {20}, kInt);
  BufHandle b_buf("B", {20}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto initA = Store::make(a_buf, {i}, 0);
  auto forJ = For::make(
      j,
      0,
      100,
      Store::make(
          a_buf, {i}, Add::make(Load::make(a_buf, {i}), Mul::make(i, j))));
  auto initB = Store::make(b_buf, {i}, Load::make(a_buf, {i}));
  auto forK = For::make(
      k,
      0,
      50,
      Store::make(
          b_buf, {i}, Add::make(Load::make(b_buf, {i}), Mul::make(i, k))));
  auto forI = For::make(i, 0, 20, Block::make({initA, forJ, initB, forK}));
  auto par = Block::make({forI});

  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: A[i] = 0
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i] =
# CHECK: for (int i
# CHECK-NEXT: B[i] = A[i]
# CHECK: for (int i
# CHECK-NEXT: for (int k
# CHECK-NEXT: B[i] =
# CHECK-NOT: for (
      )IR";

  LoopNest nest(par, {a_buf.node(), b_buf.node()});
  auto new_loops = LoopNest::distributeLoop(forI, {initA, forJ, initB});

  std::ostringstream oss;
  oss << *par;
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // The first loop after distribution must be same as the original For.
  ASSERT_EQ(new_loops.front(), forI);
}

TEST(LoopNest, DistributeLoopWithOneStmtAsPivot) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = 0;
  //     for (int j = 0; j < 100; j++) {
  //       A[i] = A[i] + i * j;
  //     }
  //     B[i] = A[i];
  //     for (int k = 0; k < 50; k++) {
  //       B[i] = B[i] + i * k;
  //     }
  //   }
  BufHandle a_buf("A", {20}, kInt);
  BufHandle b_buf("B", {20}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto initA = Store::make(a_buf, {i}, 0);
  auto forJ = For::make(
      j,
      0,
      100,
      Store::make(
          a_buf, {i}, Add::make(Load::make(a_buf, {i}), Mul::make(i, j))));
  auto initB = Store::make(b_buf, {i}, Load::make(a_buf, {i}));
  auto forK = For::make(
      k,
      0,
      50,
      Store::make(
          b_buf, {i}, Add::make(Load::make(b_buf, {i}), Mul::make(i, k))));
  auto forI = For::make(i, 0, 20, Block::make({initA, forJ, initB, forK}));
  auto par = Block::make({forI});

  LoopNest nest(par, {a_buf.node(), b_buf.node()});
  auto new_loops = LoopNest::distributeLoop(forI, {forJ});

  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: A[i] = 0
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i] =
# CHECK: for (int i
# CHECK-NEXT: B[i] = A[i]
# CHECK-NEXT: for (int k
# CHECK-NEXT: B[i] =
# CHECK-NOT: for (
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // The first loop after distribution must be same as the original For.
  ASSERT_EQ(new_loops.front(), forI);
}

TEST(LoopNest, DistributeLoopWithoutAnyPivot) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = 0;
  //     for (int j = 0; j < 100; j++) {
  //       A[i] = A[i] + i * j;
  //     }
  //     B[i] = A[i];
  //     for (int k = 0; k < 50; k++) {
  //       B[i] = B[i] + i * k;
  //     }
  //   }
  BufHandle a_buf("A", {20}, kInt);
  BufHandle b_buf("B", {20}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto initA = Store::make(a_buf, {i}, 0);
  auto forJ = For::make(
      j,
      0,
      100,
      Store::make(
          a_buf, {i}, Add::make(Load::make(a_buf, {i}), Mul::make(i, j))));
  auto initB = Store::make(b_buf, {i}, Load::make(a_buf, {i}));
  auto forK = For::make(
      k,
      0,
      50,
      Store::make(
          b_buf, {i}, Add::make(Load::make(b_buf, {i}), Mul::make(i, k))));
  auto forI = For::make(i, 0, 20, Block::make({initA, forJ, initB, forK}));
  auto par = Block::make({forI});

  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: A[i] = 0
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i] =
# CHECK: for (int i
# CHECK-NEXT: B[i] = A[i]
# CHECK: for (int i
# CHECK-NEXT: for (int k
# CHECK-NEXT: B[i] =
# CHECK-NOT: for (
      )IR";

  LoopNest nest(par, {a_buf.node(), b_buf.node()});
  auto new_loops = LoopNest::distributeLoop(forI);

  std::ostringstream oss;
  oss << *par;
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // The first loop after distribution must be same as the original For.
  ASSERT_EQ(new_loops.front(), forI);
}

TEST(LoopNest, DistributeLoopOverInnerLoops) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = 0;
  //     for (int j = 0; j < 100; j++) {
  //       A[i] = A[i] + i * j;
  //     }
  //     B[i] = A[i];
  //     for (int k = 0; k < 50; k++) {
  //       B[i] = B[i] + i * k;
  //     }
  //   }
  BufHandle a_buf("A", {20}, kInt);
  BufHandle b_buf("B", {20}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto initA = Store::make(a_buf, {i}, 0);
  auto forJ = For::make(
      j,
      0,
      100,
      Store::make(
          a_buf, {i}, Add::make(Load::make(a_buf, {i}), Mul::make(i, j))));
  auto initB = Store::make(b_buf, {i}, Load::make(a_buf, {i}));
  auto forK = For::make(
      k,
      0,
      50,
      Store::make(
          b_buf, {i}, Add::make(Load::make(b_buf, {i}), Mul::make(i, k))));
  auto forI = For::make(i, 0, 20, Block::make({initA, forJ, initB, forK}));
  auto par = Block::make({forI});

  LoopNest nest(par, {a_buf.node(), b_buf.node()});
  auto new_loops = LoopNest::distributeLoopOverInnerLoops(forI);

  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: A[i] = 0
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i] =
# CHECK: for (int i
# CHECK-NEXT: B[i] = A[i]
# CHECK-NEXT: for (int k
# CHECK-NEXT: B[i] =
# CHECK-NOT: for (
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // The first loop after distribution must be same as the original For.
  ASSERT_EQ(new_loops.front(), forI);
}

TEST(LoopNest, DistributeLoopAndParentsWithoutAnyPivot) {
  // Input IR:
  // for (int m = 0; m < 50; m++) {
  //   for (int i = 0; i < 20; i++) {
  //     A[m,i] = 0;
  //     for (int j = 0; j < 100; j++) {
  //       A[m,i] = A[m,i] + i * j;
  //     }
  //     B[m,i] = A[m,i];
  //     for (int k = 0; k < 50; k++) {
  //       B[m,i] = B[m,i] + i * k;
  //     }
  //   }
  // }
  BufHandle a_buf("A", {100, 100}, kInt);
  BufHandle b_buf("B", {100, 100}, kInt);
  VarHandle m("m", kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto initA = Store::make(a_buf, {m, i}, 0);
  auto forJ = For::make(
      j,
      0,
      100,
      Store::make(
          a_buf,
          {m, i},
          Add::make(Load::make(a_buf, {m, i}), Mul::make(i, j))));
  auto initB = Store::make(b_buf, {m, i}, Load::make(a_buf, {m, i}));
  auto forK = For::make(
      k,
      0,
      50,
      Store::make(
          b_buf,
          {m, i},
          Add::make(Load::make(b_buf, {m, i}), Mul::make(i, k))));
  auto forI = For::make(i, 0, 20, Block::make({initA, forJ, initB, forK}));

  {
    // Check the case of distributing loop and its parents over all the
    // statements in the loop.
    const std::string& verification_pattern =
        R"IR(
# CHECK: for (int m
# CHECK-NEXT: for (int i
# CHECK-NEXT: A[m, i] = 0
# CHECK: for (int m
# CHECK-NEXT: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[m, i] =
# CHECK: for (int m
# CHECK-NEXT: for (int i
# CHECK-NEXT: B[m, i] = A[m, i]
# CHECK: for (int m
# CHECK-NEXT: for (int i
# CHECK-NEXT: for (int k
# CHECK-NEXT: B[m, i] =
# CHECK-NOT: for (
        )IR";

    auto newForI = to<For>(Stmt::clone(forI));
    auto forM = For::make(m, 0, 50, newForI);
    auto par = Block::make({forM});
    LoopNest nest(par, {a_buf.node(), b_buf.node()});
    auto newLoops = LoopNest::distributeLoopAndParents(newForI);

    std::ostringstream oss;
    oss << *par;
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    // The first loop after distribution must be same as the original For.
    ASSERT_EQ(newLoops.front(), forM);
  }

  {
    // Check the case of distributing loop and its parents over all the inner
    // loops.
    const std::string& verification_pattern =
        R"IR(
# CHECK: for (int m
# CHECK-NEXT: for (int i
# CHECK-NEXT: A[m, i] = 0
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[m, i] =
# CHECK: for (int m
# CHECK-NEXT: for (int i
# CHECK-NEXT: B[m, i] = A[m, i]
# CHECK-NEXT: for (int k
# CHECK-NEXT: B[m, i] =
# CHECK-NOT: for (
        )IR";

    auto newForI = to<For>(Stmt::clone(forI));
    auto forM = For::make(m, 0, 50, newForI);
    auto par = Block::make({forM});
    LoopNest nest(par, {a_buf.node(), b_buf.node()});
    auto newLoops = LoopNest::distributeLoopAndParentsOverInnerLoops(newForI);

    std::ostringstream oss;
    oss << *par;
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    // The first loop after distribution must be same as the original For.
    ASSERT_EQ(newLoops.front(), forM);
  }
}

TEST(LoopNest, fuseLoopsSimple) {
  // Input IR:
  //   for (int j = 0; j < 100; j++) {
  //     A[j] = 10 * j;
  //   }
  //   for (int k = 0; k < 100; k++) {
  //     B[k] = 20 * k;
  //   }
  BufHandle a_buf("A", {100}, kInt);
  BufHandle b_buf("B", {100}, kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  auto forK = For::make(k, 0, 100, Store::make(b_buf, {k}, Mul::make(20, k)));
  auto par = Block::make({forJ, forK});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_TRUE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));

  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int j
# CHECK-NEXT: A[j] =
# CHECK-NEXT: B[j] =
# CHECK-NOT: for (
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // The fused loop must be the same as the first loop.
  ASSERT_EQ(fused_loop, forJ);
}

TEST(LoopNest, fuseLoopsMultiple) {
  // Input IR:
  //   for (int i = 0; i < 100; i++) {
  //     A[i+100] = 20 + i;
  //   }
  //   for (int j = 0; j < 100; j++) {
  //     A[j] = 10 * j;
  //   }
  //   for (int k = 0; k < 100; k++) {
  //     B[k] = 20 * k;
  //   }
  BufHandle a_buf("A", {200}, kInt);
  BufHandle b_buf("B", {100}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto forI =
      For::make(i, 0, 100, Store::make(a_buf, {i + 100}, Add::make(20, i)));
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  auto forK = For::make(k, 0, 100, Store::make(b_buf, {k}, Mul::make(20, k)));
  auto par = Block::make({forI, forJ, forK});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_TRUE(LoopNest::fuseLoops({forI, forJ, forK}, &fused_loop));

  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: A[i + 100] =
# CHECK-NEXT: A[i] =
# CHECK-NEXT: B[i] =
# CHECK-NOT: for (
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // The fused loop must be the same as the first loop.
  ASSERT_EQ(fused_loop, forI);
}

TEST(LoopNest, fuseLoopsNested) {
  // Input IR:
  //   for (int m = 0; m < 20; m++) {
  //     A[m] = 0;
  //     for (int j = 0; j < 100; j++) {
  //       A[m] = A[m] + m * j;
  //     }
  //   }
  //   for (int n = 0; n < 20; n++) {
  //     B[n] = A[n];
  //     for (int k = 0; k < 50; k++) {
  //       B[n] = B[n] + n * k;
  //     }
  //   }
  BufHandle a_buf("A", {20, 100}, kInt);
  BufHandle b_buf("B", {20, 100}, kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto initA = Store::make(a_buf, {m}, 0);
  auto forJ = For::make(
      j,
      0,
      100,
      Store::make(
          a_buf, {m}, Add::make(Load::make(a_buf, {m}), Mul::make(m, j))));
  auto initB = Store::make(b_buf, {n}, Load::make(a_buf, {n}));
  auto forK = For::make(
      k,
      0,
      50,
      Store::make(
          b_buf, {n}, Add::make(Load::make(b_buf, {n}), Mul::make(n, k))));
  auto forM = For::make(m, 0, 20, Block::make({initA, forJ}));
  auto forN = For::make(n, 0, 20, Block::make({initB, forK}));
  auto par = Block::make({forM, forN});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_TRUE(LoopNest::fuseLoops({forM, forN}, &fused_loop));

  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int m
# CHECK-NEXT: A[m] = 0
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[m] =
# CHECK: B[m] = A[m]
# CHECK-NEXT: for (int k
# CHECK-NEXT: B[m] =
# CHECK-NOT: for (
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // The fused loop must be the same as the first loop.
  ASSERT_EQ(fused_loop, forM);
}

TEST(LoopNest, fuseLoopsNested2D) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 100; j++) {
  //       A[i,j] = i * j * 500;
  //     }
  //   }
  //   for (int m = 0; m < 20; m++) {
  //     for (int n = 0; n < 50; n++) {
  //       B[m,n] = m + n * 100;
  //     }
  //   }
  BufHandle a_buf("A", {20, 100}, kInt);
  BufHandle b_buf("B", {20, 100}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  auto forI = For::make(
      i,
      0,
      20,
      For::make(
          j,
          0,
          100,
          Store::make(a_buf, {i, j}, Mul::make(Mul::make(i, j), 500))));
  auto forM = For::make(
      m,
      0,
      20,
      For::make(
          n,
          0,
          50,
          Store::make(b_buf, {m, n}, Add::make(m, Mul::make(n, 100)))));
  auto par = Block::make({forI, forM});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_TRUE(LoopNest::fuseLoops({forI, forM}, &fused_loop));

  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i, j] =
# CHECK: for (int n
# CHECK-NEXT: B[i, n] =
# CHECK-NOT: for (
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // The fused loop must be the same as the first loop.
  ASSERT_EQ(fused_loop, forI);
}

TEST(LoopNest, fuseLoopsNested2DInner) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 100; j++) {
  //       A[i,j] = i * j * 500;
  //     }
  //     for (int n = 0; n < 100; n++) {
  //       B[i,n] = m + n * 100;
  //     }
  //   }
  BufHandle a_buf("A", {20, 100}, kInt);
  BufHandle b_buf("B", {20, 100}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle n("n", kInt);
  auto forJ = For::make(
      j, 0, 100, Store::make(a_buf, {i, j}, Mul::make(Mul::make(i, j), 500)));
  auto forN = For::make(
      n, 0, 100, Store::make(b_buf, {i, n}, Add::make(i, Mul::make(n, 100))));
  auto forI = For::make(i, 0, 20, Block::make({forJ, forN}));
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_TRUE(LoopNest::fuseLoops({forJ, forN}, &fused_loop));

  std::ostringstream oss;
  oss << *forI;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i, j] =
# CHECK-NEXT: B[i, j] =
# CHECK-NOT: for (
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // The fused loop must be the same as the first loop.
  ASSERT_EQ(fused_loop, forJ);
}

TEST(LoopNest, fuseLoopsDifferentStopBounds) {
  // Input IR:
  //   for (int j = 0; j < 100; j++) {
  //     A[j] = 10 * j;
  //   }
  //   for (int k = 0; k < 50; k++) {
  //     B[k] = 20 * k;
  //   }
  BufHandle a_buf("A", {100}, kInt);
  BufHandle b_buf("B", {100}, kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  auto forK = For::make(k, 0, 50, Store::make(b_buf, {j}, Mul::make(20, k)));
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  auto par = Block::make({forJ, forK});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));
}

TEST(LoopNest, fuseLoopsDifferentStartBounds) {
  // Input IR:
  //   for (int j = 0; j < 100; j++) {
  //     A[j] = 10 * j;
  //   }
  //   for (int k = 50; k < 100; k++) {
  //     B[k] = 20 * k;
  //   }
  BufHandle a_buf("A", {100}, kInt);
  BufHandle b_buf("B", {100}, kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  auto forK = For::make(k, 50, 100, Store::make(b_buf, {j}, Mul::make(20, k)));
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  auto par = Block::make({forJ, forK});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));
}

TEST(LoopNest, fuseLoopsNotContiguous) {
  // Input IR:
  //   for (int j = 0; j < 100; j++) {
  //     A[j] = 10 * j;
  //   }
  //   B[0] = 0;
  //   for (int k = 0; k < 100; k++) {
  //     B[k] = 20 * k;
  //   }
  BufHandle a_buf("A", {100}, kInt);
  BufHandle b_buf("B", {100}, kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  auto initB = Store::make(b_buf, {0}, 0);
  auto forK = For::make(k, 0, 100, Store::make(b_buf, {j}, Mul::make(20, k)));
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  auto par = Block::make({forJ, initB, forK});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));
}

TEST(LoopNest, fuseLoopsWithDifferentParents) {
  // Input IR:
  //   for (int i = 0; i < 50; i++) {
  //     for (int j = 0; j < 100; j++) {
  //       A[i,j] = i * j;
  //     }
  //   }
  //   B[0] = 0;
  //   for (int k = 50; k < 100; k++) {
  //     B[k] = 20 * k;
  //   }
  BufHandle a_buf("A", {50, 100}, kInt);
  BufHandle b_buf("B", {100}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {i, j}, Mul::make(i, j)));
  auto forI = For::make(i, 0, 50, forJ);
  auto initB = Store::make(b_buf, {0}, 0);
  auto forK = For::make(k, 50, 100, Store::make(b_buf, {j}, Mul::make(20, k)));
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  auto par = Block::make({forI, initB, forK});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));
}

TEST(LoopNest, fuseLoopsWithVariableBounds) {
  // Input IR:
  //   for (int j = 0; j < N; j++) {
  //     A[j] = 10 * j;
  //   }
  //   for (int k = 0; k < N; k++) {
  //     B[k] = 20 * k;
  //   }
  BufHandle a_buf("A", {20}, kInt);
  BufHandle b_buf("B", {20}, kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  VarHandle N("N", kInt);
  auto forJ = For::make(j, 0, N, Store::make(a_buf, {j}, Mul::make(10, j)));
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks,cppcoreguidelines-avoid-magic-numbers)
  auto forK = For::make(k, 0, N, Store::make(b_buf, {j}, Mul::make(20, k)));
  auto par = Block::make({forJ, forK});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_TRUE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));

  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int j
# CHECK-NEXT: A[j] =
# CHECK-NEXT: B[j] =
# CHECK-NOT: for (
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // The fused loop must be the same as the first loop.
  ASSERT_EQ(fused_loop, forJ);
}

TEST(LoopNest, fuseLoopsWithExprBounds) {
  // Input IR:
  //   for (int j = 0; j < M + N; j++) {
  //     A[j] = 10 * j;
  //   }
  //   for (int k = 0; k < M + N; k++) {
  //     B[k] = 20 * k;
  //   }
  BufHandle a_buf("A", {20}, kInt);
  BufHandle b_buf("B", {20}, kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  VarHandle M("M", kInt);
  VarHandle N("N", kInt);
  auto forJ = For::make(j, 0, M + N, Store::make(a_buf, {j}, Mul::make(10, j)));
  auto forK = For::make(k, 0, M + N, Store::make(b_buf, {j}, Mul::make(20, k)));
  auto par = Block::make({forJ, forK});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_TRUE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));

  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int j
# CHECK-NEXT: A[j] =
# CHECK-NEXT: B[j] =
# CHECK-NOT: for (
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // The fused loop must be the same as the first loop.
  ASSERT_EQ(fused_loop, forJ);
}

TEST(LoopNest, fuseLoopsWithDifferentExprBounds) {
  // Input IR:
  //   for (int j = M; j < N * 2; j++) {
  //     A[j] = 10 * j;
  //   }
  //   for (int k = M; k < N + N; k++) {
  //     B[k] = 20 * k;
  //   }
  BufHandle a_buf("A", {20}, kInt);
  BufHandle b_buf("B", {20}, kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  VarHandle M("M", kInt);
  VarHandle N("N", kInt);
  auto forJ = For::make(j, M, N * 2, Store::make(a_buf, {j}, Mul::make(10, j)));
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks,cppcoreguidelines-avoid-magic-numbers)
  auto forK = For::make(k, M, N + N, Store::make(b_buf, {j}, Mul::make(20, k)));
  auto par = Block::make({forJ, forK});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_TRUE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));

  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int j
# CHECK-NEXT: A[j] =
# CHECK-NEXT: B[j] =
# CHECK-NOT: for (
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // The fused loop must be the same as the first loop.
  ASSERT_EQ(fused_loop, forJ);
}

TEST(LoopNest, fuseLoopsWithNonOverlappingBufferAccesses) {
  // Input IR:
  //   for (int j = 10; j < 100; j++) {
  //     A[j] = 10 * j;
  //   }
  //   for (int k = 10; k < 100; k++) {
  //     A[k+100] = 30 * k
  //   }
  BufHandle a_buf("A", {200}, kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto forJ = For::make(j, 10, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  auto forK =
      For::make(k, 10, 100, Store::make(a_buf, {k + 100}, Mul::make(30, k)));
  auto par = Block::make({forJ, forK});

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_TRUE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));

  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int j
# CHECK-NEXT: A[j] =
# CHECK-NEXT: A[j + 100] =
# CHECK-NOT: for (
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // The fused loop must be the same as the first loop.
  ASSERT_EQ(fused_loop, forJ);
}

TEST(LoopNest, fuseLoopsWithNonOverlapping2DBufferAccesses) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 100; j++) {
  //       A[i,j] = i * j * 500;
  //     }
  //   }
  //   for (int m = 0; m < 20; m++) {
  //     for (int n = 0; n < 50; n++) {
  //       A[m+20,n+100] = m + n * 100;
  //     }
  //   }
  BufHandle a_buf("A", {20, 100}, kInt);
  BufHandle b_buf("B", {20, 50}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  auto storeA1 = Store::make(a_buf, {i, j}, Mul::make(Mul::make(i, j), 500));
  auto forJ = For::make(j, 0, 100, storeA1);
  auto forI = For::make(i, 0, 20, forJ);
  auto storeA2 =
      Store::make(a_buf, {m + 20, n + 100}, Add::make(m, Mul::make(n, 100)));
  auto forN = For::make(n, 0, 50, storeA2);
  auto forM = For::make(m, 0, 20, forN);
  auto par = Block::make({forI, forM});

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_TRUE(LoopNest::fuseLoops({forI, forM}, &fused_loop));

  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i, j] =
# CHECK: for (int n
# CHECK-NEXT: A[i + 20, n + 100] =
# CHECK-NOT: for (
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // The fused loop must be the same as the first loop.
  ASSERT_EQ(fused_loop, forI);
}

TEST(LoopNest, fuseLoopsWithReductions) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = 0
  //     for (int j = 0; j < 100; j++) {
  //       A[i] = A[i] + B[i,j];
  //     }
  //   }
  //   for (int m = 0; m < 20; m++) {
  //     C[m] = A[m];
  //   }
  BufHandle a_buf("A", {20}, kInt);
  BufHandle b_buf("B", {20, 100}, kInt);
  BufHandle c_buf("C", {20}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle m("m", kInt);
  auto initA = Store::make(a_buf, {i}, 0);
  auto sumA = Store::make(
      a_buf, {i}, Add::make(Load::make(a_buf, {i}), Load::make(b_buf, {i, j})));
  auto forJ = For::make(j, 0, 100, sumA);
  auto forI = For::make(i, 0, 20, Block::make({initA, forJ}));
  auto forM =
      For::make(m, 0, 20, Store::make(c_buf, {m}, Load::make(a_buf, {m})));
  auto par = Block::make({forI, forM});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_TRUE(LoopNest::fuseLoops({forI, forM}, &fused_loop));

  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: A[i] =
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i] = (A[i]) +
# CHECK-NOT: for (
# CHECK: C[i] = A[i]
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // The fused loop must be the same as the first loop.
  ASSERT_EQ(fused_loop, forI);
}

TEST(LoopNest, fuseLoopsWith2DReductions) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 50; j++) {
  //       A[i,j] = 0
  //       for (int k = 0; k < 100; k++) {
  //         A[i,j] = A[i,j] + B[i,j,k];
  //       }
  //     }
  //   }
  //   for (int m = 0; m < 20; m++) {
  //     for (int n = 0; n < 40; n++) {
  //       C[m,n] = A[m,n];
  //     }
  //   }
  BufHandle a_buf("A", {20, 50}, kInt);
  BufHandle b_buf("B", {20, 50, 100}, kInt);
  BufHandle c_buf("C", {20, 40}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  auto initA = Store::make(a_buf, {i, j}, 0);
  auto sumA = Store::make(
      a_buf,
      {i, j},
      Add::make(Load::make(a_buf, {i, j}), Load::make(b_buf, {i, j, k})));
  auto forK = For::make(k, 0, 100, sumA);
  auto forJ = For::make(j, 0, 50, Block::make({initA, forK}));
  auto forI = For::make(i, 0, 20, forJ);
  auto storeC = Store::make(c_buf, {m, n}, Load::make(a_buf, {m, n}));
  auto forM = For::make(m, 0, 20, For::make(n, 0, 40, storeC));
  auto par = Block::make({forI, forM});

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_TRUE(LoopNest::fuseLoops({forI, forM}, &fused_loop));

  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i, j] =
# CHECK-NEXT: for (int k
# CHECK-NEXT: A[i, j] = (A[i, j]) +
# CHECK: for (int n
# CHECK-NEXT: C[i, n] = A[i, n]
# CHECK-NOT: for (
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // The fused loop must be the same as the first loop.
  ASSERT_EQ(fused_loop, forI);
}

TEST(LoopNest, fuseLoopsWithComplexIndices) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 20; j++) {
  //       A[i,j*20+j+2] = i + j;
  //     }
  //   }
  //   for (int m = 0; m < 20; m++) {
  //     for (int n = 0; n < 20; n++) {
  //       B[m,n] = A[m,n*20+n+2];
  //     }
  //   }
  BufHandle a_buf("A", {20, 400}, kInt);
  BufHandle b_buf("B", {20, 400}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  auto writeA = Store::make(a_buf, {i, j * 20 + j + 2}, i + j);
  auto forI = For::make(i, 0, 20, For::make(j, 0, 20, writeA));
  auto storeB =
      Store::make(b_buf, {m, n}, Load::make(a_buf, {m, n * 20 + n + 2}));
  auto forM = For::make(m, 0, 20, For::make(n, 0, 20, storeB));
  auto par = Block::make({forI, forM});

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_TRUE(LoopNest::fuseLoops({forI, forM}, &fused_loop));

  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i, (j * 20 + j) + 2] = i + j
# CHECK: for (int n
# CHECK-NEXT: B[i, n] = A[i, (n * 20 + n) + 2]
# CHECK-NOT: for (
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // The fused loop must be the same as the first loop.
  ASSERT_EQ(fused_loop, forI);
}

TEST(LoopNest, fuseLoopsWithMixedLoopVarsAsIndices) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 20; j++) {
  //       A[i,i*20+j] = i + j;
  //     }
  //   }
  //   for (int m = 0; m < 20; m++) {
  //     for (int n = 0; n < 20; n++) {
  //       B[m,n] = A[m,m*20+n];  // Both indices of A use m
  //     }
  //   }
  BufHandle a_buf("A", {20, 500}, kInt);
  BufHandle b_buf("B", {20, 500}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  auto writeA = Store::make(a_buf, {i, i * 20 + j}, i + j);
  auto forI = For::make(i, 0, 20, For::make(j, 0, 20, writeA));
  auto storeB = Store::make(b_buf, {m, n}, Load::make(a_buf, {m, m * 20 + n}));
  auto forM = For::make(m, 0, 20, For::make(n, 0, 20, storeB));
  auto par = Block::make({forI, forM});

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forI, forM}, &fused_loop));
}

TEST(LoopNest, fuseLoopsWithTranspose) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 20; j++) {
  //       A[i,j] = i + j;
  //     }
  //   }
  //   for (int m = 0; m < 20; m++) {
  //     for (int n = 0; n < 20; n++) {
  //       B[m,n] = A[n,m];  // Transpose
  //     }
  //   }
  BufHandle a_buf("A", {20, 20}, kInt);
  BufHandle b_buf("B", {20, 20}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  auto writeA = Store::make(a_buf, {i, j}, i + j);
  auto forI = For::make(i, 0, 20, For::make(j, 0, 20, writeA));
  auto storeB = Store::make(b_buf, {m, n}, Load::make(a_buf, {n, m}));
  auto forM = For::make(m, 0, 20, For::make(n, 0, 20, storeB));
  auto par = Block::make({forI, forM});

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forI, forM}, &fused_loop));
}

TEST(LoopNest, fuseLoopsThatViolateDependencies1) {
  // Input IR:
  //   for (int j = 10; j < 100; j++) {
  //     A[j] = 10 * j;
  //   }
  //   for (int k = 10; k < 100; k++) {
  //     A[k-1] = 20 * k;
  //   }
  BufHandle a_buf("A", {100}, kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto forJ = For::make(j, 10, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  auto forK =
      For::make(k, 10, 100, Store::make(a_buf, {k - 1}, Mul::make(20, k)));
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  auto par = Block::make({forJ, forK});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));
}

TEST(LoopNest, fuseLoopsThatViolateDependencies2) {
  // Input IR:
  //   for (int j = 10; j < 100; j++) {
  //     A[j] = 10 * j;
  //   }
  //   for (int k = 10; k < 100; k++) {
  //     A[k+50] = 20 * k;
  //   }
  BufHandle a_buf("A", {150}, kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto forJ = For::make(j, 10, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  auto forK =
      For::make(k, 10, 100, Store::make(a_buf, {k + 50}, Mul::make(20, k)));
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  auto par = Block::make({forJ, forK});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));
}

TEST(LoopNest, fuseLoopsThatViolateDependencies3) {
  // Input IR:
  //   for (int m = 0; m < 20; m++) {
  //     A[m] = 0;
  //     for (int j = 0; j < 100; j++) {
  //       A[m] = A[m] + m * j;
  //     }
  //   }
  //   for (int n = 0; n < 20; n++) {
  //     B[n] = A[n+1];
  //     for (int k = 0; k < 50; k++) {
  //       B[n] = B[n] + n * k;
  //     }
  //   }
  BufHandle a_buf("A", {25, 100}, kInt);
  BufHandle b_buf("B", {20, 50}, kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto initA = Store::make(a_buf, {m}, 0);
  auto forJ = For::make(
      j,
      0,
      100,
      Store::make(
          a_buf, {m}, Add::make(Load::make(a_buf, {m}), Mul::make(m, j))));
  auto initB = Store::make(b_buf, {n}, Load::make(a_buf, {n + 1}));
  auto forK = For::make(
      k,
      0,
      50,
      Store::make(
          b_buf, {n}, Add::make(Load::make(b_buf, {n}), Mul::make(n, k))));
  auto forM = For::make(m, 0, 20, Block::make({initA, forJ}));
  auto forN = For::make(n, 0, 20, Block::make({initB, forK}));
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  auto par = Block::make({forM, forN});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forM, forN}, &fused_loop));
}

TEST(LoopNest, fuseLoopsThatViolateDependencies4) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 100; j++) {
  //       A[i,j] = i * j * 500;
  //     }
  //   }
  //   for (int m = 0; m < 20; m++) {
  //     for (int n = 0; n < 50; n++) {
  //       A[m+1,n] = m + n * 100;
  //     }
  //   }
  BufHandle a_buf("A", {30, 100}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  auto forI = For::make(
      i,
      0,
      20,
      For::make(
          j,
          0,
          100,
          Store::make(a_buf, {i, j}, Mul::make(Mul::make(i, j), 500))));
  auto forM = For::make(
      m,
      0,
      20,
      For::make(
          n,
          0,
          50,
          Store::make(a_buf, {m + 1, n}, Add::make(m, Mul::make(n, 100)))));
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  auto par = Block::make({forI, forM});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forI, forM}, &fused_loop));
}

TEST(LoopNest, fuseLoopsThatViolateDependencies5) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 100; j++) {
  //       A[i,j] = i * j * 500;
  //     }
  //     for (int n = 0; n < 100; n++) {
  //       A[i,n+1] = m + n * 100;
  //     }
  //   }
  BufHandle a_buf("A", {20, 200}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle n("n", kInt);
  auto forJ = For::make(
      j, 0, 100, Store::make(a_buf, {i, j}, Mul::make(Mul::make(i, j), 500)));
  auto forN = For::make(
      n,
      0,
      100,
      Store::make(a_buf, {i, n + 1}, Add::make(i, Mul::make(n, 100))));
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores,cppcoreguidelines-avoid-magic-numbers)
  auto forI = For::make(i, 0, 20, Block::make({forJ, forN}));
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forN}, &fused_loop));
}

TEST(LoopNest, fuseLoopsThatViolateDependencies6) {
  // Input IR:
  //   for (int j = 0; j < 100; j++) {
  //     A[j] = 10 * j;
  //   }
  //   for (int k = 0; k < 100; k++) {
  //     B[k] = 20 * A[99-k];
  //   }
  BufHandle a_buf("A", {100}, kInt);
  BufHandle b_buf("B", {100}, kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  auto forK = For::make(
      k,
      0,
      100,
      Store::make(
          b_buf, {k}, Mul::make(20, Load::make(a_buf, {ExprHandle(99) - k}))));
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  auto par = Block::make({forJ, forK});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));
}

TEST(LoopNest, fuseLoopsThatViolateDependencies7) {
  // Input IR:
  //   for (int k = 0; k < 100; k++) {
  //     B[k] = 20 * A[99-k];
  //   }
  //   for (int j = 0; j < 100; j++) {
  //     A[j] = 10 * j;
  //   }
  BufHandle a_buf("A", {100}, kInt);
  BufHandle b_buf("B", {100}, kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto forK = For::make(
      k,
      0,
      100,
      Store::make(
          b_buf, {k}, Mul::make(20, Load::make(a_buf, {ExprHandle(99) - k}))));
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  auto par = Block::make({forK, forJ});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forK, forJ}, &fused_loop));
}

TEST(LoopNest, areLoopsPerfectlyNested) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 30; j++) {
  //       for (int k = 0; k < 40; k++) {
  //         A[i,j,k] = i * j * k;
  //       }
  //     }
  //   }
  BufHandle a_buf("A", {20, 30, 40}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto store = Store::make(a_buf, {i, j, k}, Mul::make(Mul::make(i, j), k));
  auto forK = For::make(k, 0, 40, store);
  auto forJ = For::make(j, 0, 30, forK);
  auto forI = For::make(i, 0, 20, forJ);
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  auto par = Block::make({forI});
  ASSERT_TRUE(LoopNest::areLoopsPerfectlyNested({forI, forJ, forK}));

  // Specifying the loops in any other order fails.
  ASSERT_FALSE(LoopNest::areLoopsPerfectlyNested({forJ, forI, forK}));
  ASSERT_FALSE(LoopNest::areLoopsPerfectlyNested({forI, forK, forJ}));
  ASSERT_FALSE(LoopNest::areLoopsPerfectlyNested({forK, forJ, forI}));

  // Adding a statement to forK body should be OK.
  auto init = Store::make(a_buf, {i, j}, 0);
  forK->body()->insert_stmt_before(init, store);
  ASSERT_TRUE(LoopNest::areLoopsPerfectlyNested({forI, forJ, forK}));

  // Adding a statement in forJ body should fail this test.
  forK->body()->remove_stmt(init);
  forJ->body()->insert_stmt_before(init, forK);
  ASSERT_FALSE(LoopNest::areLoopsPerfectlyNested({forI, forJ, forK}));

  // Similarly, adding a statement in forI body should fail this test.
  forJ->body()->remove_stmt(init);
  forI->body()->insert_stmt_before(init, forJ);
  ASSERT_FALSE(LoopNest::areLoopsPerfectlyNested({forI, forJ, forK}));
}

TEST(LoopNest, reorderNestedLoops2D) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 30; j++) {
  //       A[i,j] = i * j;
  //     }
  //   }
  BufHandle a_buf("A", {20, 30, 40}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto store = Store::make(a_buf, {i, j}, Mul::make(i, j));
  auto forJ = For::make(j, 0, 30, store);
  auto forI = For::make(i, 0, 20, forJ);
  auto par = Block::make({forI});

  auto reordered = LoopNest::reorder({forI, forJ}, {1, 0});

  ASSERT_EQ(reordered[0], forJ);
  ASSERT_EQ(reordered[1], forI);
  ASSERT_TRUE(LoopNest::areLoopsPerfectlyNested({forJ, forI}));
  ASSERT_EQ(forJ->get_parent(), par);
  ASSERT_EQ(store->get_parent(), forI->body());
}

TEST(LoopNest, reorderNestedLoops3D) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 30; j++) {
  //       for (int k = 0; k < 40; k++) {
  //         A[i,j,k] = i * j * k;
  //       }
  //     }
  //   }
  BufHandle a_buf("A", {20, 30, 40}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto store = Store::make(a_buf, {i, j, k}, Mul::make(Mul::make(i, j), k));
  auto forK = For::make(k, 0, 40, store);
  auto forJ = For::make(j, 0, 30, forK);
  auto forI = For::make(i, 0, 20, forJ);
  auto par = Block::make({forI});

  auto reordered = LoopNest::reorder({forI, forJ, forK}, {2, 0, 1});

  ASSERT_EQ(reordered[0], forK);
  ASSERT_EQ(reordered[1], forI);
  ASSERT_EQ(reordered[2], forJ);
  ASSERT_TRUE(LoopNest::areLoopsPerfectlyNested({forK, forI, forJ}));
  ASSERT_EQ(forK->get_parent(), par);
  ASSERT_EQ(store->get_parent(), forJ->body());
}

TEST(LoopNest, reorderNestedLoops4D) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 30; j++) {
  //       for (int k = 0; k < 40; k++) {
  //         for (int l = 0; l < 50; l++) {
  //           A[i,j,k,l] = i * j * k * l * 500;
  //         }
  //       }
  //     }
  //   }
  BufHandle a_buf("A", {20, 30, 40, 50}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  VarHandle l("l", kInt);
  auto store = Store::make(
      a_buf,
      {i, j, k, l},
      Mul::make(Mul::make(Mul::make(Mul::make(i, j), k), l), 500));
  auto forL = For::make(l, 0, 50, store);
  auto forK = For::make(k, 0, 40, forL);
  auto forJ = For::make(j, 0, 30, forK);
  auto forI = For::make(i, 0, 20, forJ);
  auto par = Block::make({forI});

  auto reordered = LoopNest::reorder({forI, forJ, forK, forL}, {2, 0, 3, 1});

  ASSERT_EQ(reordered[0], forK);
  ASSERT_EQ(reordered[1], forI);
  ASSERT_EQ(reordered[2], forL);
  ASSERT_EQ(reordered[3], forJ);
  ASSERT_TRUE(LoopNest::areLoopsPerfectlyNested({forK, forI, forL, forJ}));
  ASSERT_EQ(forK->get_parent(), par);
  ASSERT_EQ(store->get_parent(), forJ->body());
}

TEST(LoopNest, reorderTrivialPermutation) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 30; j++) {
  //       for (int k = 0; k < 40; k++) {
  //         A[i,j,k] = i * j * k;
  //       }
  //     }
  //   }
  BufHandle a_buf("A", {20, 30, 40}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto store = Store::make(a_buf, {i, j, k}, Mul::make(Mul::make(i, j), k));
  auto forK = For::make(k, 0, 40, store);
  auto forJ = For::make(j, 0, 30, forK);
  auto forI = For::make(i, 0, 20, forJ);
  auto par = Block::make({forI});

  auto reordered = LoopNest::reorder({forI, forJ, forK}, {0, 1, 2});

  ASSERT_EQ(reordered[0], forI);
  ASSERT_EQ(reordered[1], forJ);
  ASSERT_EQ(reordered[2], forK);
  ASSERT_TRUE(LoopNest::areLoopsPerfectlyNested({forI, forJ, forK}));
  ASSERT_EQ(forI->get_parent(), par);
  ASSERT_EQ(store->get_parent(), forK->body());
}

TEST(LoopNest, reorderInvalidPermutations) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 30; j++) {
  //       for (int k = 0; k < 40; k++) {
  //         A[i,j,k] = i * j * k;
  //       }
  //     }
  //   }
  BufHandle a_buf("A", {20, 30, 40}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto store = Store::make(a_buf, {i, j, k}, Mul::make(Mul::make(i, j), k));
  auto forK = For::make(k, 0, 40, store);
  auto forJ = For::make(j, 0, 30, forK);
  auto forI = For::make(i, 0, 20, forJ);
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  auto par = Block::make({forI});

  ASSERT_THROWS_WITH(
      LoopNest::reorder({forI, forJ, forK}, {0, 1, 2, 3}),
      "invalid permutation size");
  ASSERT_THROWS_WITH(
      LoopNest::reorder({forI, forJ, forK}, {1, 2}),
      "invalid permutation size");
  ASSERT_THROWS_WITH(
      LoopNest::reorder({forI, forJ, forK}, {2, 1, 3}),
      "invalid permutation for reorder");
  ASSERT_THROWS_WITH(
      LoopNest::reorder({forI, forJ, forK}, {1, 1, 0}),
      "invalid permutation for reorder");
  ASSERT_THROWS_WITH(
      LoopNest::reorder({forI, forJ, forK}, {0, 0, 0}),
      "invalid permutation for reorder");
}

TEST(LoopNest, reorderInvalidLoopNest) {
  // Input IR:
  //   for (int i = 0; i < 20; i++) {
  //     for (int j = 0; j < 30; j++) {
  //       A[i,j] = 0
  //       for (int k = 0; k < 40; k++) {
  //         A[i,j,k] = i * j * k;
  //       }
  //     }
  //   }
  BufHandle a_buf("A", {20, 30, 40}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto store = Store::make(a_buf, {i, j, k}, Mul::make(Mul::make(i, j), k));
  auto forK = For::make(k, 0, 40, store);
  auto forJ = For::make(j, 0, 30, forK);
  auto forI = For::make(i, 0, 20, forJ);
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  auto par = Block::make({forI});

  // Specifying the loops in incorrect order fails.
  ASSERT_THROWS_WITH(
      LoopNest::reorder({forK, forI, forJ}, {1, 0, 2}),
      "reorder is only allowed on perfectly nested loops");

  // Adding a statement to forJ loop fails.
  auto init = Store::make(a_buf, {i}, 0);
  forJ->body()->insert_stmt_before(init, forK);
  ASSERT_THROWS_WITH(
      LoopNest::reorder({forI, forJ, forK}, {1, 0, 2}),
      "reorder is only allowed on perfectly nested loops");

  // Moving that statement to forI loop also fails.
  forJ->body()->remove_stmt(init);
  forI->body()->insert_stmt_before(init, forJ);
  ASSERT_THROWS_WITH(
      LoopNest::reorder({forI, forJ, forK}, {1, 0, 2}),
      "reorder is only allowed on perfectly nested loops");
}

TEST(LoopNest, compressBufferSimple) {
  // Input IR:
  // for (int i = 0; i < 100; ++i) {
  //   for (int j = 0; j < 200; ++j) {
  //     A[i,j] = sin(i*j)
  //   }
  //   for (int j = 0; j < 199; ++j) {
  //     B[i,j] = A[i,j] + A[i, j+1]
  //   }
  // }
  BufHandle aBuf("A", {100, 200}, kInt);
  BufHandle bBuf("B", {100, 200}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto forJ1 = For::make(j, 0, 200, Store::make(aBuf, {i, j}, sin(i * j)));
  auto forJ2 = For::make(
      j,
      0,
      199,
      Store::make(
          bBuf,
          {i, j},
          Add::make(Load::make(aBuf, {i, j}), Load::make(aBuf, {i, j + 1}))));
  auto forI = For::make(i, 0, 100, Block::make({forJ1, forJ2}));
  auto par = Block::make({forI});
  LoopNest::compressBuffer(aBuf.node(), par);

  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[0, j] =
# CHECK: for (int j
# CHECK-NEXT: B[i, j] = (A[0, j]) + (A[0, j + 1])
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  ASSERT_EQ(aBuf.node()->ndim(), 2);
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(0), 1);
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(1), 200);
}

TEST(LoopNest, compressBufferMultipleDims) {
  // Input IR:
  // for (int i = 0; i < 100; ++i) {
  //   for (int j = 0; j < 200; ++j) {
  //     A[i,j] = sin(i*j)
  //     B[i,j] = A[i,j] + A[i,j]
  //   }
  // }
  BufHandle aBuf("A", {100, 200}, kInt);
  BufHandle bBuf("B", {100, 200}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto store1 = Store::make(aBuf, {i, j}, sin(i * j));
  auto store2 = Store::make(
      bBuf,
      {i, j},
      Add::make(Load::make(aBuf, {i, j}), Load::make(aBuf, {i, j})));
  auto forJ = For::make(j, 0, 200, Block::make({store1, store2}));
  auto forI = For::make(i, 0, 100, forJ);
  auto par = Block::make({forI});
  LoopNest::compressBuffer(aBuf.node(), par);

  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[0, 0] =
# CHECK-NEXT: B[i, j] = (A[0, 0]) + (A[0, 0])
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  ASSERT_EQ(aBuf.node()->ndim(), 2);
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(0), 1);
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(1), 1);
}

TEST(LoopNest, compressBufferMultipleDims2) {
  // Input IR:
  // for (int i = 0; i < 100; ++i) {
  //   for (int j = 0; j < 200; ++j) {
  //     for (int k = 0; k < 300; ++k) {
  //       A[i,j,k] = sin(i*j*k)
  //     }
  //     for (int k = 0; k < 299; ++j) {
  //       B[i,j,k] = A[i,j,k] + A[i,j,k+1]
  //     }
  //   }
  // }
  BufHandle aBuf("A", {100, 200, 300}, kInt);
  BufHandle bBuf("B", {100, 200, 300}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto store1 = Store::make(aBuf, {i, j, k}, sin(i * j * k));
  auto forK1 = For::make(k, 0, 300, store1);
  auto store2 = Store::make(
      bBuf,
      {i, j, k},
      Add::make(Load::make(aBuf, {i, j, k}), Load::make(aBuf, {i, j, k + 1})));
  auto forK2 = For::make(k, 0, 299, store2);
  auto forJ = For::make(j, 0, 200, Block::make({forK1, forK2}));
  auto forI = For::make(i, 0, 100, forJ);
  auto par = Block::make({forI});
  LoopNest::compressBuffer(aBuf.node(), par);

  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: for (int k
# CHECK-NEXT: A[0, 0, k] =
# CHECK: for (int k
# CHECK-NEXT: B[i, j, k] = (A[0, 0, k]) + (A[0, 0, k + 1])
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  ASSERT_EQ(aBuf.node()->ndim(), 3);
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(0), 1);
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(1), 1);
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(2), 300);
}

TEST(LoopNest, compressBufferDifferentOrderIndices) {
  // Input IR:
  // for (int i = 0; i < 100; ++i) {
  //   for (int j = 0; j < 200; ++j) {
  //     A[j, i] = sin(i*j)
  //   }
  //   for (int j = 0; j < 99; ++j) {
  //     B[i, j] = A[j, i] + A[j+1, 0]
  //   }
  // }
  BufHandle aBuf("A", {100, 200}, kInt);
  BufHandle bBuf("B", {100, 200}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto forJ1 = For::make(j, 0, 200, Store::make(aBuf, {j, i}, sin(i * j)));
  auto forJ2 = For::make(
      j,
      0,
      99,
      Store::make(
          bBuf,
          {i, j},
          Add::make(Load::make(aBuf, {j, i}), Load::make(aBuf, {j + 1, i}))));
  auto forI = For::make(i, 0, 100, Block::make({forJ1, forJ2}));
  auto par = Block::make({forI});
  LoopNest::compressBuffer(aBuf.node(), par);

  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[j, 0] =
# CHECK: for (int j
# CHECK-NEXT: B[i, j] = (A[j, 0]) + (A[j + 1, 0])
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  ASSERT_EQ(aBuf.node()->ndim(), 2);
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(0), 100);
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(1), 1);
}

TEST(LoopNest, compressBufferVariableBounds) {
  // Input IR:
  // for (int i = 0; i < M; ++i) {
  //   for (int j = 0; j < N; ++j) {
  //     A[i,j] = sin(i*j)
  //   }
  //   for (int j = 0; j < N-1; ++j) {
  //     B[i,j] = A[i,j] + A[i, j+1]
  //   }
  // }
  BufHandle aBuf("A", {100, 200}, kInt);
  BufHandle bBuf("B", {100, 200}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle M("M", kInt);
  VarHandle N("N", kInt);
  auto forJ1 = For::make(j, 0, N, Store::make(aBuf, {i, j}, sin(i * j)));
  auto forJ2 = For::make(
      j,
      0,
      N - 1,
      Store::make(
          bBuf,
          {i, j},
          Add::make(Load::make(aBuf, {i, j}), Load::make(aBuf, {i, j + 1}))));
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  auto forI = For::make(i, 0, M, Block::make({forJ1, forJ2}));
  auto par = Block::make({forI});
  LoopNest::compressBuffer(aBuf.node(), par);

  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[0, j] =
# CHECK: for (int j
# CHECK-NEXT: B[i, j] = (A[0, j]) + (A[0, j + 1])
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  ASSERT_EQ(aBuf.node()->ndim(), 2);
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(0), 1);
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(1), 200);
}

TEST(LoopNest, compressBufferNoCommonParentLoops) {
  // Input IR:
  // for (int i = 0; i < 100; ++i) {
  //   for (int j = 0; j < 200; ++j) {
  //     A[i,j] = sin(i*j)
  //   }
  // }
  // for (int i = 0; i < 100; ++i) {
  //   for (int j = 0; j < 199; ++j) {
  //     B[i,j] = A[i,j] + A[i, j+1]
  //   }
  // }
  BufHandle aBuf("A", {100, 200}, kInt);
  BufHandle bBuf("B", {100, 200}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto forJ1 = For::make(j, 0, 200, Store::make(aBuf, {i, j}, sin(i * j)));
  auto forJ2 = For::make(
      j,
      0,
      199,
      Store::make(
          bBuf,
          {i, j},
          Add::make(Load::make(aBuf, {i, j}), Load::make(aBuf, {i, j + 1}))));
  auto forI1 = For::make(i, 0, 100, forJ1);
  auto forI2 = For::make(i, 0, 100, forJ2);
  auto par = Block::make({forI1, forI2});
  LoopNest::compressBuffer(aBuf.node(), par);

  // There should be no change in the buffer or code.
  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i, j] =
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: B[i, j] = (A[i, j]) + (A[i, j + 1])
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  ASSERT_EQ(aBuf.node()->ndim(), 2);
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(0), 100);
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(1), 200);
}

TEST(LoopNest, compressBufferIndicesMixed) {
  // Input IR:
  // for (int i = 0; i < 100; ++i) {
  //   for (int j = 0; j < 200; ++j) {
  //     A[i + j, j] = sin(i*j)
  //   }
  //   for (int j = 0; j < 199; ++j) {
  //     B[i,j] = A[i + j, j] + A[i + j, j+1]
  //   }
  // }
  BufHandle aBuf("A", {300, 200}, kInt);
  BufHandle bBuf("B", {100, 200}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto forJ1 = For::make(j, 0, 200, Store::make(aBuf, {i + j, j}, sin(i * j)));
  auto forJ2 = For::make(
      j,
      0,
      199,
      Store::make(
          bBuf,
          {i, j},
          Add::make(
              Load::make(aBuf, {i + j, j}), Load::make(aBuf, {i + j, j + 1}))));
  auto forI = For::make(i, 0, 100, Block::make({forJ1, forJ2}));
  auto par = Block::make({forI});
  LoopNest::compressBuffer(aBuf.node(), par);

  // There should be no change in the buffer or code.
  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[i + j, j] =
# CHECK: for (int j
# CHECK-NEXT: B[i, j] = (A[i + j, j]) + (A[i + j, j + 1])
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  ASSERT_EQ(aBuf.node()->ndim(), 2);
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(0), 300);
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(1), 200);
}

TEST(LoopNest, compressMultipleBuffers) {
  // Input IR:
  // for (int i = 0; i < 100; ++i) {
  //   for (int j = 0; j < 200; ++j) {
  //     A[i,j] = sin(i*j)
  //   }
  //   for (int k = 0; k < 199; ++k) {
  //     B[i,k] = A[i,k] + A[i, k+1]
  //   }
  //   for (int m = 0; m < 50; ++m) {
  //     C[i,m] = B[i,m]
  //   }
  // }
  BufHandle aBuf("A", {100, 200}, kInt);
  BufHandle bBuf("B", {100, 200}, kInt);
  BufHandle cBuf("C", {100, 200}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  VarHandle m("m", kInt);
  auto forJ = For::make(j, 0, 200, Store::make(aBuf, {i, j}, sin(i * j)));
  auto forK = For::make(
      k,
      0,
      199,
      Store::make(
          bBuf,
          {i, k},
          Add::make(Load::make(aBuf, {i, k}), Load::make(aBuf, {i, k + 1}))));
  auto forM =
      For::make(m, 0, 50, Store::make(cBuf, {i, m}, Load::make(bBuf, {i, m})));
  auto forI = For::make(i, 0, 100, Block::make({forJ, forK, forM}));
  auto par = Block::make({forI});

  // This should compress all buffers A, B, and C as follows:
  //   A[100, 200] -> A[1, 200]
  //   B[100, 200] -> B[1, 200]
  //   C[100, 200] -> C[1, 1]
  LoopNest::compressAllBuffers(par);

  std::ostringstream oss;
  oss << *par;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i
# CHECK-NEXT: for (int j
# CHECK-NEXT: A[0, j] =
# CHECK: for (int k
# CHECK-NEXT: B[0, k] = (A[0, k]) + (A[0, k + 1])
# CHECK: for (int m
# CHECK-NEXT: C[0, 0] = B[0, m]
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  ASSERT_EQ(aBuf.node()->ndim(), 2);
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(0), 1);
  IS_IMM_WITH_VAL(Int, aBuf.node()->dim(1), 200);
  ASSERT_EQ(bBuf.node()->ndim(), 2);
  IS_IMM_WITH_VAL(Int, bBuf.node()->dim(0), 1);
  IS_IMM_WITH_VAL(Int, bBuf.node()->dim(1), 200);
  ASSERT_EQ(cBuf.node()->ndim(), 2);
  IS_IMM_WITH_VAL(Int, cBuf.node()->dim(0), 1);
  IS_IMM_WITH_VAL(Int, cBuf.node()->dim(1), 1);
}

TEST(LoopNest, sanitizeNames) {
  std::vector<ExprHandle> dim_args;
  // Let's pick names that would overlap with default index names if not
  // sanitized properly:
  dim_args.emplace_back(ExprHandle(alloc<Var>("i", kInt)));
  dim_args.emplace_back(ExprHandle(alloc<Var>("N:2", kInt)));
  // Now let's create a many dimensions so that we had to use the same letter
  // for different loops
  for (int i = 0; i < 10; i++) {
    dim_args.emplace_back(ExprHandle(alloc<Var>("N", kInt)));
  }

  // Now create two Computes with conflicting after sanitization names:
  Tensor X = Compute("$X:!", dim_args, [&](const std::vector<VarHandle>& v) {
    return v[0] + v[1] + v[9] + 1;
  });
  Tensor Y = Reduce(
      "%X\"+",
      {},
      Sum(),
      [&](const std::vector<VarHandle>& v) { return X.load(v); },
      dim_args);

  // Finally, let's verify what we got after sanitization:
  LoopNest l({X, Y});
  StmtPtr s = l.root_stmt();
  LoopNest::sanitizeNames(s);

  std::ostringstream oss;
  oss << *s;
  const std::string& verification_pattern =
      R"IR(
# CHECK:  for (int i = 0; i < i_1; i++) {
# CHECK-NEXT:    for (int j = 0; j < N_2_1; j++) {
# CHECK-NEXT:      for (int k = 0; k < N_9; k++) {
# CHECK-NEXT:        for (int l = 0; l < N_8; l++) {
# CHECK-NEXT:          for (int m = 0; m < N_7; m++) {
# CHECK-NEXT:            for (int n = 0; n < N_6; n++) {
# CHECK-NEXT:              for (int o = 0; o < N_5; o++) {
# CHECK-NEXT:                for (int p = 0; p < N_4; p++) {
# CHECK-NEXT:                  for (int i1 = 0; i1 < N_3; i1++) {
# CHECK-NEXT:                    for (int j1 = 0; j1 < N_2; j1++) {
# CHECK-NEXT:                      for (int k1 = 0; k1 < N_1; k1++) {
# CHECK-NEXT:                        for (int l1 = 0; l1 < N; l1++) {
# CHECK-NEXT:                          v_X__[i, j, k, l, m, n, o, p, i1, j1, k1, l1] = ((i + j) + j1) + 1;
# CHECK:  v_X___1 = int(0);
# CHECK-NEXT:  for (int i_2 = 0; i_2 < i_1; i_2++) {
# CHECK-NEXT:    for (int j_1 = 0; j_1 < N_2_1; j_1++) {
# CHECK-NEXT:      for (int k_1 = 0; k_1 < N_9; k_1++) {
# CHECK-NEXT:        for (int l_1 = 0; l_1 < N_8; l_1++) {
# CHECK-NEXT:          for (int m_1 = 0; m_1 < N_7; m_1++) {
# CHECK-NEXT:            for (int n_1 = 0; n_1 < N_6; n_1++) {
# CHECK-NEXT:              for (int o_1 = 0; o_1 < N_5; o_1++) {
# CHECK-NEXT:                for (int p_1 = 0; p_1 < N_4; p_1++) {
# CHECK-NEXT:                  for (int i1_1 = 0; i1_1 < N_3; i1_1++) {
# CHECK-NEXT:                    for (int j1_1 = 0; j1_1 < N_2; j1_1++) {
# CHECK-NEXT:                      for (int k1_1 = 0; k1_1 < N_1; k1_1++) {
# CHECK-NEXT:                        for (int l1_1 = 0; l1_1 < N; l1_1++) {
# CHECK-NEXT:                          v_X___1 = ReduceOp((v_X___1) + (v_X__[i_2, j_1, k_1, l_1, m_1, n_1, o_1, p_1, i1_1, j1_1, k1_1, l1_1]), reduce_args={i_2, j_1, k_1, l_1, m_1, n_1, o_1, p_1, i1_1, j1_1, k1_1, l1_1});
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

} // namespace jit
} // namespace torch
