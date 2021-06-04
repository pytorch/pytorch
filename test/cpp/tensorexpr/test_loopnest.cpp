#include <gtest/gtest.h>

#include <test/cpp/tensorexpr/test_base.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <test/cpp/tensorexpr/padded_buffer.h>
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

void checkIR(Stmt* s, const std::string& pattern) {
  std::ostringstream oss;
  oss << *s;
  torch::jit::testing::FileCheck().run(pattern, oss.str());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ExprSimple01) {
  KernelScope kernel_scope;
  Tensor* tensor = Compute(
      "f", {{16, "X"}, {5, "y"}}, [](const VarHandle& x, const VarHandle& y) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  LoopNest l({tensor});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);

  l.splitWithTail(loops[0], 2);
  l.splitWithTail(loops[0], 2);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ExprLower01) {
  KernelScope kernel_scope;
  Tensor* tensor = Compute(
      "f", {{16, "x"}, {5, "y"}}, [](const VarHandle& x, const VarHandle& y) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  LoopNest l({tensor});
  Stmt* stmt = l.root_stmt();
  std::ostringstream oss;
  oss << *stmt;
  ASSERT_GT(oss.str().size(), 20);
  ASSERT_LT(oss.str().size(), 200);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ExprSimple02) {
  KernelScope kernel_scope;
  auto func = [](const ExprHandle& x, const ExprHandle& y) {
    return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
  };
  Tensor* tensor = Compute("f", {{26, "x"}, {5, "y"}}, func);
  LoopNest l({tensor});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);

  l.splitWithTail(loops[0], 4);

  Stmt* stmt = l.root_stmt();
  std::ostringstream oss;
  oss << *stmt;
  ASSERT_GT(oss.str().size(), 200);
  ASSERT_LT(oss.str().size(), 600);

  {
    // Compare to a reference loop structure structure.
    VarHandle x_outer("x_outer", kInt);
    VarHandle x_inner("x_inner", kInt);
    VarHandle y("y", kInt);
    VarHandle x_tail("x_tail", kInt);
    BufHandle f("f", {26, 5}, kFloat);
    ExprHandle x_1 = x_outer * 4 + x_inner;
    ExprHandle x_outer_end = (ExprHandle(26) - 0) / 4;
    For* stmt1 = For::make(
        x_outer,
        0,
        x_outer_end,
        For::make(
            x_inner,
            0,
            4,
            For::make(y, 0, 5, Store::make(f, {x_1, y}, func(x_1, y)))));
    ExprHandle x_2 = x_tail + x_outer_end * 4;
    For* stmt2 = For::make(
        x_tail,
        0,
        (ExprHandle(26) - 0) % 4,
        For::make(y, 0, 5, Store::make(f, {x_2, y}, func(x_2, y))));
    Stmt* stmt = Block::make({stmt1, stmt2});

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

Block* getSimplifiedBody(const LoopNest& l) {
  Stmt* stmt = l.root_stmt();
  Stmt* simplified = IRSimplifier::simplify(stmt);
  return dynamic_cast<Block*>(simplified);
}

void assertForRange(For* f, int expected_start, int expected_stop) {
  ASSERT_NE(f, nullptr);
  const IntImm* start = dynamic_cast<const IntImm*>(f->start());
  ASSERT_NE(start, nullptr);
  ASSERT_EQ(start->value(), expected_start);
  const IntImm* stop = dynamic_cast<const IntImm*>(f->stop());
  ASSERT_NE(stop, nullptr);
  ASSERT_EQ(stop->value(), expected_stop);
}

void assertForRanges(
    Block* body,
    const std::vector<std::pair<int, int>>& start_stops) {
  ASSERT_EQ(body->nstmts(), start_stops.size());

  auto it = body->begin();
  for (size_t i = 0; i < start_stops.size(); i++, it++) {
    For* loop = dynamic_cast<For*>(*it);
    assertForRange(loop, start_stops[i].first, start_stops[i].second);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ExprSliceHeadWithLoopOptions) {
  KernelScope kernel_scope;
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor* tensor = Compute("f", {{10, "x"}}, func);
  LoopNest l({tensor});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* tail;
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  l.setGPUBlockIndex(loops[0], LoopOptions::IDX_Y);
  l.sliceHead(loops[0], 2, &head, &tail);

  Block* body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 2}, {0, 8}});

  ASSERT_TRUE(tail->loop_options().is_gpu_block_index());
  ASSERT_EQ(tail->loop_options().gpu_block_index(), LoopOptions::IDX_Y);

  ASSERT_TRUE(head->loop_options().isDefault());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ExprSliceTailWithLoopOptions) {
  KernelScope kernel_scope;
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor* tensor = Compute("f", {{10, "x"}}, func);
  LoopNest l({tensor});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* tail;
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  l.sliceTail(loops[0], 4, &head, &tail);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* tail_head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* tail_tail;
  l.setGPUBlockIndex(tail, LoopOptions::IDX_Y);
  l.sliceTail(tail, 2, &tail_head, &tail_tail);

  Block* body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 6}, {0, 2}, {8, 10}});

  ASSERT_TRUE(tail_head->loop_options().is_gpu_block_index());
  ASSERT_EQ(tail_head->loop_options().gpu_block_index(), LoopOptions::IDX_Y);

  ASSERT_TRUE(head->loop_options().isDefault());
  ASSERT_TRUE(tail_tail->loop_options().isDefault());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ExprSliceHeadWhenFactorEqualsSize) {
  // When factor equals the For loop's original size, keep using the original
  // For loop.
  KernelScope kernel_scope;
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor* tensor = Compute("f", {{10, "x"}}, func);
  LoopNest l({tensor});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* tail;
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  l.sliceHead(loops[0], 10, &head, &tail);

  ASSERT_EQ(head, loops[0]);
  ASSERT_EQ(tail, nullptr);

  Block* body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 10}});
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ExprSliceHeadWhenFactorLargerThanSize) {
  KernelScope kernel_scope;
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor* tensor = Compute("f", {{10, "x"}}, func);
  LoopNest l({tensor});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* tail;
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  l.sliceHead(loops[0], 100, &head, &tail);

  ASSERT_EQ(head, loops[0]);
  ASSERT_EQ(tail, nullptr);

  Block* body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 10}});
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ExprSliceHead) {
  KernelScope kernel_scope;
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor* tensor = Compute("f", {{10, "x"}}, func);
  LoopNest l({tensor});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* tail;
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  l.sliceHead(loops[0], 4, &head, &tail);

  ASSERT_NE(head, nullptr);
  ASSERT_NE(head, loops[0]);
  ASSERT_NE(tail, nullptr);
  ASSERT_NE(tail, loops[0]);

  Block* body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 4}, {4, 10}});
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ExprSliceHeadWithNonZeroStart) {
  KernelScope kernel_scope;
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor* tensor = Compute("f", {{10, "x"}}, func);
  LoopNest l({tensor});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* tail;
  l.sliceTail(loops[0], 4, &head, &tail);
  // head: [0, 6)
  // tail: [6, 10)

  l.sliceHead(tail, 2);
  // tail_head: [6, 8)
  // tail_tail: [8, 10)

  Block* body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 6}, {6, 8}, {8, 10}});
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ExprSliceTailWhenFactorEqualsSize) {
  // When factor equals the For loop's original size, keep using the original
  // For loop.
  KernelScope kernel_scope;
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor* tensor = Compute("f", {{10, "x"}}, func);
  LoopNest l({tensor});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* tail;
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  l.sliceTail(loops[0], 10, &head, &tail);

  ASSERT_EQ(head, nullptr);
  ASSERT_EQ(tail, loops[0]);

  Block* body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 10}});
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ExprSliceTailWhenFactorLargerThanSize) {
  // When factor equals the For loop's original size, keep using the original
  // For loop.
  KernelScope kernel_scope;
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor* tensor = Compute("f", {{10, "x"}}, func);
  LoopNest l({tensor});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* tail;
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  l.sliceTail(loops[0], 100, &head, &tail);

  ASSERT_EQ(head, nullptr);
  ASSERT_EQ(tail, loops[0]);

  Block* body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 10}});
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ExprSliceTail) {
  KernelScope kernel_scope;
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor* tensor = Compute("f", {{10, "x"}}, func);
  LoopNest l({tensor});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* tail;
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  l.sliceTail(loops[0], 4, &head, &tail);

  ASSERT_NE(head, nullptr);
  ASSERT_NE(head, loops[0]);
  ASSERT_NE(tail, nullptr);
  ASSERT_NE(tail, loops[0]);

  Block* body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 6}, {6, 10}});
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ExprSplitAndSlice) {
  // 0: splitWithTail
  // 1: sliceTail on inner loop
  // 2: sliceHead on outer loop
  KernelScope kernel_scope;
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor* tensor = Compute("f", {{100, "x"}}, func);
  LoopNest l({tensor});

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* inner;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* tail;
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  // outer: [0, 4)
  // inner: [0, 21)
  // tail:  [84, 100)
  l.splitWithTail(loops[0], 21, &inner, &tail);
  l.sliceTail(inner, 2);
  l.sliceHead(loops[0], 2);

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
  Block* body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 2}, {2, 4}, {0, 16}});

  auto biter = body->begin();

  For* loop = dynamic_cast<For*>(*biter++);
  assertForRanges(loop->body(), {{0, 19}, {19, 21}});

  loop = dynamic_cast<For*>(*biter);
  assertForRanges(loop->body(), {{0, 19}, {19, 21}});
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ExprSliceAndNormalize) {
  // 0: sliceHead
  // 1: normalize tail
  KernelScope kernel_scope;
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor* tensor = Compute("f", {{10, "x"}}, func);
  LoopNest l({tensor});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* head;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* tail;
  l.sliceHead(loops[0], 2, &head, &tail);
  // head: [0, 2)
  // tail: [2, 10)

  LoopNest::normalize(tail);
  // normalized_tail: [0, 8)

  Block* body = getSimplifiedBody(l);
  assertForRanges(body, {{0, 2}, {0, 8}});
}

template <typename T>
T evalExpr(const ExprHandle& expr, const VarHandle& var, T value) {
  ExprEval<SimpleIREvaluator> eval(expr, {var});
  return eval.value<T>(value);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ExprSliceWithVariableDimension) {
  auto testWithDimension =
      [](int dimension,
         const std::vector<std::pair<int, int>>& expected_for_ranges) {
        KernelScope kernel_scope;
        VarHandle dim("dim", kInt);
        Tensor* tensor =
            Compute("f", {{dim, "x"}}, [](const ExprHandle& x) { return x; });
        LoopNest l({tensor});
        std::vector<For*> loops =
            l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);

        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        For* head;
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        For* tail;
        l.sliceHead(loops[0], 2, &head, &tail);

        l.sliceTail(tail, 2);

        Block* body = getSimplifiedBody(l);
        ASSERT_EQ(expected_for_ranges.size(), 3);
        auto it = body->begin();
        for (auto& start_stop : expected_for_ranges) {
          For* loop = dynamic_cast<For*>(*it++);
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ExprSplitWithTail) {
  KernelScope kernel_scope;
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor* tensor = Compute("f", {{199, "x"}}, func);
  LoopNest l({tensor});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  l.splitWithTail(loops[0], 17);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  l.splitWithTail(loops[0], 7);

  Stmt* stmt = l.root_stmt();
  Stmt* simplified = IRSimplifier::simplify(stmt);
  Block* body = dynamic_cast<Block*>(simplified);
  ASSERT_EQ(body->nstmts(), 3);
  auto biter = body->begin();

  // Verify that the split loops are ordered correctly.
  For* loop = dynamic_cast<For*>(*biter++);
  assertForRange(loop, 0, 7);

  loop = dynamic_cast<For*>(*biter++);
  assertForRange(loop, 0, 4);

  loop = dynamic_cast<For*>(*biter);
  assertForRange(loop, 0, 12);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ExprSplitWithTailNone) {
  KernelScope kernel_scope;
  auto func = [](const ExprHandle& x, const ExprHandle& y) {
    return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
  };
  Tensor* tensor = Compute("f", {{24, "x"}, {5, "y"}}, func);
  LoopNest l({tensor});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  l.splitWithTail(loops[0], 4);

  Stmt* stmt = l.root_stmt();
  std::ostringstream oss;
  oss << *stmt;
  ASSERT_GT(oss.str().size(), 200);
  ASSERT_LT(oss.str().size(), 600);

  {
    // Compare to a reference loop structure structure.
    VarHandle x_outer("x_outer", kInt);
    VarHandle x_inner("x_inner", kInt);
    VarHandle y("y", kInt);
    VarHandle x_tail("x_tail", kInt);
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks,cppcoreguidelines-avoid-magic-numbers)
    BufHandle f("f", {24, 5}, kFloat);
    ExprHandle x_1 = x_outer * 4 + x_inner;
    ExprHandle x_outer_end = (ExprHandle(24) - 0) / 4;
    Stmt* stmt = new Block({For::make(
        x_outer,
        0,
        x_outer_end,
        For::make(
            x_inner,
            0,
            4,
            For::make(y, 0, 5, Store::make(f, {x_1, y}, func(x_1, y)))))});

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ExprSplitWithMask01) {
  KernelScope kernel_scope;
  const int M = 26;
  const int N = 5;
  Placeholder a_buf("a", kFloat, {M, N});
  Placeholder b_buf("b", kFloat, {M, N});
  Tensor* tensor = Compute(
      "f", {{M, "m"}, {N, "n"}}, [&](const ExprHandle& m, const ExprHandle& n) {
        return a_buf.load(m, n) + b_buf.load(m, n) + 1.0f;
      });

  LoopNest l({tensor});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  l.splitWithMask(loops[1], 4);

  Stmt* stmt = l.root_stmt();

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
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ExprSplitWithMaskRepeatedNoMask) {
  KernelScope kernel_scope;
  const int M = 64;
  Placeholder a_buf("a", kFloat, {M});
  Placeholder b_buf("b", kFloat, {M});
  Tensor* tensor = Compute("f", {{M, "m"}}, [&](const ExprHandle& m) {
    return a_buf.load(m) + b_buf.load(m) + 1.0f;
  });

  LoopNest l({tensor});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  l.splitWithMask(loops[0], 4);
  l.splitWithMask(loops[0], 4);

  Stmt* stmt1 = IRSimplifier::simplify(l.root_stmt());

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, SplitWithTailWithLoopOptions) {
  KernelScope kernel_scope;
  const int M = 21;
  Placeholder a_buf("a", kFloat, {M});
  Placeholder b_buf("b", kFloat, {M});
  Tensor* tensor = Compute("f", {{M, "m"}}, [&](const ExprHandle& m) {
    return a_buf.load(m) + b_buf.load(m) + 1.0f;
  });
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For *inner, *tail;

  LoopNest l({tensor});
  auto loops = NodeFinder<For>::find(l.root_stmt());
  ASSERT_GT(loops.size(), 0);
  l.setGPUBlockIndex(loops[0], LoopOptions::IDX_Y);
  l.splitWithTail(loops[0], 4, &inner, &tail);
  ASSERT_NE(inner, nullptr);
  ASSERT_NE(tail, nullptr);
  For* outer = loops[0];

  // Outer loop carries loop axis bindings.
  ASSERT_TRUE(outer->loop_options().is_gpu_block_index());
  ASSERT_EQ(outer->loop_options().gpu_block_index(), LoopOptions::IDX_Y);

  // Inner loop has none.
  ASSERT_TRUE(inner->loop_options().isDefault());

  // Tail loop has none.
  ASSERT_TRUE(tail->loop_options().isDefault());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, SplitWithMaskWithLoopOptions) {
  KernelScope kernel_scope;
  const int M = 21;
  Placeholder a_buf("a", kFloat, {M});
  Placeholder b_buf("b", kFloat, {M});
  Tensor* tensor = Compute("f", {{M, "m"}}, [&](const ExprHandle& m) {
    return a_buf.load(m) + b_buf.load(m) + 1.0f;
  });
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* inner;

  LoopNest l({tensor});
  auto loops = NodeFinder<For>::find(l.root_stmt());
  l.setGPUBlockIndex(loops[0], LoopOptions::IDX_Y);
  l.splitWithMask(loops[0], 4, &inner);
  For* outer = loops[0];

  // Outer loop carries loop axis bindings.
  ASSERT_TRUE(outer->loop_options().is_gpu_block_index());
  ASSERT_EQ(outer->loop_options().gpu_block_index(), LoopOptions::IDX_Y);

  // Inner loop has none.
  ASSERT_TRUE(inner->loop_options().isDefault());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleBroadcastAddBuffer) {
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;
  Placeholder a_buf("a", kFloat, {M, N});
  Placeholder b_buf("b", kFloat, {N, K});
  Tensor* c = Compute(
      "broadcast_add",
      {{M, "m"}, {N, "n"}, {K, "k"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) + b_buf.load(n, k);
      });
  LoopNest l({c});
  Stmt* stmt = l.root_stmt();

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleFunctionCall01) {
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;
  Placeholder a_buf("a", kFloat, {M, N});
  Placeholder b_buf("b", kFloat, {N, K});
  Tensor* c = Compute(
      "broadcast_add",
      {{M, "m"}, {N, "n"}, {K, "k"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) + b_buf.load(n, k);
      });
  Tensor* d = Compute(
      "d",
      {{M, "m"}, {N, "n"}, {K, "k"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return c->load(m, n, k) + 1;
      });

  LoopNest l({d}, {c, d});
  l.prepareForCodegen();
  Stmt* stmt = l.root_stmt();
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleInlineSimple) {
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;
  Placeholder a_buf("a", kFloat, {M, N});
  Placeholder b_buf("b", kFloat, {N, K});
  Placeholder c_buf("c", kFloat, {M, N});
  Placeholder d_buf("d", kFloat, {M, K});

  Tensor* x = Compute(
      "x",
      {{M, "m1"}, {N, "n1"}, {K, "k1"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) * b_buf.load(n, k);
      });
  Tensor* y = Compute(
      "y",
      {{M, "m2"}, {N, "n2"}, {K, "k2"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return c_buf.load(m, n) * d_buf.load(m, k) + x->load(m, n, k);
      });

  LoopNest l1({y}, {x, y});
  LoopNest l2(l1);
  l2.computeInline(x->buf());

  l1.prepareForCodegen();
  l2.prepareForCodegen();

  Stmt* stmt1 = IRSimplifier::simplify(l1.root_stmt());
  Stmt* stmt2 = IRSimplifier::simplify(l2.root_stmt());

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
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;
  Placeholder a_buf("a", kFloat, {M, N});
  Placeholder b_buf("b", kFloat, {N, K});
  Placeholder c_buf("c", kFloat, {M, N});
  Placeholder d_buf("d", kFloat, {M, K});

  Tensor* x = Compute(
      "x",
      {{M, "m1"}, {N, "n1"}, {K, "k1"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) * b_buf.load(n, k);
      });
  Tensor* y = Compute(
      "y",
      {{M, "m2"}, {N, "n2"}, {K, "k2"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return c_buf.load(m, n) * d_buf.load(m, k) + x->load(m, n, k);
      });
  Tensor* z = Compute(
      "z",
      {{M, "m3"}, {N, "n3"}, {K, "k3"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return x->load(m, n, k) + y->load(m, n, k);
      });

  LoopNest l({z}, {x, y, z});
  for (const std::string& order : inline_order) {
    if (order == "x") {
      l.computeInline(x->buf());
    } else if (order == "y") {
      l.computeInline(y->buf());
    } else {
      throw std::runtime_error("Invalid order: " + order);
    }
  }
  l.prepareForCodegen();
  Stmt* stmt = l.root_stmt();

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
    Tensor* z2 = Compute(
        "z",
        {{M, "m3"}, {N, "n3"}, {K, "k3"}},
        [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
          return a_buf.load(m, n) * b_buf.load(n, k) +
              (c_buf.load(m, n) * d_buf.load(m, k) +
               a_buf.load(m, n) * b_buf.load(n, k));
        });
    LoopNest l2({z2});
    l2.prepareForCodegen();
    Stmt* stmt2 = l2.root_stmt();

    std::ostringstream oss2;
    oss2 << *stmt2;
    std::string str2 = remove_space(oss2.str());

    ASSERT_EQ(str1, str2);
    ASSERT_GT(str1.size(), 100);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleInlineFunc01) {
  InlineFunc01Helper({"x", "y"});
  InlineFunc01Helper({"y", "x"});
  InlineFunc01Helper({"x"});
  InlineFunc01Helper({"y"});
  InlineFunc01Helper({});
}

// Make sure we cache random vars if we should.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleInlineRandom) {
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;

  Tensor* x = Compute(
      "x",
      {{M, "m1"}, {N, "n1"}, {K, "k1"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return Mod::make(Intrinsics::make(kRand, kInt), 5);
      });
  Tensor* y = Compute(
      "y",
      {{M, "m2"}, {N, "n2"}, {K, "k2"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return x->load(m, n, k) + x->load(m, n, k);
      });

  LoopNest l1({y}, {x, y});
  l1.computeInline(x->buf());

  // would normally compare results but Rand isn't implemented in the
  // SimpleIREvaluator, even if we could seed it.
  Stmt* stmt1 = IRSimplifier::simplify(l1.root_stmt());

  // Check the IR we produced
  checkIR(stmt1, R"IR(
# CHECK: for (int m2 = 0; m2 < 4; m2++)
# CHECK:   for (int n2 = 0; n2 < 5; n2++)
# CHECK:     for (int k2 = 0; k2 < 6; k2++)
# CHECK:       int x = rand();
# CHECK:       y[m2, n2, k2] = 2 * (x % 5);)IR");
}

// Make sure we don't cache random vars that are not being inlined.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleInlineRandomUnrelated) {
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;

  Tensor* x = Compute(
      "x",
      {{M, "m1"}, {N, "n1"}, {K, "k1"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return m * n * k;
      });
  Tensor* y = Compute(
      "y",
      {{M, "m2"}, {N, "n2"}, {K, "k2"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return x->load(m, n, k) + Intrinsics::make(kRand, kInt) +
            Intrinsics::make(kRand, kInt);
      });

  LoopNest l1({y}, {x, y});
  l1.computeInline(x->buf());

  // would normally compare results but Rand isn't implemented in the
  // SimpleIREvaluator, even if we could seed it.
  Stmt* stmt1 = IRSimplifier::simplify(l1.root_stmt());

  // Check the IR we produced
  checkIR(stmt1, R"IR(
# CHECK: for (int m2 = 0; m2 < 4; m2++)
# CHECK:   for (int n2 = 0; n2 < 5; n2++)
# CHECK:     for (int k2 = 0; k2 < 6; k2++)
# CHECK:       y[m2, n2, k2] = ((n2 * m2) * k2 + (rand())) + (rand());)IR");
}

// Make sure we generate the right number of random values == the dimensionality
// of the production tensor.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleInlineRandomLowerDimensions) {
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;

  Tensor* x = Compute("x", {{M, "m1"}}, [&](const VarHandle& m) {
    return Mod::make(Intrinsics::make(kRand, kInt), 5);
  });
  Tensor* y = Compute(
      "y",
      {{M, "m2"}, {N, "n2"}, {K, "k2"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return x->load(m) + x->load(m);
      });

  LoopNest l1({y}, {x, y});
  l1.computeInline(x->buf());

  // would normally compare results but Rand isn't implemented in the
  // SimpleIREvaluator, even if we could seed it.
  Stmt* stmt1 = IRSimplifier::simplify(l1.root_stmt());

  // Check the IR we produced
  checkIR(stmt1, R"IR(
# CHECK: for (int m2 = 0; m2 < 4; m2++)
# CHECK:   int x = rand();
# CHECK:   for (int n2 = 0; n2 < 5; n2++)
# CHECK:     for (int k2 = 0; k2 < 6; k2++)
# CHECK:       y[m2, n2, k2] = 2 * (x % 5);)IR");
}

// Make sure we don't screw up intrinsics thinking they're rand.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleInlineIntrinsics) {
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;
  Placeholder a_buf("a", kFloat, {M, N});
  Placeholder b_buf("b", kFloat, {N, K});

  Tensor* x = Compute(
      "x",
      {{M, "m1"}, {N, "n1"}, {K, "k1"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) * b_buf.load(n, k);
      });
  Tensor* y = Compute(
      "y",
      {{M, "m2"}, {N, "n2"}, {K, "k2"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return Intrinsics::make(kSqrt, x->load(m, n, k));
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
  l2.computeInline(x->buf());

  l1.prepareForCodegen();
  l2.prepareForCodegen();

  Stmt* stmt1 = IRSimplifier::simplify(l1.root_stmt());
  Stmt* stmt2 = IRSimplifier::simplify(l2.root_stmt());

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
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleInlineRandWithIntrinsics) {
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;

  Tensor* x = Compute(
      "x",
      {{M, "m1"}, {N, "n1"}, {K, "k1"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return Intrinsics::make(kRand, kFloat);
      });
  Tensor* y = Compute(
      "y",
      {{M, "m2"}, {N, "n2"}, {K, "k2"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return Intrinsics::make(kSqrt, x->load(m, n, k));
      });

  LoopNest l1({y}, {x, y});
  l1.computeInline(x->buf());

  Stmt* stmt1 = IRSimplifier::simplify(l1.root_stmt());

  // Check the IR we produced
  checkIR(stmt1, R"IR(
# CHECK: for (int m2 = 0; m2 < 4; m2++)
# CHECK:   for (int n2 = 0; n2 < 5; n2++)
# CHECK:     for (int k2 = 0; k2 < 6; k2++)
# CHECK:       float x = rand();
# CHECK:       y[m2, n2, k2] = sqrt(x);)IR");
}

// Split a Compute then inline it into another compute.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleSplitAThenInline) {
  KernelScope kernel_scope;
  Tensor* a =
      Compute("a", {{18, "i"}}, [&](const VarHandle& i) { return i * i; });
  Tensor* b = Compute("b", {{2, "j"}}, [&](const VarHandle& j) {
    return a->load(j + ExprHandle(8));
  });

  LoopNest l({b}, {a, b});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(a->buf()).at(0);
  l.splitWithMask(loops[0], 4);
  ASSERT_THROWS_WITH(l.computeInline(a->buf()), "compound indices");
}

// Split a Compute then inline another Compute into it.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleSplitBThenInline) {
  KernelScope kernel_scope;
  Tensor* a =
      Compute("a", {{18, "i"}}, [&](const VarHandle& i) { return i * i; });
  Tensor* b = Compute("b", {{6, "j"}}, [&](const VarHandle& j) {
    return a->load(j + ExprHandle(8));
  });

  LoopNest l({b}, {a, b});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(b->buf()).at(0);
  l.splitWithMask(loops[0], 3);
  l.computeInline(a->buf());
  l.prepareForCodegen();
  Stmt* s = IRSimplifier::simplify(l.root_stmt());

  std::vector<int> output(6, 0);
  SimpleIREvaluator eval(s, {b});
  eval(output);

  for (int i = 0; i < 6; ++i) {
    ASSERT_EQ(output[i], (i + 8) * (i + 8));
  }
}

// Split a Compute twice then inline it.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleSplitTwiceThenInline) {
  KernelScope kernel_scope;
  Tensor* a =
      Compute("a", {{18, "i"}}, [&](const VarHandle& i) { return i * i; });
  Tensor* b = Compute("b", {{2, "j"}}, [&](const VarHandle& j) {
    return a->load(j + ExprHandle(8));
  });
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* i_inner;

  LoopNest l({b}, {a, b});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(a->buf()).at(0);
  l.splitWithMask(loops[0], 4, &i_inner);
  l.splitWithMask(i_inner, 2);
  ASSERT_THROWS_WITH(l.computeInline(a->buf()), "compound indices");
}

// Inline a Compute, then split.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleInlineThenSplit) {
  KernelScope kernel_scope;
  Tensor* a =
      Compute("a", {{18, "i"}}, [&](const VarHandle& i) { return i * i; });
  Tensor* b = Compute("b", {{6, "j"}}, [&](const VarHandle& j) {
    return a->load(j + ExprHandle(8));
  });

  LoopNest l({b}, {a, b});
  l.computeInline(a->buf());

  std::vector<For*> loops = NodeFinder<For>::find(l.root_stmt());
  l.splitWithMask(loops.back(), 3);
  l.prepareForCodegen();
  Stmt* s = IRSimplifier::simplify(l.root_stmt());
  std::vector<int> output(6, 0);
  SimpleIREvaluator eval(s, {b});
  eval(output);

  for (int i = 0; i < 6; ++i) {
    ASSERT_EQ(output[i], (i + 8) * (i + 8));
  }
}

// Split a Compute, inline it, then split the result.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleSplitInlineThenSplit) {
  KernelScope kernel_scope;
  Tensor* a =
      Compute("a", {{18, "i"}}, [&](const VarHandle& i) { return i * i; });
  Tensor* b = Compute("b", {{16, "j"}}, [&](const VarHandle& j) {
    return a->load(j + ExprHandle(8));
  });

  LoopNest l({b}, {a, b});
  auto loops = NodeFinder<For>::find(l.root_stmt());
  l.splitWithMask(loops.back(), 2);
  l.computeInline(a->buf());

  loops = NodeFinder<For>::find(l.root_stmt());
  l.splitWithMask(loops.front(), 2);
  l.prepareForCodegen();
  Stmt* s = IRSimplifier::simplify(l.root_stmt());
  std::vector<int> output(16, 0);
  SimpleIREvaluator eval(s, {b});
  eval(output);

  for (int i = 0; i < 16; ++i) {
    ASSERT_EQ(output[i], (i + 8) * (i + 8));
  }
}

// Oversplit a loop that is simplified out after inlining.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleSplitInlineSimplify) {
  KernelScope kernel_scope;
  Tensor* a = Compute("a", {{18, "i"}}, [&](const VarHandle& i) {
    return ExprHandle(4) * i - ExprHandle(2) * i;
  });
  Tensor* b = Compute("b", {{2, "j"}}, [&](const VarHandle& j) {
    return a->load(j) - ExprHandle(1);
  });

  LoopNest l({b}, {a, b});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(a->buf()).at(0);
  l.splitWithMask(loops[0], 4);
  ASSERT_THROWS_WITH(l.computeInline(a->buf()), "compound indices");
}

// Inline a Compute with two consumers.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleInlineThreeMixedOnce) {
  KernelScope kernel_scope;
  Tensor* a =
      Compute("a", {{18, "i"}}, [&](const VarHandle& i) { return i * i; });
  Tensor* b = Compute("b", {{6, "j"}}, [&](const VarHandle& j) {
    return a->load(j + ExprHandle(8));
  });
  Tensor* c = Compute(
      "c", {{4, "k"}, {3, "l"}}, [&](const VarHandle& k, const VarHandle& l) {
        return a->load(k) * b->load(l);
      });

  LoopNest l({c}, {a, b, c});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(a->buf()).at(0);
  l.computeInline(a->buf());
  l.prepareForCodegen();

  Stmt* s = IRSimplifier::simplify(l.root_stmt());
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
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleInlineThreeMixedTwice) {
  KernelScope kernel_scope;
  Tensor* a =
      Compute("a", {{18, "i"}}, [&](const VarHandle& i) { return i * i; });
  Tensor* b = Compute("b", {{6, "j"}}, [&](const VarHandle& j) {
    return a->load(j + ExprHandle(8));
  });
  Tensor* c = Compute(
      "c", {{4, "k"}, {3, "l"}}, [&](const VarHandle& k, const VarHandle& l) {
        return a->load(k) * b->load(l);
      });

  LoopNest l({c}, {a, b, c});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(a->buf()).at(0);
  l.computeInline(a->buf());
  l.computeInline(b->buf());
  l.prepareForCodegen();

  Stmt* s = IRSimplifier::simplify(l.root_stmt());
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
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleInlineThreeMixedInner) {
  KernelScope kernel_scope;
  Tensor* a =
      Compute("a", {{18, "i"}}, [&](const VarHandle& i) { return i * i; });
  Tensor* b = Compute("b", {{6, "j"}}, [&](const VarHandle& j) {
    return a->load(j + ExprHandle(8));
  });
  Tensor* c = Compute(
      "c", {{4, "k"}, {3, "l"}}, [&](const VarHandle& k, const VarHandle& l) {
        return a->load(k) * b->load(l);
      });

  LoopNest l({c}, {a, b, c});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(a->buf()).at(0);
  l.computeInline(b->buf());
  l.prepareForCodegen();

  Stmt* s = IRSimplifier::simplify(l.root_stmt());
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
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleInlineThreeMixedSplit) {
  KernelScope kernel_scope;
  Tensor* a =
      Compute("a", {{18, "i"}}, [&](const VarHandle& i) { return i * i; });
  Tensor* b = Compute("b", {{6, "j"}}, [&](const VarHandle& j) {
    return a->load(j + ExprHandle(8));
  });
  Tensor* c = Compute(
      "c", {{4, "k"}, {3, "l"}}, [&](const VarHandle& k, const VarHandle& l) {
        return a->load(k) * b->load(l);
      });

  LoopNest l({c}, {a, b, c});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(a->buf()).at(0);
  l.splitWithMask(loops[0], 4);
  loops = l.getAllLoopNestsWritingToBuf(b->buf()).at(0);
  l.splitWithMask(loops[0], 3);
  loops = l.getAllLoopNestsWritingToBuf(c->buf()).at(0);
  l.splitWithMask(loops[0], 2);

  ASSERT_THROWS_WITH(l.computeInline(a->buf()), "compound indices");
}

// Check that inlining works for output tensors too
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleInlineOutputTensors) {
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;

  Tensor* x = Compute(
      "x",
      {{M, "m1"}, {N, "n1"}, {K, "k1"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return m * n * k;
      });
  Tensor* y = Compute(
      "y",
      {{M, "m2"}, {N, "n2"}, {K, "k2"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return x->load(m, n, k) + m;
      });

  LoopNest l1({x, y});
  l1.computeInline(x->buf());

  // would normally compare results but Rand isn't implemented in the
  // SimpleIREvaluator, even if we could seed it.
  Stmt* stmt1 = IRSimplifier::simplify(l1.root_stmt());

  // Check the IR we produced
  checkIR(stmt1, R"IR(
# CHECK: for (int m1 = 0; m1 < 4; m1++)
# CHECK:   for (int n1 = 0; n1 < 5; n1++)
# CHECK:     for (int k1 = 0; k1 < 6; k1++)
# CHECK:       x[m1, n1, k1] = (n1 * m1) * k1;
# CHECK: for (int m2 = 0; m2 < 4; m2++)
# CHECK:   for (int n2 = 0; n2 < 5; n2++)
# CHECK:     for (int k2 = 0; k2 < 6; k2++)
# CHECK:       y[m2, n2, k2] = (n2 * m2) * k2 + m2;)IR");
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleFuserStyle) {
  KernelScope kernel_scope;
  const int kVectorSize = 8;
  const int kVectorCount = 128;
  const int kTotalSize = kVectorSize * kVectorCount;

  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));

  Tensor* b = Compute(
      "f", {{kTotalSize, "i"}}, [&](const std::vector<VarHandle>& axes) {
        return a_buf.load(axes[0]) + 11.0f;
      });

  Tensor* c = Compute(
      "g", {{kTotalSize, "i"}}, [&](const std::vector<VarHandle>& axes) {
        return b->load(axes[0]) + 1.0f;
      });

  LoopNest l({b, c});
  l.prepareForCodegen();
  Stmt* s = l.root_stmt();

  std::vector<float> a_data(kTotalSize, 7.0f);
  std::vector<float> b_data(kTotalSize, 0.0f);
  std::vector<float> c_data(kTotalSize, 0.0f);
  SimpleIREvaluator(s, {a_buf, b, c})(a_data, b_data, c_data);

  for (int i = 0; i < kTotalSize; i++) {
    ASSERT_EQ(b_data[i], 18.0f);
    ASSERT_EQ(c_data[i], 19.0f);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleFuserThreeArg) {
  KernelScope kernel_scope;
  const int kVectorSize = 8;
  const int kVectorCount = 128;
  const int kTotalSize = kVectorSize * kVectorCount;

  Placeholder a(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder c(BufHandle("C", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder d(BufHandle("D", {ExprHandle(kTotalSize)}, kFloat));

  Tensor* e = Compute("e", {{kTotalSize, "i"}}, [&](const VarHandle& i) {
    return a.load(i) + b.load(i);
  });
  Tensor* f = Compute("f", {{kTotalSize, "i"}}, [&](const VarHandle& i) {
    return e->load(i) + c.load(i);
  });
  Tensor* g = Compute("g", {{kTotalSize, "i"}}, [&](const VarHandle& i) {
    return f->load(i) + d.load(i);
  });

  LoopNest l({g}, {e, f, g});
  l.computeInline(l.getLoopBodyFor(e));
  l.computeInline(l.getLoopBodyFor(f));
  l.prepareForCodegen();
  Stmt* s = l.root_stmt();

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ScheduleDynamicShape2D) {
  KernelScope kernel_scope;
  auto testWithSize = [](int32_t M, int32_t N) {
    VarHandle m("m", kInt);
    VarHandle n("n", kInt);
    Placeholder a(BufHandle("a", {m, n}, kFloat));
    Placeholder b(BufHandle("b", {m, n}, kFloat));
    Tensor* c = Compute(
        "c", {{m, "m"}, {n, "n"}}, [&](const VarHandle& i, const VarHandle& j) {
          return a.load(i, j) + b.load(i, j);
        });
    LoopNest l({c});
    Stmt* s = l.root_stmt();
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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
  KernelScope kernel_scope;
  VarHandle N("N", kInt);
  Tensor* A = Compute(
      "A", {{N, "i_a"}}, [&](const VarHandle& i_a) { return i_a * i_a; });
  Tensor* B = Compute(
      "B", {{N, "i_b"}}, [&](const VarHandle& i_b) { return A->load(i_b); });
  LoopNest l({B}, {A, B});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(B->buf()).at(0);
  l.computeAt(l.getLoopBodyFor(A), loops[0]);
  l.prepareForCodegen();
  Stmt* s = l.root_stmt();

  checkIR(s, R"IR(
# CHECK: for (int i_b = 0; i_b < N; i_b++)
# CHECK:   Allocate(temp); // dtype=int, dims=[1]
# CHECK:   temp[
# CHECK-NOT: A[
# CHECK:   B[i_b] = temp[0]
# CHECK:   Free(temp))IR");

  // Now check that the loop still produces the correct result.
  std::vector<int> b_data(100, 0);
  SimpleIREvaluator cg(s, {B, N});
  cg.call({b_data, 100});

  std::vector<int> b_ref(100, 0);
  for (int i = 0; i < 100; i++) {
    b_ref[i] = i * i;
  }
  assertAllEqual(b_data, b_ref);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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
  KernelScope kernel_scope;

  const int kW = 16, kH = 16;
  VarHandle W("W", kInt);
  VarHandle H("H", kInt);
  Tensor* p = Compute(
      "prod",
      {{H + 1, "py"}, {W + 1, "px"}},
      [&](const VarHandle& py, const VarHandle& px) { return px * py; });
  Tensor* c = Compute(
      "cons",
      {{H, "cy"}, {W, "cx"}},
      [&](const VarHandle& y, const VarHandle& x) {
        return p->load(y, x) + p->load(y + 1, x) + p->load(y, x + 1) +
            p->load(y + 1, x + 1);
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
    std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(c->buf()).at(0);
    l.computeAt(l.getLoopBodyFor(p), loops[0]);
    l.prepareForCodegen();
    Stmt* s = l.root_stmt();

    // Check the IR we produced
    checkIR(s, R"IR(
# CHECK: for (int cy = 0; cy < H; cy++)
# CHECK:   Allocate(temp); // dtype=int, dims=[2, W + 1]
# CHECK:   for
# CHECK:     for
# CHECK:   for (int cx = 0; cx < W; cx++)
# CHECK-NOT: prod[
# CHECK:     cons[
# CHECK:   Free(temp))IR");

    // Now check that the loop still produces the correct result.
    std::vector<int> c_data(kW * kH, 0);
    SimpleIREvaluator cg(s, {c, W, H});
    cg.call({c_data, kW, kH});

    assertAllEqual(c_data, c_ref);
  }
  {
    // Now let's try to compute P at axis cx (the inner loop)
    LoopNest l(orig_loopnest);
    std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(c->buf()).at(0);
    l.computeAt(l.getLoopBodyFor(p), loops[1]);
    l.prepareForCodegen();
    Stmt* s = l.root_stmt();

    // Check the IR we produced
    checkIR(s, R"IR(
# CHECK: for (int cy = 0; cy < H; cy++)
# CHECK:   for (int cx = 0; cx < W; cx++)
# CHECK:     Allocate(temp); // dtype=int, dims=[2, 2]
# CHECK:     for
# CHECK:       for
# CHECK-NOT: prod[
# CHECK:     cons[
# CHECK:     Free(temp))IR");

    // Now check that the loop still produces the correct result.
    std::vector<int> c_data(kW * kH, 0);
    SimpleIREvaluator cg(s, {c, W, H});
    cg.call({c_data, kW, kH});

    assertAllEqual(c_data, c_ref);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, LoopNestComputeAt_3) {
  // Verify that compute_at works on the following example:
  //
  // A(x,y) = x*y
  // B(x,y) = A(x, y)
  // C(x,y) = B(x+1, y)
  // D(x,y) = A(x, y+1) + C(x, y)
  //
  // i.e. when 'A' comes to 'D' directly and indirectly through 'C'.
  KernelScope kernel_scope;

  const int kW = 16, kH = 16;
  VarHandle W("W", kInt);
  VarHandle H("H", kInt);
  Tensor* A = Compute(
      "A",
      {{H + 1, "ay"}, {W + 1, "ax"}},
      [&](const VarHandle& ay, const VarHandle& ax) { return ax * ay; });
  Tensor* B = Compute(
      "B",
      {{H + 1, "by"}, {W + 1, "bx"}},
      [&](const VarHandle& by, const VarHandle& bx) {
        return A->load(by, bx);
      });
  Tensor* C = Compute(
      "C",
      {{H, "cy"}, {W, "cx"}},
      [&](const VarHandle& cy, const VarHandle& cx) {
        return B->load(cy, cx + 1);
      });
  Tensor* D = Compute(
      "D",
      {{H, "dy"}, {W, "dx"}},
      [&](const VarHandle& dy, const VarHandle& dx) {
        return A->load(dy + 1, dx) + C->load(dy, dx);
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
    std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(D->buf()).at(0);
    l.computeAt(l.getLoopBodyFor(A), loops[0]);
    l.prepareForCodegen();
    Stmt* s = l.root_stmt();

    // Check the IR we produced
    checkIR(s, R"IR(
# CHECK: for (int ay = 0; ay < H + 1; ay++)
# CHECK:   for (int ax = 0; ax < W + 1; ax++)
# CHECK:     A[
# CHECK: for (int by = 0; by < H + 1; by++)
# CHECK:   for (int bx = 0; bx < W + 1; bx++)
# CHECK:     B[
# CHECK: for (int cy = 0; cy < H; cy++)
# CHECK:   for (int cx = 0; cx < W; cx++)
# CHECK:     C[
# CHECK: for (int dy = 0; dy < H; dy++)
# CHECK:   Allocate(temp); // dtype=int, dims=[1, W]
# CHECK:   for (int dx = 0; dx < W; dx++)
# CHECK-NOT: A[)IR");

    // Now check that the loop still produces the correct result.
    std::vector<int> c_data(kW * kH, 0);
    SimpleIREvaluator cg(s, {D, W, H});
    cg.call({c_data, kW, kH});

    assertAllEqual(c_data, c_ref);
  }
  {
    // Now let's try to compute A at axis dx (the inner loop)
    LoopNest l(orig_loopnest);
    std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(D->buf()).at(0);
    l.computeAt(l.getLoopBodyFor(A), loops[1]);
    l.prepareForCodegen();
    Stmt* s = l.root_stmt();

    // Check the IR we produced
    checkIR(s, R"IR(
# CHECK: for (int ay = 0; ay < H + 1; ay++)
# CHECK:   for (int ax = 0; ax < W + 1; ax++)
# CHECK:     A[
# CHECK: for (int by = 0; by < H + 1; by++)
# CHECK:   for (int bx = 0; bx < W + 1; bx++)
# CHECK:     B[
# CHECK: for (int cy = 0; cy < H; cy++)
# CHECK:   for (int cx = 0; cx < W; cx++)
# CHECK:     C[
# CHECK: for (int dy = 0; dy < H; dy++)
# CHECK:   for (int dx = 0; dx < W; dx++)
# CHECK:     Allocate(temp); // dtype=int, dims=[1, 1]
# CHECK-NOT: A[)IR");

    // Now check that the loop still produces the correct result.
    std::vector<int> c_data(kW * kH, 0);
    SimpleIREvaluator cg(s, {D, W, H});
    cg.call({c_data, kW, kH});

    assertAllEqual(c_data, c_ref);
  }
}

using Axis = const VarHandle&;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, Reduce2dComputeAt) {
  KernelScope kernel_scope;

  const int kW = 16, kH = 16;
  VarHandle W("W", kInt);
  VarHandle H("H", kInt);

  Tensor* p =
      Compute("prod", {{H + 1, "py"}, {W + 1, "px"}}, [&](Axis py, Axis px) {
        return px * py;
      });
  Tensor* c = Reduce(
      "cons",
      {{H, "cy"}, {W, "cx"}},
      Sum(),
      [&](Axis y, Axis x, Axis r, Axis s) { return p->load(y + r, x + s); },
      {{2, "r"}, {2, "s"}});

  std::vector<int> c_ref(kW * kH, 0);
  for (int y = 0; y < kH; y++) {
    for (int x = 0; x < kW; x++) {
      c_ref[y * kW + x] = y * x + (y + 1) * x + y * (x + 1) + (y + 1) * (x + 1);
    }
  }
  LoopNest orig_loopnest({c}, {p, c});
  checkIR(orig_loopnest.root_stmt(), R"IR(
# CHECK: for (int py = 0; py < H + 1; py++) {
# CHECK:   for (int px = 0; px < W + 1; px++) {
# CHECK:     prod[py, px] = px * py;
# CHECK:   }
# CHECK: }
# CHECK: for (int cy = 0; cy < H; cy++) {
# CHECK:   for (int cx = 0; cx < W; cx++) {
# CHECK:     cons[cy, cx] = int(0);
# CHECK:     for (int r = 0; r < 2; r++) {
# CHECK:       for (int s = 0; s < 2; s++) {
# CHECK:         cons[cy, cx] = ReduceOp((cons[cy, cx]) + (prod[cy + r, cx + s]), reduce_args={r, s});
# CHECK:       }
# CHECK:     }
# CHECK:   }
# CHECK: }
)IR");

  {
    // First let's try to compute P at axis cy (the outer loop)
    LoopNest l(orig_loopnest);
    auto loops = l.getAllLoopNestsWritingToBuf(c->buf()).at(0);
    l.computeAt(l.getLoopBodyFor(p), loops[0]);
    // FIXME: Calling simplify here breaks the IR:
    // MALFORMED INPUT: could not find base node in Load - temp[...]
    // l.simplify();
    l.eliminateDeadStores();
    l.prepareForCodegen();
    checkIR(l.root_stmt(), R"IR(
# CHECK: for (int cy = 0; cy < H; cy++) {
# CHECK:   Allocate(temp); // dtype=int, dims=[2, W + 1]
# CHECK:   for (int idx0 = 0; idx0 < 2; idx0++) {
# CHECK:     for (int idx1 = 0; idx1 < W + 1; idx1++) {
# CHECK:       temp[(0 + idx0 * (1 * (W + 1))) + idx1 * 1] = (idx0 + cy) * (idx1 + 0);
# CHECK:     }
# CHECK:   }
# CHECK:   for (int cx = 0; cx < W; cx++) {
# CHECK:     cons[(0 + cy * (1 * W)) + cx * 1] = int(0);
# CHECK:     for (int r = 0; r < 2; r++) {
# CHECK:       for (int s = 0; s < 2; s++) {
# CHECK:         cons[(0 + cy * (1 * W)) + cx * 1] = (cons[(0 + cy * (1 * W)) + cx * 1]) + (temp[(0 + r * (1 * (W + 1))) + (s + cx) * 1]);
# CHECK:       }
# CHECK:     }
# CHECK:   }
# CHECK:   Free(temp);
# CHECK: }
)IR");
    Stmt* s = l.root_stmt();

    // Now check that the loop still produces the correct result.
    std::vector<int> c_data(kW * kH, 0);
    SimpleIREvaluator cg(s, {c, W, H});
    cg.call({c_data, kW, kH});
    assertAllEqual(c_data, c_ref);
  }
  {
    // Now let's try to compute P at axis cx (the inner loop)
    LoopNest l(orig_loopnest);
    std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(c->buf()).at(0);
    l.computeAt(l.getLoopBodyFor(p), loops[1]);
    l.simplify();
    l.eliminateDeadStores();
    l.prepareForCodegen();
    checkIR(l.root_stmt(), R"IR(
# CHECK: for (int cy = 0; cy < H; cy++) {
# CHECK:   for (int cx = 0; cx < W; cx++) {
# CHECK:     Allocate(temp); // dtype=int, dims=[2, 2]
# CHECK:     for (int idx0 = 0; idx0 < 2; idx0++) {
# CHECK:       for (int idx1 = 0; idx1 < 2; idx1++) {
# CHECK:         temp[(0 + idx0 * (1 * 2)) + idx1 * 1] = (cy + idx0) * (cx + idx1);
# CHECK:       }
# CHECK:     }
# CHECK:     cons[(0 + cy * (1 * W)) + cx * 1] = 0;
# CHECK:     for (int r = 0; r < 2; r++) {
# CHECK:       for (int s = 0; s < 2; s++) {
# CHECK:         cons[(0 + cy * (1 * W)) + cx * 1] = (cons[(0 + cy * (1 * W)) + cx * 1]) + (temp[(0 + r * (1 * 2)) + s * 1]);
# CHECK:       }
# CHECK:     }
# CHECK:     Free(temp);
# CHECK:   }
# CHECK: }
)IR");
    Stmt* s = l.root_stmt();

    // Now check that the loop still produces the correct result.
    std::vector<int> c_data(kW * kH, 0);
    SimpleIREvaluator cg(s, {c, W, H});
    cg.call({c_data, kW, kH});
    assertAllEqual(c_data, c_ref);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, DISABLED_Conv1d_NH) {
  // Lots of stuff is broken here.  The computeAt swaps the axes for some odd
  // reason.  Even without that, the index flattener fails due to "dimensions
  // mismatch in flatten index".
  KernelScope kernel_scope;

  int N = 4;
  int H = 256;
  int R = 3;
  int Pad = 1;
  Placeholder IP("input", kFloat, {H});

  Tensor* A =
      Compute("A", {{N, "np"}, {H + 2 * Pad, "hp"}}, [&](Axis n, Axis h) {
        auto cond = CompareSelect::make(h, Pad, 1, 0, kLT);
        cond = CompareSelect::make(h, H + Pad, 1, cond, kGE);
        return ifThenElse(cond, 0.f, IP.load(n, h - Pad));
      });
  Tensor* B = Reduce(
      "B",
      {{N, "n"}, {H, "h"}},
      Sum(),
      [&](Axis n, Axis h, Axis r) { return A->load(n, h + r); },
      {{R, "r"}});
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
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(B->buf()).at(0);
  l.computeAt(l.getLoopBodyFor(A), loops[0]);
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
  Stmt* s = l.root_stmt();

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
  std::string getOrder(Stmt* s) {
    ordering.str("");
    s->accept(this);
    return ordering.str();
  }

  // NOLINTNEXTLINE(cppcoreguidelines-explicit-virtual-functions,modernize-use-override)
  void visit(const For* v) {
    ordering << v->var()->name_hint() << ",";
    IRVisitor::visit(v);
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, LoopNestReorderAxis1) {
  KernelScope kernel_scope;
  Tensor* tensor = Compute(
      "f", {{2, "x"}, {3, "y"}}, [](const VarHandle& x, const VarHandle& y) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  LoopNest l({tensor});
  Stmt* stmt1 = Stmt::clone(l.root_stmt());

  std::vector<int> stmt1_output(6, 0);
  SimpleIREvaluator cg(stmt1, {tensor});
  cg.call({stmt1_output});

  auto loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  l.reorderAxis(loops[0], loops[1]);
  Stmt* stmt2 = Stmt::clone(l.root_stmt());

  ASSERT_NE(stmt1, stmt2);
  LoopOrderHelper loopOrderHelper;
  std::string order1 = loopOrderHelper.getOrder(stmt1);
  std::string order2 = loopOrderHelper.getOrder(stmt2);

  ASSERT_EQ(order1, "x,y,");
  ASSERT_EQ(order2, "y,x,");

  std::vector<int> stmt2_output(6, 0);
  SimpleIREvaluator cg2(stmt2, {tensor});
  cg.call({stmt2_output});

  for (int i = 0; i < 6; ++i) {
    ASSERT_EQ(stmt1_output[i], stmt2_output[i]);
  }

  // Reorder them back.
  loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  l.reorderAxis(loops[0], loops[1]);
  Stmt* stmt3 = l.root_stmt();

  std::string order3 = loopOrderHelper.getOrder(stmt3);
  ASSERT_EQ(order3, order1);

  std::ostringstream oss1, oss2;
  oss1 << *stmt1;
  oss2 << *stmt3;

  // Should be identical to the unreordered statement.
  ASSERT_EQ(oss1.str(), oss2.str());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, LoopNestReorderPartialAxes) {
  KernelScope kernel_scope;
  Tensor* tensor = Compute(
      "f",
      {{2, "x"}, {3, "y"}, {4, "z"}},
      [](const VarHandle& x, const VarHandle& y, const VarHandle& z) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y +
            cast<float>(z) * z;
      });
  LoopNest l({tensor});

  LoopOrderHelper loopOrderHelper;
  Stmt* stmt1 = Stmt::clone(l.root_stmt());
  ASSERT_EQ(loopOrderHelper.getOrder(stmt1), "x,y,z,");

  std::vector<int> stmt1_output(24, 0);
  SimpleIREvaluator cg(stmt1, {tensor});
  cg.call({stmt1_output});

  auto loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  l.reorderAxis(loops[0], loops[1]);
  ASSERT_EQ(loopOrderHelper.getOrder(l.root_stmt()), "y,x,z,");

  Stmt* stmt2 = Stmt::clone(l.root_stmt());

  std::vector<int> stmt2_output(24, 0);
  SimpleIREvaluator cg2(stmt2, {tensor});
  cg2.call({stmt2_output});

  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(stmt1_output[i], stmt2_output[i]);
  }

  loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  l.reorderAxis(loops[1], loops[2]);
  ASSERT_EQ(loopOrderHelper.getOrder(l.root_stmt()), "y,z,x,");

  Stmt* stmt3 = Stmt::clone(l.root_stmt());

  std::vector<int> stmt3_output(24, 0);
  SimpleIREvaluator cg3(stmt3, {tensor});
  cg3.call({stmt3_output});

  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(stmt1_output[i], stmt3_output[i]);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, LoopNestReorderInternalAxis) {
  KernelScope kernel_scope;
  Tensor* tensor = Compute(
      "f",
      {{1, "w"}, {2, "x"}, {3, "y"}, {4, "z"}},
      [](const VarHandle& w,
         const VarHandle& x,
         const VarHandle& y,
         const VarHandle& z) {
        return ExprHandle(1.0f) + w + cast<float>(x) * x + cast<float>(y) * y +
            cast<float>(z) * z;
      });
  LoopNest l({tensor});

  LoopOrderHelper loopOrderHelper;
  Stmt* stmt1 = Stmt::clone(l.root_stmt());
  ASSERT_EQ(loopOrderHelper.getOrder(stmt1), "w,x,y,z,");

  std::vector<int> stmt1_output(24, 0);
  SimpleIREvaluator cg(stmt1, {tensor});
  cg.call({stmt1_output});

  auto loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  l.reorderAxis(loops[2], loops[1]);
  ASSERT_EQ(loopOrderHelper.getOrder(l.root_stmt()), "w,y,x,z,");

  Stmt* stmt2 = l.root_stmt();

  std::vector<int> stmt2_output(24, 0);
  SimpleIREvaluator cg2(stmt2, {tensor});
  cg2.call({stmt2_output});

  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(stmt1_output[i], stmt2_output[i]);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, LoopNestReorderEnclosingAxis) {
  KernelScope kernel_scope;
  Tensor* tensor = Compute(
      "f",
      {{1, "w"}, {2, "x"}, {3, "y"}, {4, "z"}},
      [](const VarHandle& w,
         const VarHandle& x,
         const VarHandle& y,
         const VarHandle& z) {
        return ExprHandle(1.0f) + w + cast<float>(x) * x + cast<float>(y) * y +
            cast<float>(z) * z;
      });
  LoopNest l({tensor});

  LoopOrderHelper loopOrderHelper;
  Stmt* stmt1 = Stmt::clone(l.root_stmt());

  std::vector<int> stmt1_output(24, 0);
  SimpleIREvaluator cg(stmt1, {tensor});
  cg.call({stmt1_output});

  auto loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  l.reorderAxis(loops[0], loops[3]);
  ASSERT_EQ(loopOrderHelper.getOrder(l.root_stmt()), "z,x,y,w,");

  Stmt* stmt2 = l.root_stmt();

  std::vector<int> stmt2_output(24, 0);
  SimpleIREvaluator cg2(stmt2, {tensor});
  cg2.call({stmt2_output});

  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(stmt1_output[i], stmt2_output[i]);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, LoopNestReorderSameAxis) {
  KernelScope kernel_scope;
  Tensor* tensor = Compute(
      "f", {{2, "x"}, {3, "y"}}, [](const VarHandle& x, const VarHandle& y) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  LoopNest l({tensor});
  Stmt* stmt1 = Stmt::clone(l.root_stmt());

  auto loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  l.reorderAxis(loops[1], loops[1]);
  Stmt* stmt2 = Stmt::clone(l.root_stmt());

  std::ostringstream oss, oss2;
  oss << *stmt1;
  oss2 << *stmt2;
  ASSERT_EQ(oss.str(), oss2.str());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, LoopNestReorderExtraStatements) {
  /* We're going for a structure like this:
   * for x in ...
   *   Stmt 1
   *   for y in ...
   *     Stmt 2
   *     for z in ...
   *       Stmt 3
   *     Stmt 4
   */

  KernelScope kernel_scope;

  Tensor* tensor = Compute(
      "f",
      {{2, "x"}, {3, "y"}, {4, "z"}},
      [](const VarHandle& x, const VarHandle& y, const VarHandle& z) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y +
            cast<float>(z) * z;
      });
  LoopNest l({tensor});

  Placeholder extra(BufHandle("res", {6, 3}, kFloat));

  auto loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);

  VarHandle i = VarHandle(loops[0]->var());

  Stmt* store_1 = Store::make(BufHandle(extra.data()), {i, 0}, ExprHandle(1.f));
  Stmt* store_2 = Store::make(BufHandle(extra.data()), {i, 1}, ExprHandle(2.f));
  // stmt 3 is the Function body.
  Stmt* store_3 = Store::make(BufHandle(extra.data()), {i, 2}, ExprHandle(4.f));

  loops[0]->body()->prepend_stmt(store_1);
  loops[1]->body()->prepend_stmt(store_2);
  loops[1]->body()->append_stmt(store_3);
  Stmt* stmt1 = Stmt::clone(l.root_stmt());

  std::vector<int> extra1(6, 0);
  std::vector<int> res1(24, 0);
  SimpleIREvaluator cg(stmt1, {tensor, extra});
  cg.call({res1, extra1});

  /* Then we reorder loop y and z, we want it to look like:
   *
   * for x in ...
   *   Stmt 1
   *   for y in ...
   *     Stmt 2
   *   for z in ...
   *    for y in ...
   *       Stmt 3
   *   for y in ...
   *     Stmt 4
   *
   * We need extra loops because we don't have dependency info about stmt 3
   * and 4.
   *
   */

  l.reorderAxis(loops[1], loops[2]);
  Stmt* stmt2 = Stmt::clone(l.root_stmt());

  // Check the IR we produced
  checkIR(stmt2, R"IR(
# CHECK: for (int x
# CHECK:   res[x, 0] = 1
# CHECK:   for (int y
# CHECK:     res[x, 1] = 2
# CHECK:   for (int z
# CHECK:     for (int y
# CHECK:       f[
# CHECK:   for (int y
# CHECK:     res[x, 2] = 4
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
  loops = l.getAllLoopNestsWritingToBuf(tensor->buf()).at(0);
  l.reorderAxis(loops[0], loops[2]);
  Stmt* stmt3 = Stmt::clone(l.root_stmt());

  // Check the IR we produced
  checkIR(stmt3, R"IR(
# CHECK: for (int x
# CHECK:   res[x, 0] = 1
# CHECK:   for (int y
# CHECK:     res[x, 1] = 2
# CHECK: for (int y
# CHECK:   for (int z
# CHECK:     for (int x
# CHECK:       f[
# CHECK: for (int x
# CHECK:   for (int y
# CHECK:     res[x, 2] = 4
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
  KernelScope kernel_scope;

  Tensor* c = Compute(
      "5d",
      {{2, "a"}, {3, "b"}, {2, "c"}, {3, "d"}, {2, "e"}},
      [](const std::vector<VarHandle>&) { return -1; });
  LoopNest l({c});

  Placeholder extra(BufHandle("extra", {5}, kInt));

  auto loops = l.getAllLoopNestsWritingToBuf(c->buf()).at(0);
  int j = 0;
  for (auto* l : loops) {
    // Add an increment at each layer of the loop which counts the number of
    // times the loop executes.
    Load* load = new Load(extra.data(), {new IntImm(j)});
    Add* add = new Add(load, new IntImm(1));
    Stmt* store = new Store(extra.data(), {new IntImm(j)}, add);
    if (prepend) {
      l->body()->prepend_stmt(store);
    }
    if (append) {
      l->body()->append_stmt(Stmt::clone(store));
    }

    j++;
  }

  Stmt* stmt1 = Stmt::clone(l.root_stmt());

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

  loops = l.getAllLoopNestsWritingToBuf(c->buf()).at(0);
  l.reorderAxis(loops[index1], loops[index2]);
  Stmt* stmt2 = Stmt::clone(l.root_stmt());

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, LoopNestReorderInternalLoopNest) {
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;
  Placeholder a_buf("a", kFloat, {M, N});
  Placeholder b_buf("b", kFloat, {N, K});
  Placeholder c_buf("c", kFloat, {M, N});
  Placeholder d_buf("d", kFloat, {M, K});

  Tensor* x = Compute(
      "x",
      {{M, "m1"}, {N, "n1"}, {K, "k1"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) * b_buf.load(n, k);
      });
  Tensor* y = Compute(
      "y",
      {{M, "m2"}, {N, "n2"}, {K, "k2"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return c_buf.load(m, n) * d_buf.load(m, k) + x->load(m, n, k);
      });
  Tensor* z = Compute(
      "z",
      {{M, "m3"}, {N, "n3"}, {K, "k3"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return x->load(m, n, k) + y->load(m, n, k);
      });

  LoopNest l({z}, {x, y, z});
  For* a = nullptr;
  For* b = nullptr;
  auto fors = NodeFinder<For>::find(l.root_stmt());
  for (auto* f : fors) {
    if (f->var()->name_hint() == "m2") {
      a = f;
    } else if (f->var()->name_hint() == "k2") {
      b = f;
    }
  }
  l.reorderAxis(a, b);

  l.prepareForCodegen();
  Stmt* stmt = IRSimplifier::simplify(l.root_stmt());

  // Check the IR we produced has the 3 nests in the right order, but k and m
  // swapped in the middle.
  checkIR(stmt, R"IR(
# CHECK: for (int m1
# CHECK:   for (int n1
# CHECK:     for (int k1
# CHECK: for (int k2
# CHECK:   for (int n2
# CHECK:     for (int m2
# CHECK: for (int m3
# CHECK:   for (int n3
# CHECK:     for (int k3)IR");

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, OuterLoopVectorization) {
  KernelScope kernel_scope;
  Tensor* tensor = Compute(
      "f", {{8, "X"}, {8, "y"}}, [](const VarHandle& x, const VarHandle& y) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  LoopNest l({tensor});

  l.vectorize(l.getAllLoopNestsWritingToBuf(tensor->buf())[0][0]);

  Stmt* root_stmt = l.root_stmt();
  Block* outer_block = dynamic_cast<Block*>(root_stmt);
  ASSERT_NE(outer_block, nullptr);
  while (Block* inner_block = dynamic_cast<Block*>(outer_block->front())) {
    outer_block = inner_block;
  }

  // Verify that we have only a single loop level remaining after
  // vectorization.
  ASSERT_EQ(outer_block->nstmts(), 1);
  For* for_loop = dynamic_cast<For*>(outer_block->front());
  ASSERT_NE(for_loop, nullptr);
  Block* for_body = for_loop->body();
  ASSERT_EQ(for_body->nstmts(), 1);
  ASSERT_EQ(dynamic_cast<For*>(for_body->front()), nullptr);
}

namespace {

std::string constantUpperBoundLoopIR(int upper_bound_val) {
  KernelScope kernel_scope;
  ExprHandle upper_bound(upper_bound_val);
  Tensor* A = Compute(
      "A", {{upper_bound, "x"}}, [&](const VarHandle& x) { return x * 2; });
  LoopNest l({A});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(A->buf())[0];
  Stmt* unrolled = nullptr;
  LoopNest::unroll(loops[0], &unrolled);
  std::ostringstream oss;
  oss << *unrolled;
  return oss.str();
}

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, Unroll) {
  const std::string actual = constantUpperBoundLoopIR(3);
  const std::string& verification_pattern =
      R"IR(
# CHECK: A[0] = 0;
# CHECK: A[1] = 2;
# CHECK: A[2] = 4)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, actual);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, UnrollOuter) {
  KernelScope kernel_scope;
  ExprHandle outer_bound(3);
  ExprHandle inner_bound(4);
  Tensor* A = Compute(
      "A",
      {{outer_bound, "x"}, {inner_bound, "y"}},
      [&](const VarHandle& x, const VarHandle& y) { return x + y; });
  LoopNest l({A});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(A->buf())[0];
  Stmt* unrolled = nullptr;
  LoopNest::unroll(loops[0], &unrolled);
  checkIR(unrolled, R"IR(
# CHECK: for (int y = 0; y < 4; y++) {
# CHECK: A[0, y] = y;
# CHECK: }
# CHECK: for (int y = 0; y < 4; y++) {
# CHECK: A[1, y] = y + 1;
# CHECK: }
# CHECK: for (int y = 0; y < 4; y++) {
# CHECK: A[2, y] = y + 2;
# CHECK: })IR");
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, UnrollInner) {
  KernelScope kernel_scope;
  ExprHandle outer_bound(3);
  ExprHandle inner_bound(4);
  Tensor* A = Compute(
      "A",
      {{outer_bound, "x"}, {inner_bound, "y"}},
      [&](const VarHandle& x, const VarHandle& y) { return x + y; });
  LoopNest l({A});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(A->buf())[0];
  Stmt* unrolled = nullptr;
  LoopNest::unroll(
      static_cast<For*>(loops[0]->body()->stmts().front()), &unrolled);
  checkIR(loops[0], R"IR(
# CHECK: for (int x = 0; x < 3; x++) {
# CHECK: A[x, 0] = x;
# CHECK: A[x, 1] = x + 1;
# CHECK: A[x, 2] = x + 2;
# CHECK: A[x, 3] = x + 3;
# CHECK: })IR");
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, UnrollMultipleStatements) {
  KernelScope kernel_scope;
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
  Block::make({f});
  Stmt* unrolled = nullptr;
  LoopNest::unroll(f, &unrolled);
  checkIR(unrolled, R"IR(
# CHECK: A[0] = 0;
# CHECK: B[0] = A[0];
# CHECK: A[1] = 2;
# CHECK: B[1] = A[1];
# CHECK: A[2] = 4
# CHECK: B[2] = A[2];)IR");
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, UnrollNonLiteralConstantBounds) {
  KernelScope kernel_scope;

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

  std::vector<For*> loops = {outer_for, inner_for};
  Stmt* unrolled = nullptr;
  LoopNest::unroll(loops[0], &unrolled);
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, UnrollEmpty) {
  const std::string actual = constantUpperBoundLoopIR(0);
  const std::string& verification_pattern = R"IR(
# CHECK-NOT: A[
  )IR";

  torch::jit::testing::FileCheck().run(verification_pattern, actual);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, NoUnroll) {
  KernelScope kernel_scope;
  VarHandle upper_bound("N", kInt);
  Tensor* A = Compute(
      "A", {{upper_bound, "x"}}, [&](const VarHandle& x) { return x * 2; });
  LoopNest l({A});
  std::vector<For*> loops = l.getAllLoopNestsWritingToBuf(A->buf())[0];
  Stmt* unrolled = nullptr;
  ASSERT_THROWS_WITH(
      LoopNest::unroll(loops[0], &unrolled), "non-constant loop");
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, UnrollWithLet) {
  KernelScope kernel_scope;
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
  Block::make({f});
  Stmt* unrolled = nullptr;
  LoopNest::unroll(f, &unrolled);
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
  KernelScope kernel_scope;

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

  for_stmt->setStart(new IntImm(0));
  ASSERT_TRUE(LoopNest::isNormalized(for_stmt));

  VarHandle N("N", kInt);
  for_stmt->setStart(N.node());
  ASSERT_FALSE(LoopNest::isNormalized(for_stmt));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, NormalizeStartPositive) {
  KernelScope kernel_scope;

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, NormalizeStartNegative) {
  KernelScope kernel_scope;

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, NormalizeStartZero) {
  KernelScope kernel_scope;

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, NormalizeStartVariable) {
  KernelScope kernel_scope;

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
  Block::make({for_stmt});

  LoopNest::normalize(for_stmt);

  auto result = IRSimplifier::simplify(for_stmt);
  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int x = 0; x < 100 - y; x++) {
        # CHECK:   A[y + x] = B[y + x];
        # CHECK:   B[y + x] = 2 * (y + x);
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, NormalizeOnNestedOuterLoop) {
  KernelScope kernel_scope;

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, NormalizeOnNestedInnerLoop) {
  KernelScope kernel_scope;

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
        # CHECK:     A[x] = (((B[y + 10]) + 2 * y) + (A[x])) + 20;
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, NormalizeAndSplitWithTail) {
  KernelScope kernel_scope;

  // Create a dummy tensor to construct LoopNest.
  ExprHandle n(100);
  Placeholder a(BufHandle("a", {n}, kFloat));
  Tensor* b =
      Compute("b", {{n, "i"}}, [&](const VarHandle& i) { return a.load(i); });
  LoopNest l({b});

  // Input IR:
  //   for (int x = 5; x < 10; x++) {
  //     A[x] = x * 2;
  //   }
  const int kTotalSize = 5;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  VarHandle x("x", kInt);
  auto for_stmt = For::make(x, 5, 10, Store::make(a_buf, {x}, x * 2));
  Block::make({for_stmt});

  LoopNest::normalize(for_stmt);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* x_inner;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* x_tail;
  l.splitWithTail(for_stmt, 10, &x_inner, &x_tail);

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, FlattenSimpleLoopNest2D) {
  KernelScope kernel_scope;

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
  Block::make({outer_for});

  std::vector<For*> loops = {outer_for, inner_for};
  For* flattened = nullptr;
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, FlattenSimpleLoopNest3D) {
  KernelScope kernel_scope;

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
  Block::make({for3});

  std::vector<For*> loops = {for3, for2, for1};
  For* flattened = nullptr;
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, FlattenLoopNestAfterNormalize) {
  KernelScope kernel_scope;

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
  Block::make({outer_for});

  std::vector<For*> loops = {outer_for, inner_for};
  For* flattened = nullptr;
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, FlattenLoopNestWithNonLiteralConstantBounds) {
  KernelScope kernel_scope;

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

  std::vector<For*> loops = {outer_for, inner_for};
  For* flattened = nullptr;
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, FlattenImperfectLoopNest) {
  KernelScope kernel_scope;

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

  std::vector<For*> loops = {outer_for, inner_for};
  For* flattened = nullptr;
  ASSERT_FALSE(LoopNest::flatten(loops, &flattened));
  ASSERT_EQ(flattened, nullptr);
  auto hash_after = hasher.hash(par);
  ASSERT_EQ(hash_before, hash_after);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, FlattenReductionLoopNest) {
  KernelScope kernel_scope;

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

  std::vector<For*> loops = {outer_for, inner_for};
  For* flattened = nullptr;
  ASSERT_FALSE(LoopNest::flatten(loops, &flattened));
  ASSERT_EQ(flattened, nullptr);
  auto hash_after = hasher.hash(par);
  ASSERT_EQ(hash_before, hash_after);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, FlattenReductionLoopNestFromTensor) {
  KernelScope kernel_scope;
  const int M = 3;
  const int N = 7;
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  Placeholder b(BufHandle("b", {m, n}, kFloat));
  Tensor* c = Reduce("sum", {{M, "m"}}, Sum(), b, {{N, "n"}});
  LoopNest loop({c});
  HashProvider hasher;
  auto hash_before = hasher.hash(loop.root_stmt());

  auto loops = loop.getAllLoopNestsWritingToBuf(c->buf())[1];
  For* flattened = nullptr;
  ASSERT_FALSE(LoopNest::flatten(loops, &flattened));
  ASSERT_EQ(flattened, nullptr);
  auto hash_after = hasher.hash(loop.root_stmt());
  ASSERT_EQ(hash_before, hash_after);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, FlattenIncorrectLoopsAsInput) {
  KernelScope kernel_scope;

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

  std::vector<For*> loops = {outer_for1, inner_for2};
  For* flattened = nullptr;
  ASSERT_FALSE(LoopNest::flatten(loops, &flattened));
  ASSERT_EQ(flattened, nullptr);
  auto hash_after = hasher.hash(par);
  ASSERT_EQ(hash_before, hash_after);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, DetectInlineRankMismatch) {
  KernelScope kernel_scope;
  const int kTotalSize = 8;

  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Tensor* a = Compute("a", {{kTotalSize, "i"}}, [&](const VarHandle& i) {
    return a_buf.load(i);
  });
  Tensor* reshape = Compute(
      "reshape",
      {{kTotalSize / 2, "i"}, {2, "j"}},
      [&](const VarHandle& i, const VarHandle& j) { return a->load(i, j); });
  LoopNest l({reshape}, {a, reshape});
  ASSERT_THROWS_WITH(
      l.computeInline(l.getLoopBodyFor(a)),
      "Placeholder indexed access is inconsistent with its rank");
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, CacheReadsSimple) {
  KernelScope kernel_scope;

  Tensor* A = Compute(
      "A", {{64, "i"}, {64, "j"}}, [](const VarHandle& i, const VarHandle& j) {
        return i * j;
      });
  Tensor* B = Compute(
      "B", {{20, "i"}, {10, "j"}}, [&](const VarHandle& i, const VarHandle& j) {
        return A->load(i + 30, j + 3);
      });
  Tensor* C = Compute(
      "C", {{20, "i"}, {10, "j"}}, [&](const VarHandle& i, const VarHandle& j) {
        return A->load(i + 10, j + 20) + A->load(i + 30, j + 40);
      });

  LoopNest l({B, C}, {A, B, C});
  Stmt* j_loop = l.getAllLoopNestsWritingToBuf(B->buf())[0][1];
  l.cacheAccesses(A->buf(), "A_local", j_loop);

  l.prepareForCodegen();
  Stmt* result = IRSimplifier::simplify(l.root_stmt());

  // just this once: verify the whole thing.
  checkIR(result, R"IR(
#CHECK: Allocate(A); // dtype=int, dims=[64, 64]
#CHECK: for (int i
#CHECK:  for (int j
#CHECK:   A[
#CHECK:  }
#CHECK: }
#CHECK: for (int i_1
#CHECK:  Allocate(A_local); // dtype=int, dims=[1, 10]
#CHECK:  for (int j_1
#CHECK:   A_local[j_1] = A[
#CHECK:  }
#CHECK:  for (int j_2
#CHECK:   B[10 * i_1 + j_2] = A_local[j_2];
#CHECK:  }
#CHECK:  Free(A_local);
#CHECK: }
#CHECK: for (int i_2
#CHECK:  for (int j_3
#CHECK:   C[
#CHECK:  }
#CHECK: }
#CHECK: Free(A);
      )IR");

  std::vector<int> b_data(200, 0);
  std::vector<int> c_data(200, 0);
  SimpleIREvaluator cg(l.root_stmt(), {B, C});
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, CacheReadsOuter) {
  KernelScope kernel_scope;

  Tensor* A = Compute(
      "A", {{64, "i"}, {64, "j"}}, [](const VarHandle& i, const VarHandle& j) {
        return i * j;
      });
  Tensor* B = Compute(
      "B", {{20, "i"}, {10, "j"}}, [&](const VarHandle& i, const VarHandle& j) {
        return A->load(i + 30, j + 40) + A->load(i + 31, j + 41);
      });
  Tensor* C = Compute(
      "C", {{20, "i"}, {10, "j"}}, [&](const VarHandle& i, const VarHandle& j) {
        return A->load(i + 10, j + 20) + A->load(i + 30, j + 40);
      });

  LoopNest l({B, C}, {A, B, C});
  Stmt* i_loop = l.getAllLoopNestsWritingToBuf(B->buf())[0][0];
  l.cacheAccesses(A->buf(), "A_local", i_loop);

  l.prepareForCodegen();
  Stmt* result = IRSimplifier::simplify(l.root_stmt());

  checkIR(result, R"IR(
#CHECK: Allocate(A_local); // dtype=int, dims=[21, 11]
#CHECK: A_local[j_1 + 11 * i_1] =
#CHECK: B[10 * i_2 + j_2] = (A_local[(j_2 + 11 * i_2) + 12]) + (A_local[j_2 + 11 * i_2]);
      )IR");

  std::vector<int> b_data(200, 0);
  std::vector<int> c_data(200, 0);
  SimpleIREvaluator cg(l.root_stmt(), {B, C});
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, CacheReadsInternal) {
  KernelScope kernel_scope;

  Tensor* A = Compute(
      "A", {{64, "i"}, {64, "j"}}, [](const VarHandle& i, const VarHandle& j) {
        return i * j;
      });
  Tensor* B = Compute(
      "B", {{20, "i"}, {10, "j"}}, [&](const VarHandle& i, const VarHandle& j) {
        return A->load(i + 30, j + 40) + A->load(i + 31, j + 41);
      });
  Tensor* C = Compute(
      "C", {{20, "i"}, {10, "j"}}, [&](const VarHandle& i, const VarHandle& j) {
        return A->load(i + 10, j + 20) + A->load(i + 30, j + 40);
      });

  LoopNest l({B, C}, {A, B, C});
  Stmt* j_loop = l.getAllLoopNestsWritingToBuf(B->buf())[0][1];
  l.cacheAccesses(A->buf(), "A_local", j_loop);
  l.prepareForCodegen();
  Stmt* result = IRSimplifier::simplify(l.root_stmt());

  checkIR(result, R"IR(
#CHECK: Allocate(A_local); // dtype=int, dims=[2, 11]
#CHECK: A_local[j_1 + 11 * i_2] =
#CHECK: B[10 * i_1 + j_2] = (A_local[j_2 + 12]) + (A_local[j_2]);
      )IR");

  std::vector<int> b_data(200, 0);
  std::vector<int> c_data(200, 0);
  SimpleIREvaluator cg(l.root_stmt(), {B, C});
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, CacheReadsInner) {
  KernelScope kernel_scope;

  Tensor* A = Compute(
      "A", {{64, "i"}, {64, "j"}}, [](const VarHandle& i, const VarHandle& j) {
        return i * j;
      });
  // note im changing the offset of the first arg of the first call to A.
  Tensor* B = Compute(
      "B", {{20, "i"}, {10, "j"}}, [&](const VarHandle& i, const VarHandle& j) {
        return A->load(i + 34, j + 40) + A->load(i + 30, j + 41);
      });
  Tensor* C = Compute(
      "C", {{20, "i"}, {10, "j"}}, [&](const VarHandle& i, const VarHandle& j) {
        return A->load(i + 10, j + 20) + A->load(i + 30, j + 40);
      });

  LoopNest l({B, C}, {A, B, C});
  Stmt* body = l.getLoopBodyFor(B);
  l.cacheAccesses(A->buf(), "A_local", body);
  l.prepareForCodegen();
  Stmt* result = IRSimplifier::simplify(l.root_stmt());

  checkIR(result, R"IR(
#CHECK: Allocate(A_local); // dtype=int, dims=[5, 2]
#CHECK: A_local[2 * i_2 + j_2] =
#CHECK: B[10 * i_1 + j_1] = (A_local[1]) + (A_local[8]);
      )IR");

  std::vector<int> b_data(200, 0);
  std::vector<int> c_data(200, 0);
  SimpleIREvaluator cg(l.root_stmt(), {B, C});
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, CacheWritesSimple) {
  KernelScope kernel_scope;

  Tensor* A = Compute(
      "A", {{64, "i"}, {64, "j"}}, [](const VarHandle& i, const VarHandle& j) {
        return i * j;
      });
  Tensor* B = Compute(
      "B", {{20, "i"}, {10, "j"}}, [&](const VarHandle& i, const VarHandle& j) {
        return A->load(i + 30, j + 40) + A->load(i + 31, j + 41);
      });
  Tensor* C = Compute(
      "C", {{20, "i"}, {10, "j"}}, [&](const VarHandle& i, const VarHandle& j) {
        return A->load(i + 10, j + 20) + A->load(i + 30, j + 40);
      });

  LoopNest l({B, C}, {A, B, C});
  Stmt* a_loop = l.getAllLoopNestsWritingToBuf(A->buf())[0][1];
  l.cacheAccesses(A->buf(), "A_local", a_loop);

  l.prepareForCodegen();
  Stmt* result = IRSimplifier::simplify(l.root_stmt());

  checkIR(result, R"IR(
#CHECK: Allocate(A_local); // dtype=int, dims=[1, 64]
#CHECK: for (int j = 0; j < 64
#CHECK:   A_local[j] = i * j;
#CHECK: for (int j_1 = 0; j_1 < 64
#CHECK:   A[64 * i + j_1] = A_local[
#CHECK: Free(A_local);
#CHECK-NOT: A_local
      )IR");

  std::vector<int> b_data(200, 0);
  std::vector<int> c_data(200, 0);
  SimpleIREvaluator cg(l.root_stmt(), {B, C});
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, DeadStoreElimination) {
  KernelScope kernel_scope;
  VarHandle y("y", kInt);
  VarHandle x("x_tail", kInt);
  BufHandle f("f", {26, 5}, kInt);
  BufHandle g("g", {26, 5}, kInt);
  ExprHandle x_outer_end = 5;
  ExprHandle x_2 = x + x_outer_end * 4;
  For* stmt1 = For::make(
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
  Stmt* stmt = Block::make({stmt1});

  // Will eliminate if not used by an output.
  LoopNest loop(stmt, {f.node()});
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, DeadStoreEliminationWithIntermediates) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);
  BufHandle f("f", {26 * 5}, kInt);
  BufHandle g("g", {26 * 5}, kInt);
  BufHandle h("h", {26, 5}, kInt);
  ExprHandle x_outer_end = 5;
  ExprHandle x_2 = x + x_outer_end * 4;
  For* stmt1 = For::make(x, 0, 26 * 5, Store::make(f, {x}, x));
  For* stmt2 = For::make(z, 0, 26 * 5, Store::make(g, {z}, z + 1));
  For* stmt3 = For::make(
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
  Stmt* stmt = Block::make({stmt1, stmt2, stmt3});

  // Will eliminate the write to g, but not f since it used by the producer of
  // h.
  LoopNest loop(stmt, {h.node()});
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, CompoundTensorSimple) {
  KernelScope kernel_scope;

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
  Block* body = Block::make({outer_for1, outer_for2});

  Tensor* A = new Tensor(a_buf.node(), body);

  LoopNest l({A});
  l.prepareForCodegen();

  std::vector<int> a_data(50, 0);

  Stmt* s = IRSimplifier::simplify(l.root_stmt());
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, InlineConstantIndex) {
  KernelScope kernel_scope;
  const int N = 10;
  Placeholder x_buf("a", kFloat, {1, N, 1});
  Tensor* y = Compute(
      "f",
      {{1, "m"}, {N, "n"}, {1, "o"}},
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& o) {
        return x_buf.load(m, n, o);
      });
  Tensor* z = Compute(
      "f",
      {{1, "m"}, {N, "n"}, {1, "o"}},
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& o) {
        return y->load(m, n, o);
      });

  LoopNest l({z}, {y, z});
  l.simplify();
  ASSERT_TRUE(l.computeInline(y->buf()));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, CompoundTensorUsed) {
  KernelScope kernel_scope;

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
  Block* body = Block::make({outer_for1, outer_for2});

  Tensor* A = new Tensor(a_buf.node(), body);
  Tensor* B = Compute(
      "B", {{10, "i"}, {3, "j"}}, [&](const VarHandle& i, const VarHandle& j) {
        return A->load(i, j + 1) + A->load(i, j + 2);
      });

  LoopNest l({B}, {A, B});
  ASSERT_FALSE(l.computeInline(A->buf()));
  l.prepareForCodegen();

  std::vector<int> a_data(50, 0);
  std::vector<int> b_data(50, 0);

  Stmt* s = IRSimplifier::simplify(l.root_stmt());
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, InlineFromLoad) {
  KernelScope kernel_scope;

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
  KernelScope kernel_scope;

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
  KernelScope kernel_scope;

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
  KernelScope kernel_scope;

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
  KernelScope kernel_scope;

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
# CHECK-NEXT: B[i] = IfThenElse(i<30 ? 1 : 0, C[i], D[i])
# CHECK: for (int i = 0; i < 45
# CHECK-NEXT: A[i + 5] = C[i]
# CHECK-NEXT: B[i + 5] = IfThenElse(i + 5<30 ? 1 : 0, C[i + 5], D[i + 5])
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(LoopNest, OptimizeConditionalsOuterLoopVar) {
  KernelScope kernel_scope;

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
  KernelScope kernel_scope;

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
  KernelScope kernel_scope;

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
  KernelScope kernel_scope;

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
  KernelScope kernel_scope;

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
  KernelScope kernel_scope;

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
  KernelScope kernel_scope;

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
  KernelScope kernel_scope;

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

static std::pair<std::unique_ptr<Placeholder>, Tensor*> colReduce(
    int M,
    int N) {
  auto a =
      std::make_unique<Placeholder>("a", kFloat, std::vector<ExprHandle>{M, N});
  Tensor* t = Reduce(
      "b",
      {{N, "n"}},
      Sum(),
      [&](const VarHandle& n, const VarHandle& m) { return a->load(m, n); },
      {{M, "m"}});
  return {std::move(a), t};
}

static Stmt* splitTailReorder(Tensor* b) {
  constexpr int kVectorWidth = 8;
  LoopNest nest({b});
  auto loops = nest.getAllLoopNestsWritingToBuf(b->buf())[0];
  nest.splitWithTail(loops[0], kVectorWidth);
  // Now the loopnests will look like:
  //
  // for (int n_outer = 0; ...
  //   for (int n_inner = 0; ...
  //     b[n_outer * 8 + n_inner] = float(0);
  //     for (int m = 0; ...
  //       b[n_outer * 8 + n_inner] = ReduceOp(...);
  //
  // for (int n_tail = 0; ...
  //   b[n_tail + ((100 - 0) / 8) * 8] = float(0);
  //   for (int m = 0; ...
  //     b[n_tail + ((100 - 0) / 8) * 8] = ReduceOp(...);
  //
  // Since there are 4 writes to b, we will get 4 loopnests from the
  // call to `getAllLoopNestsWritingToBuf` below.
  //
  // Write #2: "b[n_outer * 8 + n_inner] = ReduceOp(...)"
  // Loopnest #2: {n_outer, n_inner, m};
  // We will have to reorder n_inner and m.
  auto loopnests = nest.getAllLoopNestsWritingToBuf(b->buf());
  nest.reorderAxis(loopnests[1][1], loopnests[1][2]);
  nest.prepareForCodegen();
  return nest.root_stmt();
}

static Stmt* splitMaskReorder(Tensor* b) {
  constexpr int kVectorWidth = 8;
  LoopNest nest({b});
  auto loops = nest.getAllLoopNestsWritingToBuf(b->buf())[1];
  nest.splitWithMask(loops[0], kVectorWidth);
  loops = nest.getAllLoopNestsWritingToBuf(b->buf())[1];
  nest.reorderAxis(loops[1], loops[2]);
  nest.prepareForCodegen();
  return nest.root_stmt();
}

static void checkColReduce(Stmt* s, Placeholder& p, Tensor* t) {
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ColReduceSplitTailEvenReorder) {
  KernelScope kernel_scope;
  constexpr int M = 76, N = 128;
  auto p = colReduce(M, N);
  Stmt* s = splitTailReorder(p.second);

  std::ostringstream oss;
  oss << *s;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int n_outer
# CHECK-NEXT: for (int n_inner
# CHECK-NEXT: b[
# CHECK: for (int m
# CHECK-NEXT: for (int n_inner
# CHECK-NEXT: b[
# CHECK-NOT: for (
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  checkColReduce(s, *p.first, p.second);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ColReduceSplitTailUnevenReorder) {
  KernelScope kernel_scope;
  constexpr int M = 76, N = 100;
  auto p = colReduce(M, N);
  Stmt* s = splitTailReorder(p.second);

  std::ostringstream oss;
  oss << *s;
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int n_outer
# CHECK-NEXT: for (int n_inner
# CHECK-NEXT: b[
# CHECK: for (int m
# CHECK-NEXT: for (int n_inner
# CHECK-NEXT: b[
# CHECK: for (int n_tail
# CHECK-NEXT: b[
# CHECK-NEXT: for (int m
# CHECK-NEXT: b[
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  checkColReduce(s, *p.first, p.second);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ColReduceSplitMaskEvenReorder) {
  KernelScope kernel_scope;
  constexpr int M = 76, N = 128;
  auto p = colReduce(M, N);
  Stmt* s = splitMaskReorder(p.second);
  checkColReduce(s, *p.first, p.second);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ColReduceSplitMaskUnevenReorder) {
  KernelScope kernel_scope;
  constexpr int M = 76, N = 100;
  auto p = colReduce(M, N);
  Stmt* s = splitMaskReorder(p.second);
  checkColReduce(s, *p.first, p.second);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, ReorderAxisWithMultipleConds) {
  KernelScope kernel_scope;

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
  Stmt* par = Block::make({forI});
  LoopNest l(par, {a_buf.node()});
  l.reorderAxis(forI, forJ);
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, VectorizeUse) {
  KernelScope kernel_scope;
  constexpr int N = 8;
  Placeholder a("a", kFloat, {N});
  Tensor* b = Compute(
      "b", {{N, "n"}}, [&](const VarHandle& n) { return a.load(n) + 1.0f; });
  Tensor* c = Compute(
      "c", {{N, "n"}}, [&](const VarHandle& n) { return b->load(n) + 2.0f; });
  LoopNest nest({c}, {b, c});
  auto loops = nest.getAllLoopNestsWritingToBuf(b->buf())[0];
  nest.vectorize(loops[0]);
  loops = nest.getAllLoopNestsWritingToBuf(c->buf())[0];
  nest.vectorize(loops[0]);
  nest.prepareForCodegen();
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  Stmt* s = nest.root_stmt();
  std::ostringstream oss;
  oss << *nest.root_stmt();
  torch::jit::testing::FileCheck().run(
      R"IR(
# CHECK: c[Ramp
)IR",
      oss.str());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
const char* int64Loop = R"IR(
{
  for (int64_t n = 0; n < 12; n++) {
    b[n] = (a[n]) + 1;
  }
}
)IR";

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, DISABLED_Int64Direct) {
  KernelScope kernel_scope;

  constexpr int64_t N = 12;
  Placeholder a("a", kLong, {N});
  Placeholder b("b", kLong, {N});
  VarHandle n("n", kLong);
  Stmt* s = For::make(n, 0, N, b.store({n}, a.load({n}) + LongImm::make(1l)));
  s = IRSimplifier::simplify(s);
  std::ostringstream oss;
  oss << *s;
  ASSERT_EQ(oss.str(), int64Loop);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, DISABLED_Int64Compute) {
  KernelScope kernel_scope;

  constexpr int64_t N = 12;
  Placeholder a("a", kLong, {N});
  Tensor* b = Compute("b", {{N, "n"}}, [&](const VarHandle& n) {
    return a.load(n) + LongImm::make(1l);
  });
  LoopNest nest({b});
  nest.prepareForCodegen();
  nest.simplify();
  std::ostringstream oss;
  oss << *nest.root_stmt();
  ASSERT_EQ(oss.str(), int64Loop);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, DistributeLoopWithAllStmtsAsPivots) {
  KernelScope kernel_scope;

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, DistributeLoopWithOneStmtAsPivot) {
  KernelScope kernel_scope;

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, DistributeLoopWithoutAnyPivot) {
  KernelScope kernel_scope;

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, DistributeLoopOverInnerLoops) {
  KernelScope kernel_scope;

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsSimple) {
  KernelScope kernel_scope;

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
  For* fused_loop;
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsMultiple) {
  KernelScope kernel_scope;

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
  For* fused_loop;
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsNested) {
  KernelScope kernel_scope;

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
  For* fused_loop;
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsNested2D) {
  KernelScope kernel_scope;

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
  For* fused_loop;
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsNested2DInner) {
  KernelScope kernel_scope;

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
  For* fused_loop;
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsDifferentStopBounds) {
  KernelScope kernel_scope;

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
  For* fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsDifferentStartBounds) {
  KernelScope kernel_scope;

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
  For* fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsNotContiguous) {
  KernelScope kernel_scope;

  // Input IR:
  //   for (int j = 0; j < 100; j++) {
  //     A[j] = 10 * j;
  //   }
  //   B[0] = 0;
  //   for (int k = 50; k < 100; k++) {
  //     B[k] = 20 * k;
  //   }
  BufHandle a_buf("A", {100}, kInt);
  BufHandle b_buf("B", {100}, kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  auto initB = Store::make(b_buf, {0}, 0);
  auto forK = For::make(k, 50, 100, Store::make(b_buf, {j}, Mul::make(20, k)));
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  auto par = Block::make({forJ, initB, forK});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsWithDifferentParents) {
  KernelScope kernel_scope;

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
  For* fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsWithVariableBounds) {
  KernelScope kernel_scope;

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
  For* fused_loop;
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsWithExprBounds) {
  KernelScope kernel_scope;

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
  For* fused_loop;
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsWithDifferentExprBounds) {
  KernelScope kernel_scope;

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
  For* fused_loop;
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsWithNonOverlappingBufferAccesses) {
  KernelScope kernel_scope;

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
  For* fused_loop;
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsWithNonOverlapping2DBufferAccesses) {
  KernelScope kernel_scope;

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
  For* fused_loop;
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsWithReductions) {
  KernelScope kernel_scope;

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
  For* fused_loop;
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsThatViolateDependencies1) {
  KernelScope kernel_scope;

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
  For* fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsThatViolateDependencies2) {
  KernelScope kernel_scope;

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
  For* fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsThatViolateDependencies3) {
  KernelScope kernel_scope;

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
  For* fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forM, forN}, &fused_loop));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsThatViolateDependencies4) {
  KernelScope kernel_scope;

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
  For* fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forI, forM}, &fused_loop));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsThatViolateDependencies5) {
  KernelScope kernel_scope;

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
  For* fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forN}, &fused_loop));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsThatViolateDependencies6) {
  KernelScope kernel_scope;

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
  For* fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forJ, forK}, &fused_loop));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, fuseLoopsThatViolateDependencies7) {
  KernelScope kernel_scope;

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
  For* fused_loop;
  ASSERT_FALSE(LoopNest::fuseLoops({forK, forJ}, &fused_loop));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, areLoopsPerfectlyNested) {
  KernelScope kernel_scope;

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

  // Adding a statment to forK body should be OK.
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, reorderNestedLoops2D) {
  KernelScope kernel_scope;

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, reorderNestedLoops3D) {
  KernelScope kernel_scope;

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

  auto reordered = LoopNest::reorder({forI, forJ, forK}, {2, 1, 0});

  ASSERT_EQ(reordered[0], forK);
  ASSERT_EQ(reordered[1], forJ);
  ASSERT_EQ(reordered[2], forI);
  ASSERT_TRUE(LoopNest::areLoopsPerfectlyNested({forK, forJ, forI}));
  ASSERT_EQ(forK->get_parent(), par);
  ASSERT_EQ(store->get_parent(), forI->body());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, reorderNestedLoops4D) {
  KernelScope kernel_scope;

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

  auto reordered = LoopNest::reorder({forI, forJ, forK, forL}, {2, 3, 0, 1});

  ASSERT_EQ(reordered[0], forK);
  ASSERT_EQ(reordered[1], forL);
  ASSERT_EQ(reordered[2], forI);
  ASSERT_EQ(reordered[3], forJ);
  ASSERT_TRUE(LoopNest::areLoopsPerfectlyNested({forK, forL, forI, forJ}));
  ASSERT_EQ(forK->get_parent(), par);
  ASSERT_EQ(store->get_parent(), forJ->body());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, reorderTrivialPermutation) {
  KernelScope kernel_scope;

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, reorderInvalidPermutations) {
  KernelScope kernel_scope;

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, reorderInvalidLoopNest) {
  KernelScope kernel_scope;

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, compressBufferSimple) {
  KernelScope kernel_scope;

  // Input IR:
  // for (int i = 0; i < 100; ++i) {
  //   for (int j = 0; j < 200; ++j) {
  //     A[i,j] = sin(i*j)
  //   }
  //   for (int j = 0; j < 199; ++j) {
  //     B[i,j] = A[i,j] + A[i, j+1]
  //   }
  // }
  Buf* A = new Buf("A", {new IntImm(100), new IntImm(200)}, kInt);
  Buf* B = new Buf("B", {new IntImm(100), new IntImm(200)}, kInt);
  BufHandle a_buf(A);
  BufHandle b_buf(B);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto forJ1 = For::make(j, 0, 200, Store::make(a_buf, {i, j}, sin(i * j)));
  auto forJ2 = For::make(
      j,
      0,
      199,
      Store::make(
          b_buf,
          {i, j},
          Add::make(Load::make(a_buf, {i, j}), Load::make(a_buf, {i, j + 1}))));
  auto forI = For::make(i, 0, 100, Block::make({forJ1, forJ2}));
  auto par = Block::make({forI});
  LoopNest::compressBuffer(A, par);

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

  ASSERT_EQ(A->ndim(), 2);
  ASSERT_EQ(dynamic_cast<const IntImm*>(A->dim(0))->value(), 1);
  ASSERT_EQ(dynamic_cast<const IntImm*>(A->dim(1))->value(), 200);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, compressBufferMultipleDims) {
  KernelScope kernel_scope;

  // Input IR:
  // for (int i = 0; i < 100; ++i) {
  //   for (int j = 0; j < 200; ++j) {
  //     A[i,j] = sin(i*j)
  //     B[i,j] = A[i,j] + A[i,j]
  //   }
  // }
  Buf* A = new Buf("A", {new IntImm(100), new IntImm(200)}, kInt);
  Buf* B = new Buf("B", {new IntImm(100), new IntImm(200)}, kInt);
  BufHandle a_buf(A);
  BufHandle b_buf(B);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto store1 = Store::make(a_buf, {i, j}, sin(i * j));
  auto store2 = Store::make(
      b_buf,
      {i, j},
      Add::make(Load::make(a_buf, {i, j}), Load::make(a_buf, {i, j})));
  auto forJ = For::make(j, 0, 200, Block::make({store1, store2}));
  auto forI = For::make(i, 0, 100, forJ);
  auto par = Block::make({forI});
  LoopNest::compressBuffer(A, par);

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

  ASSERT_EQ(A->ndim(), 2);
  ASSERT_EQ(dynamic_cast<const IntImm*>(A->dim(0))->value(), 1);
  ASSERT_EQ(dynamic_cast<const IntImm*>(A->dim(1))->value(), 1);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, compressBufferMultipleDims2) {
  KernelScope kernel_scope;

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
  Buf* A =
      new Buf("A", {new IntImm(100), new IntImm(200), new IntImm(300)}, kInt);
  Buf* B =
      new Buf("B", {new IntImm(100), new IntImm(200), new IntImm(300)}, kInt);
  BufHandle a_buf(A);
  BufHandle b_buf(B);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto store1 = Store::make(a_buf, {i, j, k}, sin(i * j * k));
  auto forK1 = For::make(k, 0, 300, store1);
  auto store2 = Store::make(
      b_buf,
      {i, j, k},
      Add::make(
          Load::make(a_buf, {i, j, k}), Load::make(a_buf, {i, j, k + 1})));
  auto forK2 = For::make(k, 0, 299, store2);
  auto forJ = For::make(j, 0, 200, Block::make({forK1, forK2}));
  auto forI = For::make(i, 0, 100, forJ);
  auto par = Block::make({forI});
  LoopNest::compressBuffer(A, par);

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

  ASSERT_EQ(A->ndim(), 3);
  ASSERT_EQ(dynamic_cast<const IntImm*>(A->dim(0))->value(), 1);
  ASSERT_EQ(dynamic_cast<const IntImm*>(A->dim(1))->value(), 1);
  ASSERT_EQ(dynamic_cast<const IntImm*>(A->dim(2))->value(), 300);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, compressBufferDifferentOrderIndices) {
  KernelScope kernel_scope;

  // Input IR:
  // for (int i = 0; i < 100; ++i) {
  //   for (int j = 0; j < 200; ++j) {
  //     A[j, i] = sin(i*j)
  //   }
  //   for (int j = 0; j < 99; ++j) {
  //     B[i, j] = A[j, i] + A[j+1, 0]
  //   }
  // }
  Buf* A = new Buf("A", {new IntImm(100), new IntImm(200)}, kInt);
  Buf* B = new Buf("B", {new IntImm(100), new IntImm(200)}, kInt);
  BufHandle a_buf(A);
  BufHandle b_buf(B);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto forJ1 = For::make(j, 0, 200, Store::make(a_buf, {j, i}, sin(i * j)));
  auto forJ2 = For::make(
      j,
      0,
      99,
      Store::make(
          b_buf,
          {i, j},
          Add::make(Load::make(a_buf, {j, i}), Load::make(a_buf, {j + 1, i}))));
  auto forI = For::make(i, 0, 100, Block::make({forJ1, forJ2}));
  auto par = Block::make({forI});
  LoopNest::compressBuffer(A, par);

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

  ASSERT_EQ(A->ndim(), 2);
  ASSERT_EQ(dynamic_cast<const IntImm*>(A->dim(0))->value(), 100);
  ASSERT_EQ(dynamic_cast<const IntImm*>(A->dim(1))->value(), 1);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, compressBufferVariableBounds) {
  KernelScope kernel_scope;

  // Input IR:
  // for (int i = 0; i < M; ++i) {
  //   for (int j = 0; j < N; ++j) {
  //     A[i,j] = sin(i*j)
  //   }
  //   for (int j = 0; j < N-1; ++j) {
  //     B[i,j] = A[i,j] + A[i, j+1]
  //   }
  // }
  Buf* A = new Buf("A", {new IntImm(100), new IntImm(200)}, kInt);
  Buf* B = new Buf("B", {new IntImm(100), new IntImm(200)}, kInt);
  BufHandle a_buf(A);
  BufHandle b_buf(B);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle M("M", kInt);
  VarHandle N("N", kInt);
  auto forJ1 = For::make(j, 0, N, Store::make(a_buf, {i, j}, sin(i * j)));
  auto forJ2 = For::make(
      j,
      0,
      N - 1,
      Store::make(
          b_buf,
          {i, j},
          Add::make(Load::make(a_buf, {i, j}), Load::make(a_buf, {i, j + 1}))));
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  auto forI = For::make(i, 0, M, Block::make({forJ1, forJ2}));
  auto par = Block::make({forI});
  LoopNest::compressBuffer(A, par);

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

  ASSERT_EQ(A->ndim(), 2);
  ASSERT_EQ(dynamic_cast<const IntImm*>(A->dim(0))->value(), 1);
  ASSERT_EQ(dynamic_cast<const IntImm*>(A->dim(1))->value(), 200);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, compressBufferNoCommonParentLoops) {
  KernelScope kernel_scope;

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
  Buf* A = new Buf("A", {new IntImm(100), new IntImm(200)}, kInt);
  Buf* B = new Buf("B", {new IntImm(100), new IntImm(200)}, kInt);
  BufHandle a_buf(A);
  BufHandle b_buf(B);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto forJ1 = For::make(j, 0, 200, Store::make(a_buf, {i, j}, sin(i * j)));
  auto forJ2 = For::make(
      j,
      0,
      199,
      Store::make(
          b_buf,
          {i, j},
          Add::make(Load::make(a_buf, {i, j}), Load::make(a_buf, {i, j + 1}))));
  auto forI1 = For::make(i, 0, 100, forJ1);
  auto forI2 = For::make(i, 0, 100, forJ2);
  auto par = Block::make({forI1, forI2});
  LoopNest::compressBuffer(A, par);

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

  ASSERT_EQ(A->ndim(), 2);
  ASSERT_EQ(dynamic_cast<const IntImm*>(A->dim(0))->value(), 100);
  ASSERT_EQ(dynamic_cast<const IntImm*>(A->dim(1))->value(), 200);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LoopNest, compressBufferIndicesMixed) {
  KernelScope kernel_scope;

  // Input IR:
  // for (int i = 0; i < 100; ++i) {
  //   for (int j = 0; j < 200; ++j) {
  //     A[i + j, j] = sin(i*j)
  //   }
  //   for (int j = 0; j < 199; ++j) {
  //     B[i,j] = A[i + j, j] + A[i + j, j+1]
  //   }
  // }
  Buf* A = new Buf("A", {new IntImm(300), new IntImm(200)}, kInt);
  Buf* B = new Buf("B", {new IntImm(100), new IntImm(200)}, kInt);
  BufHandle a_buf(A);
  BufHandle b_buf(B);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto forJ1 = For::make(j, 0, 200, Store::make(a_buf, {i + j, j}, sin(i * j)));
  auto forJ2 = For::make(
      j,
      0,
      199,
      Store::make(
          b_buf,
          {i, j},
          Add::make(
              Load::make(a_buf, {i + j, j}),
              Load::make(a_buf, {i + j, j + 1}))));
  auto forI = For::make(i, 0, 100, Block::make({forJ1, forJ2}));
  auto par = Block::make({forI});
  LoopNest::compressBuffer(A, par);

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

  ASSERT_EQ(A->ndim(), 2);
  ASSERT_EQ(dynamic_cast<const IntImm*>(A->dim(0))->value(), 300);
  ASSERT_EQ(dynamic_cast<const IntImm*>(A->dim(1))->value(), 200);
}

} // namespace jit
} // namespace torch
