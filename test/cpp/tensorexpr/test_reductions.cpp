#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <test/cpp/tensorexpr/test_base.h>

#include <test/cpp/tensorexpr/padded_buffer.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
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

TEST(Reductions, ReduceSum0D_1) {
  KernelScope kernel_scope;
  const int M = 10;

  Placeholder b(BufHandle("b", {M}, kFloat));
  std::vector<float> in(M);
  for (int j = 0; j < M; ++j) {
    in[j] = j;
  }

  std::vector<float> out(M, -1.f);

  Tensor c = Reduce("sum", {{M, "m"}}, Sum(), b, {});
  LoopNest loop({c});
  loop.prepareForCodegen();
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {b, c});

  cg.call({in, out});
  for (int i = 0; i < M; ++i) {
    ASSERT_EQ(out[i], in[i]);
  }
}

TEST(Reductions, ReduceSum0D_2) {
  KernelScope kernel_scope;
  const int M = 10;

  Placeholder b(BufHandle("b", {}, kFloat));
  std::vector<float> in(1);
  in[0] = 77.7;

  std::vector<float> out(1, -1.f);

  Tensor c = Reduce("sum", {}, Sum(), b, {});
  LoopNest loop({c});
  loop.prepareForCodegen();
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {b, c});

  cg.call({in, out});
  ASSERT_EQ(out[0], in[0]);
}

// Sum an array to a single value.
TEST(Reductions, ReduceSum1D) {
  KernelScope kernel_scope;

  Placeholder b(BufHandle("b", {10}, kFloat));
  std::vector<float> in(10);
  for (int j = 0; j < 10; ++j) {
    in[j] = j;
  }

  std::vector<float> out(1, -1.f);

  Tensor c = Reduce("sum", {}, Sum(), b, {{10, "m"}});
  LoopNest loop({c});
  loop.prepareForCodegen();
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {b, c});

  cg.call({in, out});
  ASSERT_EQ(out[0], 45);
}
// Sum a 2D tensor to a 1D tensor with dynamic shapes.
TEST(Reductions, ReduceSum2D) {
  KernelScope kernel_scope;

  const int M = 3;
  const int N = 7;

  VarHandle m("m", kInt);
  VarHandle n("n", kInt);

  Placeholder b(BufHandle("b", {m, n}, kFloat));
  std::vector<float> in(M * N);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      in[i * N + j] = j;
    }
  }

  std::vector<float> out(M, -1.f);

  Tensor c = Reduce("sum", {{M, "m"}}, Sum(), b, {{N, "n"}});
  LoopNest loop({c});
  loop.prepareForCodegen();
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {b, c, n, m});

  cg.call({in, out, 5, 7});

  float expected = 0;
  for (int i = 0; i < N; ++i) {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    expected += i;
  }

  for (int i = 0; i < M; ++i) {
    ASSERT_EQ(out[i], expected);
  }
}

// Sum a 3D tensor to both a 2D and 1D tensor, then reduce the 2D tensor flat to
// check our work.
TEST(Reductions, ReduceSum3D) {
  KernelScope kernel_scope;

  const int M = 10;
  VarHandle m("m", kInt);

  Placeholder b(BufHandle("b", {2, 3, m}, kFloat));

  Tensor c = Reduce("sum", {{2, "l"}, {3, "n"}}, Sum(), b, {{m, "m"}});
  LoopNest loop({c});
  loop.prepareForCodegen();
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {b, c, m});

  std::vector<float> bData(2 * 3 * M, 0);
  std::vector<float> cData(2 * 3, 6.0f);
  std::vector<float> dData(2, 1.0f);
  std::vector<float> eData(2, 1.0f);

  for (int i = 0; i < 2 * 3; ++i) {
    for (int j = 0; j < M; ++j) {
      bData[i * M + j] = j;
    }
  }

  cg.call({bData, cData, M});
  float expected = 0;
  for (int i = 0; i < M; ++i) {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    expected += i;
  }

  for (int i = 0; i < 2 * 3; ++i) {
    ASSERT_EQ(cData[i], expected);
  }

  Tensor d = Reduce("sum2", {{2, "l"}}, Sum(), b, {{3, "n"}, {m, "m"}});
  LoopNest loop2({d});
  loop2.prepareForCodegen();
  StmtPtr s2 = loop2.root_stmt();
  s2 = IRSimplifier::simplify(s2);

  SimpleIREvaluator cg2(s2, {b, d, m});
  cg2.call({bData, dData, M});

  // We're combining an additional dimension of 3, so the sum is 3x.
  expected = expected * 3;

  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(dData[i], expected);
  }

  // This is the same as just reducing the original result across that axis.
  Placeholder c_buf(BufHandle(c.buf()));
  Tensor e = Reduce("sum3", {{2, "l"}}, Sum(), c_buf, {{3, "m"}});
  LoopNest loop3({e});
  loop3.prepareForCodegen();
  StmtPtr s3 = loop3.root_stmt();
  s3 = IRSimplifier::simplify(s3);

  SimpleIREvaluator cg3(s3, {c, e});
  cg3.call({cData, eData});

  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(eData[i], expected);
  }
}

// Sum a large (10 D) Tensor 5 dimensions in.
TEST(Reductions, ReduceSum10D) {
  KernelScope kernel_scope;

  Placeholder in_(BufHandle("in_", {2, 3, 2, 3, 2, 3, 2, 3, 2, 3}, kFloat));
  const int InputSize = 2 * 3 * 2 * 3 * 2 * 3 * 2 * 3 * 2 * 3;
  Placeholder out_(BufHandle("out_", {2, 3, 2, 3, 2}, kFloat));
  const int OutputSize = 2 * 3 * 2 * 3 * 2;

  std::vector<float> in(InputSize, 1.f);
  std::vector<float> out(OutputSize, -1.f);

  Tensor c = Reduce(
      "sum",
      {{2, "a"}, {3, "b"}, {2, "c"}, {3, "d"}, {2, "e"}},
      Sum(),
      in_,
      {{3, "f"}, {2, "g"}, {3, "h"}, {2, "i"}, {3, "j"}});
  LoopNest loop({c});
  loop.prepareForCodegen();
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {in_, c});

  cg.call({in, out});

  // NOLINTNEXTLINE(bugprone-integer-division)
  float expected = InputSize / OutputSize;
  for (int i = 0; i < OutputSize; ++i) {
    ASSERT_EQ(out[i], expected);
  }
}

// Reduce via Mul rather than Add using a custom Reducer.
TEST(Reductions, ReduceProduct) {
  KernelScope kernel_scope;

  const int M = 4;
  const int N = 4;

  Placeholder b(BufHandle("b", {M, N}, kFloat));
  std::vector<float> in(M * N);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      in[i * N + j] = 2 + j;
    }
  }

  std::vector<float> out(M, -1.f);

  Reducer product(
      ExprHandle(1.f), [](ExprHandle a, ExprHandle b) { return a * b; });

  Tensor c = Reduce("product", {{M, "m"}}, product, b, {{N, "n"}});
  LoopNest loop({c});
  loop.prepareForCodegen();
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {b, c});

  cg.call({in, out});

  float expected = 1;
  for (int i = 0; i < N; ++i) {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    expected *= 2 + i;
  }

  for (int i = 0; i < M; ++i) {
    ASSERT_EQ(out[i], expected);
  }
}

// Maximum reductions.
TEST(Reductions, ReduceMax) {
  KernelScope kernel_scope;

  Placeholder in_(BufHandle("b", {10}, kFloat));

  std::vector<float> in(10);
  std::vector<float> out(1, -1.f);
  for (int j = 0; j < 10; ++j) {
    in[j] = j;
  }

  Tensor dm1 = Reduce("max", {}, Maximum(kFloat), in_, {{10, "m"}});

  LoopNest loop({dm1});
  loop.prepareForCodegen();
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);
  SimpleIREvaluator cg(s, {in_, dm1});

  cg.call({in, out});

  ASSERT_EQ(out[0], 9);

  Placeholder in2_(BufHandle("b", {2, 5}, kFloat));
  std::vector<float> out2(2, -1.f);

  Tensor m2d = Reduce("max", {{2, "n"}}, Maximum(kFloat), in2_, {{5, "m"}});

  LoopNest loop2({m2d});
  loop2.prepareForCodegen();
  s = loop2.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg2(s, {in2_, m2d});
  cg2.call({in, out2});

  ASSERT_EQ(out2[0], 4);
  ASSERT_EQ(out2[1], 9);
}

// Minimum reduction, with custom initialization.
TEST(Reductions, ReduceMinCustomInitializer) {
  KernelScope kernel_scope;

  VarHandle minInit("minInit", kFloat);
  Placeholder in_(BufHandle("b", {10}, kFloat));

  std::vector<float> in(10);
  std::vector<float> out(1, -1.f);
  for (int j = 0; j < 10; ++j) {
    in[j] = 10 + j;
  }

  Tensor min = Reduce(
      "min",
      {},
      Minimum(ExprHandle(minInit)),
      [&](ParameterList& v) { return in_.load(v); },
      {{10, "m"}});

  LoopNest loop({min});
  loop.prepareForCodegen();
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {in_, min, minInit});

  // Works normally (note that out data starts lower than the correct
  // minimum).
  cg.call({in, out, std::numeric_limits<float>::max()});
  ASSERT_EQ(out[0], 10);

  // With an initalizer lower than the min, that's the min.
  cg.call({in, out, 5.f});
  ASSERT_EQ(out[0], 5);
}

// Example implementation of Any/All.
// TODO: this is very awkward without logical And/Or operators.
TEST(Reductions, ReduceAnyAll) {
  KernelScope kernel_scope;

  VarHandle searchValue("searchValue", kInt);
  Placeholder b(BufHandle("b", {4, 10}, kInt));

  Reducer anyEqSV(ExprHandle(0), [](ExprHandle a, ExprHandle b) {
    return CompareSelect::make(a, 1, 1, b, kEQ);
  });

  Tensor any = Reduce(
      "anyEqual",
      {{4, "i"}},
      anyEqSV,
      [&](const auto& i, const auto& j) {
        return CompareSelect::make(b.load(i, j), searchValue, kEQ);
      },
      {{10, "j"}});

  LoopNest loop({any});
  loop.prepareForCodegen();
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {b, any, searchValue});

  std::vector<int> in(40, 0);
  std::vector<int> out(4, 0);

  // input has 0-39 in 4 rows.
  for (int i = 0; i < 40; ++i) {
    in[i] = i;
  }
  cg.call({in, out, 1});

  // only the first row has 1
  ASSERT_EQ(out[0], 1);
  ASSERT_EQ(out[1], 0);
  ASSERT_EQ(out[2], 0);
  ASSERT_EQ(out[3], 0);

  cg.call({in, out, 15});

  // 15 in the 3rd row
  ASSERT_EQ(out[0], 0);
  ASSERT_EQ(out[1], 1);
  ASSERT_EQ(out[2], 0);
  ASSERT_EQ(out[3], 0);

  Reducer allGTSV(ExprHandle(1), [](ExprHandle a, ExprHandle b) {
    return CompareSelect::make(a, 0, 0, b, kEQ);
  });

  Tensor allGreaterThan = Reduce(
      "allGreaterThan",
      {{4, "i"}},
      allGTSV,
      [&](const auto& i, const auto& j) {
        return CompareSelect::make(b.load(i, j), searchValue, kGT);
      },
      {{10, "j"}});

  LoopNest loop2({allGreaterThan});
  loop2.prepareForCodegen();
  s = loop2.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg2(s, {b, allGreaterThan, searchValue});

  cg2.call({in, out, 11});

  // 11 is in row 2.
  ASSERT_EQ(out[0], 0);
  ASSERT_EQ(out[1], 0);
  ASSERT_EQ(out[2], 1);
  ASSERT_EQ(out[3], 1);

  cg2.call({in, out, -3});

  // All are positive.
  ASSERT_EQ(out[0], 1);
  ASSERT_EQ(out[1], 1);
  ASSERT_EQ(out[2], 1);
  ASSERT_EQ(out[3], 1);
}

TEST(Reductions, ReduceMatmul2D) {
  KernelScope kernel_scope;

  Placeholder tA(BufHandle("tA", {3, 2}, kFloat));
  Placeholder tB(BufHandle("tB", {2, 3}, kFloat));

  std::vector<float> tA_(6);
  std::vector<float> tB_(6);

  std::vector<float> out(9, -1.f);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 2; ++j) {
      tA_[i * 2 + j] = i * 2 + j;
      tB_[j * 3 + i] = i * 2 + j;
    }
  }

  Tensor mm = Reduce(
      "mm",
      {{3, "m"}, {3, "n"}},
      Sum(),
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
        return tA.load(m, k) * tB.load(k, n);
      },
      {{2, "k"}});

  LoopNest loop({mm});
  loop.prepareForCodegen();
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {tA, tB, mm});
  cg.call({tA_, tB_, out});

  std::vector<float> expected(
      {1.f, 3.f, 5.f, 3.f, 13.f, 23.f, 5.f, 23.f, 41.f});

  for (int i = 0; i < 9; ++i) {
    ASSERT_EQ(out[i], expected[i]);
  }
}

TEST(Reductions, ReduceRfactorLike) {
  KernelScope kernel_scope;

  Placeholder in(BufHandle("in", {10, 10}, kFloat));
  std::vector<float> in_(100);
  for (int i = 0; i < 100; ++i) {
    in_[i] = i;
  }
  std::vector<float> in_rf_(10, -2.f);
  std::vector<float> out(1, -1.f);

  Tensor l1 = Reduce("l1", {{10, "i"}}, Sum(), in, {{10, "j"}});
  Placeholder in_rf(BufHandle(l1.buf()));

  Tensor l2 = Reduce("l2", {}, Sum(), in_rf, {{10, "i"}});

  LoopNest loop({l1, l2});
  loop.prepareForCodegen();
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {in, l1, l2});
  cg.call({in_, in_rf_, out});

  ASSERT_EQ(out[0], 99 * 50);
}

TEST(Reductions, ReduceAsProducer) {
  KernelScope kernel_scope;

  const int M = 10;
  VarHandle m("m", kInt);

  Placeholder a(BufHandle("a", {2, 3}, kFloat));
  Placeholder b(BufHandle("b", {2, 3, m}, kFloat));

  Tensor c = Reduce("sum", {{2, "l1"}, {3, "n1"}}, Sum(), b, {{m, "m1"}});
  Tensor d = Compute(
      "scale",
      {{2, "l2"}, {3, "n1"}},
      [&](const VarHandle& l, const VarHandle& n) {
        return c.load(l, n) * a.load(l, n);
      });
  LoopNest loop({d}, {c, d});
  loop.prepareForCodegen();
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {a, b, d, m});

  std::vector<float> aData(2 * 3, 0);
  std::vector<float> bData(2 * 3 * M, 0);
  std::vector<float> dData(2 * 3, 6.0f);

  for (int i = 0; i < 2 * 3; ++i) {
    aData[i] = 6 - i;
    for (int j = 0; j < M; ++j) {
      bData[i * M + j] = j;
    }
  }

  cg.call({aData, bData, dData, M});
  float expected = 0;
  for (int i = 0; i < M; ++i) {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    expected += i;
  }
  for (int i = 0; i < 2 * 3; ++i) {
    ASSERT_EQ(dData[i], expected * (6 - i));
  }
}

TEST(Reductions, ReduceAsConsumer) {
  KernelScope kernel_scope;

  const int M = 10;
  VarHandle m("m", kInt);

  Placeholder a(BufHandle("a", {2, 3, m}, kFloat));
  Placeholder b(BufHandle("b", {2, 3, m}, kFloat));

  Tensor c = Compute(
      "scale",
      {{2, "l2"}, {3, "n1"}, {m, "m1"}},
      [&](const VarHandle& l, const VarHandle& n, const VarHandle& m) {
        return b.load(l, n, m) * a.load(l, n, m);
      });
  Tensor d = Reduce("sum", {{2, "l1"}}, Sum(), c, {{3, "n1"}, {m, "m1"}});
  LoopNest loop({d}, {c, d});
  loop.prepareForCodegen();
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {a, b, d, m});

  std::vector<float> aData(2 * 3 * M, 0);
  std::vector<float> bData(2 * 3 * M, 0);
  std::vector<float> dData(2, 6.0f);

  for (int i = 0; i < 2 * 3; ++i) {
    for (int j = 0; j < M; ++j) {
      bData[i * M + j] = j + 1;
      aData[i * M + j] = 6 - i;
    }
  }

  cg.call({aData, bData, dData, M});
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  float expected[2] = {0, 0};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < M; ++k) {
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        expected[i] += (k + 1) * (6 - (i * 3 + j));
      }
    }
  }

  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(dData[i], expected[i]);
  }
}

TEST(Reductions, SplitReduceAxis) {
  KernelScope kernel_scope;

  Placeholder in(BufHandle("in", {16, 8}, kFloat));

  std::vector<float> in_(16 * 8);
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 8; ++j) {
      in_[i * 8 + j] = i;
    }
  }
  std::vector<float> out(16, -1.f);

  Tensor tensor = Reduce("sum", {{16, "m"}}, Sum(), in, {{8, "n"}});
  LoopNest l({tensor});
  std::vector<ForPtr> loops = l.getLoopStmtsFor(tensor);
  LoopNest::splitWithTail(loops[1], 2);

  l.prepareForCodegen();

  StmtPtr s = l.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {in, tensor});
  cg.call({in_, out});

  for (int i = 0; i < 16; ++i) {
    ASSERT_EQ(out[i], i * 8);
  }
}

TEST(Reductions, SplitNonReduceAxis) {
  KernelScope kernel_scope;

  Placeholder in(BufHandle("in", {16, 8}, kFloat));

  std::vector<float> in_(16 * 8);
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 8; ++j) {
      in_[i * 8 + j] = i;
    }
  }
  std::vector<float> out(16, -1.f);
  Tensor tensor = Reduce("sum", {{16, "m"}}, Sum(), in, {{8, "n"}});
  LoopNest l({tensor});
  std::vector<ForPtr> loops = l.getLoopStmtsFor(tensor);
  LoopNest::splitWithTail(loops[0], 2);
  LoopNest::splitWithTail(loops[0], 2);

  l.prepareForCodegen();

  StmtPtr s = l.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {in, tensor});
  cg.call({in_, out});

  for (int i = 0; i < 16; ++i) {
    ASSERT_EQ(out[i], i * 8);
  }
}

TEST(Reductions, ReorderedReductionInitializer) {
  KernelScope kernel_scope;
  /* From the quip:
  for k in 0..1:  // blockIdx
    for m in 0..128:
      for n in 0..64: // threadIdx
        SumOp(c(k, n), 0, a(k, m, n), {m})
  */

  Placeholder in(BufHandle("in", {1, 12, 6}, kFloat));
  std::vector<float> in_(12 * 6, 1.f);

  Tensor tensor_ = Reduce("sum", {{1, "k"}, {12, "n"}}, Sum(), in, {{6, "m"}});
  LoopNest l_({tensor_});

  l_.prepareForCodegen();
  StmtPtr s_ = Stmt::clone(l_.root_stmt());
  s_ = IRSimplifier::simplify(s_);

  Tensor tensor = Reduce("sum", {{1, "k"}, {12, "n"}}, Sum(), in, {{6, "m"}});
  LoopNest l({tensor});

  auto loops = l.getLoopStmtsFor(tensor);
  loops[0]->set_gpu_block_index(0);
  loops[1]->set_gpu_thread_index(0);

  LoopNest::reorderAxis(loops[1], loops[2]);

  StmtPtr s = l.root_stmt();
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  s = IRSimplifier::simplify(s);

  l.prepareForCodegen();

  s = l.root_stmt();
  s = IRSimplifier::simplify(s);

  std::vector<float> out1(16, -1.f);
  SimpleIREvaluator cg(s_, {in, tensor_});
  cg.call({in_, out1});

  std::vector<float> out2(16, -1.f);
  SimpleIREvaluator cg2(s, {in, tensor});
  cg2.call({in_, out2});

  for (int i = 0; i < 16; ++i) {
    ASSERT_EQ(out1[i], out2[i]);
  }
}

TEST(Reductions, ReduceRfactor) {
  KernelScope kernel_scope;

  const int M = 10;
  const int N = 10;
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);

  Placeholder b(BufHandle("b", {m, n}, kFloat));
  std::vector<float> in(M * N);
  for (int j = 0; j < M * N; ++j) {
    in[j] = j;
  }

  std::vector<float> out(1, -1.f);

  Tensor c = Reduce("sum", {}, Sum(), b, {{m, "m"}, {n, "n"}});
  LoopNest loop({c});
  std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);
  auto c_body = loop.getAllWritesToBuf(c.buf())[1];
  ASSERT_TRUE(loop.rfactor(c_body, loops.at(0)));
  auto rc = NodeFinder<ReduceOp>::find(loop.root_stmt());
  ASSERT_EQ(rc.size(), 2);
  loop.prepareForCodegen();
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {b, c, m, n});

  cg.call({in, out, M, N});
  ASSERT_EQ(out[0], 4950);
}

TEST(Reductions, Reduce3DRfactorInner) {
  KernelScope kernel_scope;

  const int M = 10;
  const int N = 10;
  const int K = 10;
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  VarHandle k("k", kInt);

  Placeholder b(BufHandle("b", {m, n, k}, kFloat));
  std::vector<float> in(M * N * K);
  for (int j = 0; j < M * N * K; ++j) {
    in[j] = j;
  }

  std::vector<float> out(1, -1.f);

  Tensor c = Reduce("sum", {}, Sum(), b, {{m, "m"}, {n, "n"}, {k, "k"}});
  LoopNest loop({c});
  std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);
  auto c_body = loop.getAllWritesToBuf(c.buf())[1];
  ASSERT_FALSE(loop.rfactor(c_body, loops.at(2)));
  auto rc = NodeFinder<ReduceOp>::find(loop.root_stmt());
  ASSERT_EQ(rc.size(), 1);
  loop.prepareForCodegen();
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {b, c, m, n, k});

  cg.call({in, out, M, N, K});
  ASSERT_EQ(out[0], 499500);
}

TEST(Reductions, Reduce3DRfactorOuter) {
  KernelScope kernel_scope;

  const int M = 10;
  const int N = 10;
  const int K = 10;
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  VarHandle k("k", kInt);

  Placeholder b(BufHandle("b", {m, n, k}, kFloat));
  std::vector<float> in(M * N * K);
  for (int j = 0; j < M * N * K; ++j) {
    in[j] = j;
  }

  std::vector<float> out(1, -1.f);

  Tensor c = Reduce("sum", {}, Sum(), b, {{m, "m"}, {n, "n"}, {k, "k"}});
  LoopNest loop({c});
  std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);
  auto c_body = loop.getAllWritesToBuf(c.buf())[1];
  ASSERT_TRUE(loop.rfactor(c_body, loops.at(0)));
  auto rc = NodeFinder<ReduceOp>::find(loop.root_stmt());
  ASSERT_EQ(rc.size(), 2);
  loop.prepareForCodegen();
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  SimpleIREvaluator cg(s, {b, c, m, n, k});
  cg.call({in, out, M, N, K});
  ASSERT_EQ(out[0], 499500);
}

TEST(Reductions, ReduceRepeatedInternalRfactor) {
  KernelScope kernel_scope;

  Placeholder in_(BufHandle("in_", {2, 3, 4, 5, 6}, kFloat));
  const int InputSize = 2 * 3 * 4 * 5 * 6;

  std::vector<float> in(InputSize, 1.f);
  std::vector<float> out(1, -1.f);
  std::vector<float> ref(1, -1.f);

  Tensor c = Reduce(
      "sum",
      {},
      Sum(),
      in_,
      {{2, "a"}, {3, "b"}, {4, "c"}, {5, "d"}, {6, "e"}});
  LoopNest orig_loop({c});

  // Try rfactoring N outer loops
  for (int rfac_number = 1; rfac_number < 5; rfac_number++) {
    LoopNest refloop(orig_loop);
    LoopNest loop(orig_loop);
    refloop.prepareForCodegen();
    SimpleIREvaluator ref_cg(
        IRSimplifier::simplify(refloop.root_stmt()), {in_, c});
    ref_cg.call({in, ref});

    BufPtr tmp_buf = c.buf();

    for (int idx = 0; idx < rfac_number; idx++) {
      auto reduce = loop.getAllWritesToBuf(tmp_buf)[1];
      ASSERT_TRUE(loop.rfactor(
          reduce, loop.getLoopStmtsFor(tmp_buf).at(idx), &tmp_buf));
    }

    loop.prepareForCodegen();
    StmtPtr s = loop.root_stmt();
    s = IRSimplifier::simplify(s);

    SimpleIREvaluator cg(s, {in_, c});
    cg.call({in, out});

    ASSERT_EQ(ref[0], out[0]);
  }
}

// Split a reduction axis with a tail loop.
TEST(Reductions, ReduceSplitTail) {
  KernelScope kernel_scope;

  const int M = 10;
  const int N = 10;
  const int K = 10;

  Placeholder b(BufHandle("b", {M, N, K}, kFloat));
  std::vector<float> in(M * N * K);
  for (int j = 0; j < M * N * K; ++j) {
    in[j] = j;
  }

  for (int i = 0; i < 3; ++i) {
    std::vector<float> out(M, -1.f);

    Tensor c = Reduce("sum", {{M, "m"}}, Sum(), b, {{N, "n"}, {K, "k"}});
    LoopNest loop({c});
    std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);
    LoopNest::splitWithTail(loops[i], 8);

    loop.prepareForCodegen();
    StmtPtr s = loop.root_stmt();
    s = IRSimplifier::simplify(s);

    SimpleIREvaluator cg(s, {b, c});

    cg.call({in, out});
    ASSERT_EQ(out[0], 4950);
  }
}

// Split a reduction axis cleanly so there is no tail loop.
TEST(Reductions, ReduceSplitNoTail) {
  KernelScope kernel_scope;

  const int M = 10;
  const int N = 10;
  const int K = 10;
  Placeholder b(BufHandle("b", {M, N, K}, kFloat));
  std::vector<float> in(M * N * K);
  for (int j = 0; j < M * N * K; ++j) {
    in[j] = j;
  }

  for (int i = 0; i < 3; ++i) {
    std::vector<float> out(M, -1.f);

    Tensor c = Reduce("sum", {{M, "m"}}, Sum(), b, {{N, "n"}, {K, "k"}});
    LoopNest loop({c});
    std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);
    LoopNest::splitWithTail(loops[i], 5);

    loop.prepareForCodegen();
    StmtPtr s = loop.root_stmt();
    s = IRSimplifier::simplify(s);

    SimpleIREvaluator cg(s, {b, c});

    cg.call({in, out});
    ASSERT_EQ(out[0], 4950);
  }
}

// Split a reduction axis with only a tail loop (the split loop will be size 0
// and eliminated out).
TEST(Reductions, ReduceOverSplitTail) {
  KernelScope kernel_scope;

  const int M = 10;
  const int N = 10;
  const int K = 10;

  Placeholder b(BufHandle("b", {M, N, K}, kFloat));
  std::vector<float> in(M * N * K);
  for (int j = 0; j < M * N * K; ++j) {
    in[j] = j;
  }

  for (int i = 0; i < 3; ++i) {
    std::vector<float> out(M, -1.f);

    Tensor c = Reduce("sum", {{M, "m"}}, Sum(), b, {{N, "n"}, {K, "k"}});
    LoopNest loop({c});
    std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);
    LoopNest::splitWithTail(loops[i], 16);

    loop.prepareForCodegen();
    StmtPtr s = loop.root_stmt();
    s = IRSimplifier::simplify(s);

    SimpleIREvaluator cg(s, {b, c});

    cg.call({in, out});
    ASSERT_EQ(out[0], 4950);
  }
}

// Split a reduction axis with a mask.
TEST(Reductions, ReduceSplitMask) {
  KernelScope kernel_scope;

  const int M = 10;
  const int N = 10;
  const int K = 10;

  Placeholder b(BufHandle("b", {M, N, K}, kFloat));
  std::vector<float> in(M * N * K);
  for (int j = 0; j < M * N * K; ++j) {
    in[j] = j;
  }

  for (int i = 0; i < 3; ++i) {
    std::vector<float> out(M, -1.f);

    Tensor c = Reduce("sum", {{M, "m"}}, Sum(), b, {{N, "n"}, {K, "k"}});
    LoopNest loop({c});
    std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);
    LoopNest::splitWithMask(loops[i], 8);

    loop.prepareForCodegen();
    StmtPtr s = loop.root_stmt();
    s = IRSimplifier::simplify(s);

    SimpleIREvaluator cg(s, {b, c});

    cg.call({in, out});
    ASSERT_EQ(out[0], 4950);
  }
}

// Split a reduction axis cleanly not requiring a mask.
TEST(Reductions, ReduceSplitNoMask) {
  KernelScope kernel_scope;

  const int M = 10;
  const int N = 10;
  const int K = 10;
  Placeholder b(BufHandle("b", {M, N, K}, kFloat));
  std::vector<float> in(M * N * K);
  for (int j = 0; j < M * N * K; ++j) {
    in[j] = j;
  }

  for (int i = 0; i < 3; ++i) {
    std::vector<float> out(M, -1.f);

    Tensor c = Reduce("sum", {{M, "m"}}, Sum(), b, {{N, "n"}, {K, "k"}});
    LoopNest loop({c});
    std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);
    LoopNest::splitWithMask(loops[i], 5);

    loop.prepareForCodegen();
    StmtPtr s = loop.root_stmt();
    s = IRSimplifier::simplify(s);

    SimpleIREvaluator cg(s, {b, c});

    cg.call({in, out});
    ASSERT_EQ(out[0], 4950);
  }
}

// Split a reduction axis with all logic in the mask.
TEST(Reductions, ReduceOverSplitMask) {
  KernelScope kernel_scope;

  const int M = 10;
  const int N = 10;
  const int K = 10;

  Placeholder b(BufHandle("b", {M, N, K}, kFloat));
  std::vector<float> in(M * N * K);
  for (int j = 0; j < M * N * K; ++j) {
    in[j] = j;
  }

  for (int i = 0; i < 3; ++i) {
    std::vector<float> out(M, -1.f);

    Tensor c = Reduce("sum", {{M, "m"}}, Sum(), b, {{N, "n"}, {K, "k"}});
    LoopNest loop({c});
    std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);
    LoopNest::splitWithMask(loops[i], 16);

    loop.prepareForCodegen();
    StmtPtr s = loop.root_stmt();
    s = IRSimplifier::simplify(s);

    SimpleIREvaluator cg(s, {b, c});

    cg.call({in, out});
    ASSERT_EQ(out[0], 4950);
  }
}

// Test an rfactor when there are two ReduceOps in the graph due to a
// splitWithTail.
TEST(Reductions, ReduceSplitRfactor) {
  KernelScope kernel_scope;

  const int M = 2;
  const int N = 10;
  const int K = 10;
  const int SPLIT_FACTOR = 4;

  Placeholder b(BufHandle("b", {M, N, K}, kFloat));
  std::vector<float> in(M * N * K);
  for (int m = 0; m < M; ++m) {
    for (int j = 0; j < N * K; ++j) {
      in[m * N * K + j] = j;
    }
  }

  std::vector<float> out(M, -1.f);

  Tensor c = Reduce("sum", {{M, "m"}}, Sum(), b, {{N, "n"}, {K, "k"}});
  LoopNest loop({c});
  std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);
  LoopNest::splitWithTail(loops[2], SPLIT_FACTOR);

  auto c_body = loop.getAllWritesToBuf(c.buf())[2];
  auto all_loops = loop.getAllLoopNestsWritingToBuf(c.buf());
  ASSERT_TRUE(all_loops.size() == 3 && all_loops.at(2).size() == 3);
  LoopNest::reorderAxis(all_loops[2][1], all_loops[2][2]);
  all_loops = loop.getAllLoopNestsWritingToBuf(c.buf());
  ASSERT_TRUE(all_loops.size() == 3 && all_loops.at(2).size() == 3);
  ASSERT_TRUE(loop.rfactor(c_body, all_loops[2][1]));
  loop.prepareForCodegen();
  loop.simplify();
  StmtPtr s = loop.root_stmt();

  SimpleIREvaluator cg(s, {b, c});

  cg.call({in, out});
  for (int i = 0; i < M; ++i) {
    ASSERT_EQ(out[0], 4950);
  }
}

// Test an rfactor which ends up being eliminated since the total loop size is
// smaller than the split factor.
TEST(Reductions, ReduceOverSplitRfactor) {
  KernelScope kernel_scope;

  const int N = 10;
  const int K = 10;
  const int SPLIT_FACTOR = 16;

  Placeholder b(BufHandle("b", {N, K}, kFloat));
  std::vector<float> in(N * K);
  for (int j = 0; j < N * K; ++j) {
    in[j] = j;
  }

  std::vector<float> out(1, -1.f);

  Tensor c = Reduce("sum", {}, Sum(), b, {{N, "n"}, {K, "k"}});
  LoopNest loop({c});
  std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr i, t;
  LoopNest::splitWithTail(loops[1], SPLIT_FACTOR, &i, &t);
  LoopNest::reorderAxis(loops[0], i);

  auto all_loops = loop.getAllLoopNestsWritingToBuf(c.buf());
  ASSERT_TRUE(all_loops.size() == 3 && all_loops.at(1).size() == 3);
  auto c_body = loop.getAllWritesToBuf(c.buf())[1];
  ASSERT_TRUE(loop.rfactor(c_body, all_loops[1][0]));
  LoopNest::reorderAxis(all_loops[1][0], all_loops[1][2]);

  loop.prepareForCodegen();
  loop.simplify();
  StmtPtr s = loop.root_stmt();

  SimpleIREvaluator cg(s, {b, c});

  cg.call({in, out});
  ASSERT_EQ(out[0], 4950);

  std::ostringstream oss;
  oss << *s;

  // Check the IR to verify the rfactored reduce is eliminated.
  // TODO: The alloc free should be eliminated here since it is size 0.
  const std::string& verification_pattern =
      R"IR(
# CHECK: Allocate(tmp_buf); // dtype=float, dims=[0]
# CHECK: sum[0] = 0.f;
# CHECK: for (int n = 0; n < 10; n++) {
# CHECK:   for (int k_tail = 0; k_tail < 10; k_tail++) {
# CHECK:     sum[0] = (sum[0]) + (b[k_tail + 10 * n]);
# CHECK:   }
# CHECK: }
# CHECK: Free(tmp_buf);)IR";
  // TODO: rfactor output is not consistent yet, will fix (@nickg).
  // torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST(Reductions, ReduceInlineReduction) {
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;

  Placeholder a_buf("a", kFloat, {M});
  Placeholder b_buf("b", kFloat, {M, N, K});

  Tensor x = Reduce("x", {{M, "m1"}}, Sum(), b_buf, {{N, "n1"}, {K, "k1"}});
  Tensor y = Compute("y", {{M, "m2"}}, [&](const VarHandle& m) {
    return a_buf.load(m) + x.load(m);
  });

  PaddedBuffer<float> a_v(M);
  PaddedBuffer<float> b_v(M, N, K);

  for (int i = 0; i < M; i++) {
    a_v(i) = i * i;
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        b_v(i, j, k) = j * j * k;
      }
    }
  }

  LoopNest l1({y}, {x, y});
  // Cannot inline a reduction computation
  ASSERT_FALSE(l1.computeInline(x.buf()));
}

TEST(Reductions, ReduceInlineConsumer) {
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;

  Placeholder a_buf("a", kFloat, {M, N, K});
  Placeholder b_buf("b", kFloat, {M, N, K});

  Tensor x = Compute(
      "x",
      {{M, "m1"}, {N, "n1"}, {K, "k1"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n, k) + b_buf.load(m, n, k);
      });
  Tensor y = Reduce("y", {{M, "m2"}}, Sum(), x, {{N, "n2"}, {K, "k2"}});

  PaddedBuffer<float> a_v(M, N, K);
  PaddedBuffer<float> b_v(M, N, K);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        a_v(i, j, k) = i * i + k;
        b_v(i, j, k) = j * j + k;
      }
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

  PaddedBuffer<float> y_1(M);
  PaddedBuffer<float> y_2(M);

  eval1(a_v, b_v, y_1);
  eval2(a_v, b_v, y_2);
  ExpectAllNear(y_1, y_2, 1e-5);
  std::ostringstream oss1, oss2;
  oss1 << *stmt1;
  oss2 << *stmt2;
  ASSERT_GT(oss1.str().size(), oss2.str().size());
}

TEST(Reductions, ReduceInlineReducerInternal) {
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;

  Placeholder a_buf("a", kFloat, {M, N, K});
  Placeholder b_buf("b", kFloat, {M, N, K});

  Tensor x = Compute(
      "x",
      {{M, "m1"}, {N, "n1"}, {K, "k1"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n, k) + b_buf.load(m, n, k);
      });

  Reducer minimum(ExprHandle(0.f), [&](ExprHandle a, ExprHandle b) {
    return Add::make(ExprHandle(1.f), Min::make(a, b, false));
  });
  Tensor y = Reduce("y", {{M, "m2"}}, minimum, x, {{N, "n2"}, {K, "k2"}});

  PaddedBuffer<float> a_v(M, N, K);
  PaddedBuffer<float> b_v(M, N, K);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        a_v(i, j, k) = i * i + k;
        b_v(i, j, k) = j * j + k;
      }
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

  PaddedBuffer<float> y_1(M);
  PaddedBuffer<float> y_2(M);

  eval1(a_v, b_v, y_1);
  eval2(a_v, b_v, y_2);
  ExpectAllNear(y_1, y_2, 1e-5);
  std::ostringstream oss1, oss2;
  oss1 << *stmt1;
  oss2 << *stmt2;
  ASSERT_GT(oss1.str().size(), oss2.str().size());
}

TEST(Reductions, ReductionCacheAccessesOperatorAxis) {
  KernelScope kernel_scope;

  int L = 4;
  int N = 3;
  int M = 2;

  Placeholder a(BufHandle("a", {L, N, M}, kFloat));
  Placeholder b(BufHandle("b", {L, N, M}, kFloat));

  Tensor c = Compute(
      "scale",
      {{L, "l2"}, {N, "n1"}, {M, "m1"}},
      [&](const VarHandle& l, const VarHandle& n, const VarHandle& m) {
        return b.load(l, n, m) * a.load(l, n, m);
      });
  Tensor d = Reduce("sum", {{L, "l1"}}, Sum(), c, {{N, "n1"}, {M, "m1"}});

  Tensor e = Compute("scale", {{L, "l"}}, [&](const VarHandle& l) {
    return b.load(0, 0, l) * d.load(l);
  });

  LoopNest l({e}, {c, d, e});
  LoopNest l_before(l);
  l_before.prepareForCodegen();
  SimpleIREvaluator cg_before(l_before.root_stmt(), {a, b, e});

  StmtPtr d_loop = l.getLoopStmtsFor(d)[0];
  l.cacheAccesses(d.buf(), "d_local", d_loop);
  l.prepareForCodegen();

  StmtPtr result = IRSimplifier::simplify(l.root_stmt());
  SimpleIREvaluator cg_after(result, {a, b, e});

  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
#CHECK: Allocate(d_local); // dtype=float, dims=[4]
#CHECK: for (int l1
#CHECK:   d_local[l1] = 0.f
#CHECK:   for (int n1
#CHECK:     for (int m1
#CHECK:       d_local[l1] = (d_local[l1]) + (scale[
#CHECK:     }
#CHECK:   }
#CHECK: }
#CHECK: for (int i
#CHECK:   sum[i] = d_local[i]
#CHECK: Free(d_local);
#CHECK-NOT: d_local
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());

  PaddedBuffer<float> a_v(L, M, N, "a");
  PaddedBuffer<float> b_v(L, M, N, "b");
  PaddedBuffer<float> c_v(L, M, N, "c");
  PaddedBuffer<float> d_v(L, "d");
  PaddedBuffer<float> e_before(L, "e_before");
  PaddedBuffer<float> e_after(L, "e_after");

  for (int l = 0; l < L; l++) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        a_v(l, m, n) = at::randn({1}).item().to<float>();
        b_v(l, m, n) = at::randn({1}).item().to<float>();
      }
    }
  }

  cg_before.call({a_v, b_v, e_before});
  cg_after.call({a_v, b_v, e_after});

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  ExpectAllNear(e_before, e_after, 1e-5);
}

TEST(Reductions, ReductionCacheAccessesOuterReduceAxis) {
  KernelScope kernel_scope;

  int L = 4;
  int N = 3;
  int M = 2;

  Placeholder a(BufHandle("a", {L, N, M}, kFloat));
  Placeholder b(BufHandle("b", {L, N, M}, kFloat));

  Tensor c = Compute(
      "scale",
      {{L, "l2"}, {N, "n1"}, {M, "m1"}},
      [&](const VarHandle& l, const VarHandle& n, const VarHandle& m) {
        return b.load(l, n, m) * a.load(l, n, m);
      });
  Tensor d = Reduce("sum", {{L, "l1"}}, Sum(), c, {{N, "n1"}, {M, "m1"}});

  Tensor e = Compute("scale", {{L, "l"}}, [&](const VarHandle& l) {
    return b.load(0, 0, l) * d.load(l);
  });

  LoopNest l({e}, {c, d, e});
  LoopNest l_before(l);
  l_before.prepareForCodegen();
  SimpleIREvaluator cg_before(l_before.root_stmt(), {a, b, e});

  StmtPtr d_loop = l.getLoopStmtsFor(d)[1];
  l.cacheAccesses(d.buf(), "d_local", d_loop);
  l.prepareForCodegen();

  StmtPtr result = IRSimplifier::simplify(l.root_stmt());
  SimpleIREvaluator cg_after(result, {a, b, e});

  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
#CHECK: Allocate(d_local); // dtype=float, dims=[1]
#CHECK: sum[l1] = 0
#CHECK: d_local[0] = sum[l1]
#CHECK: for (int n1
#CHECK:   for (int m1
#CHECK: d_local[0] = (d_local[0]) + (scale[
#CHECK:   }
#CHECK: }
#CHECK: sum[l1] = d_local[0]
#CHECK: Free(d_local);
#CHECK-NOT: d_local
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());

  PaddedBuffer<float> a_v(L, M, N, "a");
  PaddedBuffer<float> b_v(L, M, N, "b");
  PaddedBuffer<float> c_v(L, M, N, "c");
  PaddedBuffer<float> d_v(L, "d");
  PaddedBuffer<float> e_before(L, "e_before");
  PaddedBuffer<float> e_after(L, "e_after");

  for (int l = 0; l < L; l++) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        a_v(l, m, n) = at::randn({1}).item().to<float>();
        b_v(l, m, n) = at::randn({1}).item().to<float>();
      }
    }
  }

  cg_before.call({a_v, b_v, e_before});
  cg_after.call({a_v, b_v, e_after});

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  ExpectAllNear(e_before, e_after, 1e-5);
}

TEST(Reductions, ReductionCacheAccessesInnerReduceAxis) {
  KernelScope kernel_scope;

  int L = 4;
  int N = 3;
  int M = 2;

  Placeholder a(BufHandle("a", {L, N, M}, kFloat));
  Placeholder b(BufHandle("b", {L, N, M}, kFloat));

  Tensor c = Compute(
      "scale",
      {{L, "l2"}, {N, "n1"}, {M, "m1"}},
      [&](const VarHandle& l, const VarHandle& n, const VarHandle& m) {
        return b.load(l, n, m) * a.load(l, n, m);
      });
  Tensor d = Reduce("sum", {{L, "l1"}}, Sum(), c, {{N, "n1"}, {M, "m1"}});

  Tensor e = Compute("scale", {{L, "l"}}, [&](const VarHandle& l) {
    return b.load(0, 0, l) * d.load(l);
  });

  LoopNest l({e}, {c, d, e});
  LoopNest l_before(l);
  l_before.prepareForCodegen();
  SimpleIREvaluator cg_before(l_before.root_stmt(), {a, b, e});

  StmtPtr d_loop = l.getLoopStmtsFor(d)[2];
  l.cacheAccesses(d.buf(), "d_local", d_loop);
  l.prepareForCodegen();

  StmtPtr result = IRSimplifier::simplify(l.root_stmt());
  SimpleIREvaluator cg_after(result, {a, b, e});

  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
#CHECK: Allocate(d_local); // dtype=float, dims=[1]
#CHECK: sum[l1] = 0
#CHECK: for (int n1
#CHECK:   d_local[0] = 0
#CHECK:   for (int m1
#CHECK:     d_local[0] = (d_local[0]) + (scale[
#CHECK:   }
#CHECK:   sum[l1] = (sum[l1]) + (d_local[0])
#CHECK: }
#CHECK: Free(d_local);
#CHECK-NOT: d_local
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());

  PaddedBuffer<float> a_v(L, M, N, "a");
  PaddedBuffer<float> b_v(L, M, N, "b");
  PaddedBuffer<float> c_v(L, M, N, "c");
  PaddedBuffer<float> d_v(L, "d");
  PaddedBuffer<float> e_before(L, "e_before");
  PaddedBuffer<float> e_after(L, "e_after");

  for (int l = 0; l < L; l++) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        a_v(l, m, n) = at::randn({1}).item().to<float>();
        b_v(l, m, n) = at::randn({1}).item().to<float>();
      }
    }
  }

  cg_before.call({a_v, b_v, e_before});
  cg_after.call({a_v, b_v, e_after});

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  ExpectAllNear(e_before, e_after, 1e-5);
}

TEST(Reductions, ReductionCacheBodyAccess) {
  KernelScope kernel_scope;

  Placeholder a(BufHandle("a", {24, 32, 12}, kFloat));
  Placeholder b(BufHandle("b", {24, 32, 12}, kFloat));

  Tensor c = Compute(
      "scale",
      {{24, "l2"}, {32, "n1"}, {12, "m1"}},
      [&](const VarHandle& l, const VarHandle& n, const VarHandle& m) {
        return b.load(l, n, m) * a.load(l, n, m);
      });
  Tensor d = Reduce("sum", {{24, "l1"}}, Sum(), c, {{32, "n1"}, {12, "m1"}});

  Tensor e = Compute("scale", {{24, "l"}}, [&](const VarHandle& l) {
    return b.load(0, 0, l) * d.load(l);
  });

  LoopNest l({e}, {c, d, e});

  StmtPtr d_loop = l.getLoopStmtsFor(d)[1];
  l.cacheAccesses(c.buf(), "scale_local", d_loop);

  l.prepareForCodegen();
  StmtPtr result = IRSimplifier::simplify(l.root_stmt());

  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
#CHECK: Allocate(scale_local); // dtype=float, dims=[1, 32, 12]
#CHECK: for (int j = 0; j < 32; j++) {
#CHECK:   for (int k = 0; k < 12; k++) {
#CHECK:     scale_local[k + 12 * j] = scale[(k + 12 * j) + 384 * l1];
#CHECK: sum[l1] = (sum[l1]) + (scale_local[m1_1 + 12 * n1_1]);
#CHECK: scale_1[l] = (b[l]) * (sum[l]);
#CHECK: Free(scale_local);
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

TEST(Reductions, ReductionCacheConsumerAccess) {
  KernelScope kernel_scope;

  Placeholder a(BufHandle("a", {24, 32, 12}, kFloat));
  Placeholder b(BufHandle("b", {24, 32, 12}, kFloat));

  Tensor c = Compute(
      "scale",
      {{24, "l2"}, {32, "n1"}, {12, "m1"}},
      [&](const VarHandle& l, const VarHandle& n, const VarHandle& m) {
        return b.load(l, n, m) * a.load(l, n, m);
      });
  Tensor d = Reduce("sum", {{24, "l1"}}, Sum(), c, {{32, "n1"}, {12, "m1"}});

  Tensor e = Compute("scale", {{24, "l"}}, [&](const VarHandle& l) {
    return b.load(0, 0, l) * d.load(l);
  });

  LoopNest l({e}, {c, d, e});

  LoopNest::splitWithMask(l.getLoopStmtsFor(e)[0], 4);

  StmtPtr e_loop = l.getLoopStmtsFor(e)[1];
  l.cacheAccesses(d.buf(), "sum_local", e_loop);
  l.prepareForCodegen();

  StmtPtr result = IRSimplifier::simplify(l.root_stmt());

  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
#CHECK: Allocate(sum_local); // dtype=float, dims=[4]
#CHECK: sum[l1] = (sum[l1]) + (scale[
#CHECK: for (int i = 0; i < 4
#CHECK:   sum_local[i] = sum[i + 4 * l_outer];
#CHECK:   scale_1[l_inner + 4 * l_outer] = (b[l_inner + 4 * l_outer]) * (sum_local[l_inner]);
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

TEST(Reductions, ReductionSplitCacheConsumerAccess) {
  KernelScope kernel_scope;

  Placeholder a(BufHandle("a", {24, 32, 12}, kFloat));
  Placeholder b(BufHandle("b", {24, 32, 12}, kFloat));

  Tensor c = Compute(
      "scale",
      {{24, "l2"}, {32, "n1"}, {12, "m1"}},
      [&](const VarHandle& l, const VarHandle& n, const VarHandle& m) {
        return b.load(l, n, m) * a.load(l, n, m);
      });
  Tensor d = Reduce("sum", {{24, "l1"}}, Sum(), c, {{32, "n1"}, {12, "m1"}});

  Tensor e = Compute("scale", {{24, "l"}}, [&](const VarHandle& l) {
    return b.load(0, 0, l) * d.load(l);
  });

  LoopNest l({e}, {c, d, e});

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr inner;

  // Split outer reduction axis.
  LoopNest::splitWithMask(l.getLoopStmtsFor(d)[0], 4, &inner);

  // Split reduction consumer.
  LoopNest::splitWithMask(l.getLoopStmtsFor(e)[0], 4, &inner);

  l.cacheAccesses(d.buf(), "sum_local", inner);
  l.prepareForCodegen();

  StmtPtr result = IRSimplifier::simplify(l.root_stmt());

  // reduction changes but cache does not.
  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
#CHECK: Allocate(sum_local); // dtype=float, dims=[4]
#CHECK: sum[l1_inner + 4 * l1_outer] = (sum[l1_inner + 4 * l1_outer]) + (scale[((m1_1 + 12 * n1_1) + 1536 * l1_outer) + 384 * l1_inner]);
#CHECK: for (int i = 0; i < 4
#CHECK:   sum_local[i] = sum[i + 4 * l_outer];
#CHECK:   scale_1[l_inner + 4 * l_outer] = (b[l_inner + 4 * l_outer]) * (sum_local[l_inner]);
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

TEST(Reductions, ReductionReorderCacheConsumerAccess) {
  KernelScope kernel_scope;

  Placeholder a(BufHandle("a", {24, 32, 12}, kFloat));
  Placeholder b(BufHandle("b", {24, 32, 12}, kFloat));

  Tensor c = Compute(
      "scale",
      {{24, "l2"}, {32, "n1"}, {12, "m1"}},
      [&](const VarHandle& l, const VarHandle& n, const VarHandle& m) {
        return b.load(l, n, m) * a.load(l, n, m);
      });
  Tensor d = Reduce("sum", {{24, "l1"}}, Sum(), c, {{32, "n1"}, {12, "m1"}});

  Tensor e = Compute("scale", {{24, "l"}}, [&](const VarHandle& l) {
    return b.load(0, 0, l) * d.load(l);
  });

  LoopNest l({e}, {c, d, e});

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr inner;

  // reorder outer reduction axes.
  auto loops = l.getLoopStmtsFor(d);
  LoopNest::reorderAxis(loops[0], loops[1]);

  // Split reduction consumer.
  LoopNest::splitWithMask(l.getLoopStmtsFor(e)[0], 4, &inner);

  l.cacheAccesses(d.buf(), "sum_local", inner);
  l.prepareForCodegen();

  StmtPtr result = IRSimplifier::simplify(l.root_stmt());

  // neither reduction body not cache changes.
  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
#CHECK: Allocate(sum_local); // dtype=float, dims=[4]
#CHECK: sum[l1] = (sum[l1]) + (scale[(m1_1 + 12 * n1_1) + 384 * l1]);
#CHECK: for (int i = 0; i < 4
#CHECK:   sum_local[i] = sum[i + 4 * l_outer];
#CHECK: scale_1[l_inner + 4 * l_outer] = (b[l_inner + 4 * l_outer]) * (sum_local[l_inner]);
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

TEST(Reductions, ReductionRfactorCacheTempOuter) {
  KernelScope kernel_scope;

  const int M = 10;
  const int N = 10;
  const int K = 10;
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  VarHandle k("k", kInt);

  Placeholder b(BufHandle("B", {m, n, k}, kFloat));
  std::vector<float> in(M * N * K);
  for (int j = 0; j < M * N * K; ++j) {
    in[j] = j;
  }

  std::vector<float> out(1, -1.f);

  Tensor c = Reduce("sum", {}, Sum(), b, {{m, "a"}, {n, "b"}, {k, "c"}});
  LoopNest loop({c});

  std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);
  LoopNest::reorderAxis(loops.at(0), loops.at(1));
  loops = loop.getLoopStmtsFor(c);
  auto c_body = loop.getAllWritesToBuf(c.buf())[1];
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  BufPtr rfac_buf;
  ASSERT_TRUE(loop.rfactor(c_body, loops.at(0), &rfac_buf));
  loop.distributeLoop(loops.at(0));

  auto all_loops = loop.getAllLoopNestsWritingToBuf(rfac_buf);
  ASSERT_TRUE(all_loops.size() == 2 && all_loops.at(1).size() == 3);
  LoopNest::reorderAxis(all_loops[1][0], all_loops[1][1]);

  all_loops = loop.getAllLoopNestsWritingToBuf(rfac_buf);
  LoopNest::cacheAccesses(rfac_buf, "tmp", all_loops[1][1]);
  loop.simplify();
  loop.prepareForCodegen();
  StmtPtr s = loop.root_stmt();

  std::ostringstream oss;
  oss << *s;
  const std::string& expected_ir =
      R"IR(
#CHECK: Allocate(sum_rfac); // dtype=float, dims=[n]
#CHECK: Allocate(tmp); // dtype=float, dims=[n]
#CHECK: for (int a = 0; a < m
#CHECK:   for (int i = 0; i < n
#CHECK:     tmp[i] = 0
#CHECK:   }
#CHECK:   for (int b = 0; b < n
#CHECK:     for (int c
#CHECK:       tmp[b] = (tmp[b]) + (B[
#CHECK:     }
#CHECK:   }
#CHECK:   for (int i = 0; i < n
#CHECK:     sum_rfac[i] = (sum_rfac[i]) + (tmp[i]);
#CHECK:   }
#CHECK:   Free(tmp);
#CHECK-NOT: tmp
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());

  SimpleIREvaluator cg(s, {b, c, m, n, k});

  cg.call({in, out, M, N, K});
  ASSERT_EQ(out[0], 499500);
}

TEST(Reductions, ReductionRfactorCacheTempInner) {
  KernelScope kernel_scope;

  const int M = 10;
  const int N = 10;
  const int K = 10;
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  VarHandle k("k", kInt);

  Placeholder b(BufHandle("B", {m, n, k}, kFloat));
  std::vector<float> in(M * N * K);
  for (int j = 0; j < M * N * K; ++j) {
    in[j] = j;
  }

  std::vector<float> out(1, -1.f);

  Tensor c = Reduce("sum", {}, Sum(), b, {{m, "a"}, {n, "b"}, {k, "c"}});
  LoopNest loop({c});
  std::vector<ForPtr> loops = loop.getLoopStmtsFor(c);
  auto c_body = loop.getAllWritesToBuf(c.buf())[1];

  LoopNest::reorderAxis(loops.at(0), loops.at(1));
  loops = loop.getLoopStmtsFor(c);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  BufPtr rfac_buf;
  ASSERT_TRUE(loop.rfactor(c_body, loops.at(0), &rfac_buf));
  loop.distributeLoop(loops.at(0));
  auto all_loops = loop.getAllLoopNestsWritingToBuf(rfac_buf);
  ASSERT_TRUE(all_loops.size() == 2 && all_loops.at(1).size() == 3);
  LoopNest::reorderAxis(all_loops[1][0], all_loops[1][1]);

  all_loops = loop.getAllLoopNestsWritingToBuf(rfac_buf);
  ASSERT_TRUE(all_loops.size() == 2 && all_loops.at(1).size() == 3);
  LoopNest::cacheAccesses(rfac_buf, "tmp", all_loops[1][2]);
  loop.prepareForCodegen();
  loop.simplify();
  StmtPtr s = loop.root_stmt();

  std::ostringstream oss;
  oss << *s;
  const std::string& expected_ir =
      R"IR(
#CHECK: Allocate(sum_rfac); // dtype=float, dims=[n]
#CHECK: Allocate(tmp); // dtype=float, dims=[1]
#CHECK: for (int a = 0; a < m
#CHECK:   for (int b = 0; b < n
#CHECK:     tmp[0] = 0
#CHECK:     for (int c
#CHECK:       tmp[0] = (tmp[0]) + (B[
#CHECK:     }
#CHECK:   sum_rfac[b] = (sum_rfac[b]) + (tmp[0]);
#CHECK:   Free(tmp);
#CHECK-NOT: tmp
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());

  SimpleIREvaluator cg(s, {b, c, m, n, k});

  cg.call({in, out, M, N, K});
  ASSERT_EQ(out[0], 499500);
}

TEST(Reductions, ReductionVectorize) {
  KernelScope kernel_scope;

  std::vector<float> in_(8 * 8);
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      in_[i * 8 + j] = i;
    }
  }
  std::vector<float> out_before(8, -1.f);
  std::vector<float> out_after(8, -1.f);

  Placeholder in(BufHandle("in", {8, 8}, kFloat));

  Tensor tensor = Reduce("sum", {{8, "m"}}, Sum(), in, {{8, "n"}});
  LoopNest l_before({tensor});
  LoopNest l(l_before);
  l_before.prepareForCodegen();
  SimpleIREvaluator cg_before(l_before.root_stmt(), {in, tensor});
  cg_before.call({in_, out_before});

  ASSERT_TRUE(LoopNest::vectorize(l.getLoopStmtsFor(tensor)[0]));

  StmtPtr s = l.root_stmt();
  s = IRSimplifier::simplify(s);

  std::ostringstream oss;
  oss << *s;
  const std::string& expected_ir =
      R"IR(
#CHECK: sum[Ramp(0, 1, 8)] = Broadcast(0.f, 8);
#CHECK: for (int n = 0; n < 8; n++) {
#CHECK: sum[Ramp(0, 1, 8)] = ReduceOp((sum[Ramp(0, 1, 8)]) + (in[Ramp(n, 8, 8)]), reduce_args={n});
#CHECK: }
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());

  // Vectorizing should not change result.
  l.prepareForCodegen();
  s = IRSimplifier::simplify(l.root_stmt());
  SimpleIREvaluator cg_after(s, {in, tensor});
  cg_after.call({in_, out_after});
  for (int i = 0; i < 8; ++i) {
    ASSERT_EQ(out_before[i], out_after[i]);
  }
}

TEST(Reductions, ReductionVectorizeInner) {
  KernelScope kernel_scope;

  Placeholder in(BufHandle("in", {8, 8}, kFloat));

  Tensor tensor = Reduce("sum", {{8, "m"}}, Sum(), in, {{8, "n"}});
  LoopNest l({tensor});

  ASSERT_FALSE(LoopNest::vectorize(l.getLoopStmtsFor(tensor)[1]));
}

TEST(Reductions, ReductionVectorizeRfactor) {
  KernelScope kernel_scope;

  std::vector<float> in_(8 * 8);
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      in_[i * 8 + j] = i;
    }
  }
  std::vector<float> out_before(1, -1.f);
  std::vector<float> out_after(1, -1.f);

  Placeholder in(BufHandle("in", {8, 8}, kFloat));

  Tensor tensor = Reduce("sum", {}, Sum(), in, {{8, "m"}, {8, "n"}});

  LoopNest l_before({tensor});
  LoopNest l(l_before);
  l_before.prepareForCodegen();
  SimpleIREvaluator cg_before(l_before.root_stmt(), {in, tensor});
  cg_before.call({in_, out_before});

  ASSERT_FALSE(LoopNest::vectorize(l.getLoopStmtsFor(tensor)[1]));

  // But if we rfactor this so it's not a reduce axis we can vectorize that
  // loop.
  std::vector<ForPtr> loops = l.getLoopStmtsFor(tensor);
  LoopNest::reorderAxis(loops[0], loops[1]);
  loops = l.getLoopStmtsFor(tensor);
  auto tensor_body = l.getAllWritesToBuf(tensor.buf())[1];
  BufPtr rfac_buf = nullptr;
  ASSERT_TRUE(LoopNest::rfactor(tensor_body, loops.at(0), &rfac_buf));

  LoopNest::distributeLoop(loops.at(0));
  auto rfac_loops = l.getAllLoopNestsWritingToBuf(rfac_buf);

  ASSERT_TRUE(LoopNest::vectorize(rfac_loops[1][0]));
  l.simplify();

  StmtPtr s = l.root_stmt();

  std::ostringstream oss;
  oss << *s;
  const std::string& expected_ir =
      R"IR(
#CHECK: sum = 0.f;
#CHECK: for (int n = 0; n < 8; n++) {
#CHECK:   sum_rfac[n] = 0.f;
#CHECK: }
#CHECK: for (int m = 0; m < 8; m++) {
#CHECK:   sum_rfac[Ramp(0, 1, 8)] = ReduceOp((sum_rfac[Ramp(0, 1, 8)]) + (in[Ramp(8 * m, 1, 8)]), reduce_args={m});
#CHECK: }
#CHECK: for (int n = 0; n < 8; n++) {
#CHECK:   sum = ReduceOp((sum) + (sum_rfac[n]), reduce_args={n});
#CHECK: }
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());

  // Vectorizing should not change result.
  l.prepareForCodegen();
  s = IRSimplifier::simplify(l.root_stmt());
  SimpleIREvaluator cg_after(s, {in, tensor});
  cg_after.call({in_, out_after});

  ASSERT_EQ(out_before[0], out_after[0]);
}

TEST(Reductions, InitFunction) {
  KernelScope ks;
  constexpr int M = 32;
  constexpr int N = 16;
  Placeholder A("A", kFloat, {M, N});
  Placeholder B("B", kFloat, {N});
  Tensor C = Reduce(
      "C",
      {{N, "n"}},
      Sum(),
      [&](const std::vector<VarHandle>& v) { return B.load(v[0]); },
      [&](const std::vector<VarHandle>& v) { return A.load(v[1], v[0]); },
      {{M, "m"}});
  LoopNest nest({C});
  nest.prepareForCodegen();
  StmtPtr s = IRSimplifier::simplify(nest.root_stmt());
  std::ostringstream oss;
  oss << *s << "\n";
  const std::string& expected_ir =
      R"IR(
#CHECK:  for (int n = 0; n < 16; n++) {
#CHECK:    C[n] = B[n];
#CHECK:    for (int m = 0; m < 32; m++) {
#CHECK:      C[n] = (C[n]) + (A[n + 16 * m]);
#CHECK:    }
#CHECK:  }
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}
} // namespace jit
} // namespace torch
