#include <gtest/gtest.h>

#include "test/cpp/tensorexpr/test_base.h"

#include <c10/util/irange.h>
#include <torch/csrc/jit/tensorexpr/cpp_codegen.h>
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

#define STR_CHECK(node, expected) \
  std::stringstream ss;           \
  CppPrinter printer(&ss);        \
  printer.visit(node);            \
  ASSERT_EQ(ss.str(), expected)

#define FILE_CHECK(node, pattern) \
  std::stringstream ss;           \
  CppPrinter printer(&ss);        \
  printer.visit(node);            \
  torch::jit::testing::FileCheck().run(pattern, ss.str())

TEST(CppPrinter, IntImm) {
  auto i = alloc<IntImm>(10);
  STR_CHECK(i, "10");
}

TEST(CppPrinter, FloatImm) {
  auto f = alloc<FloatImm>(10);
  STR_CHECK(f, "10.f");
}

TEST(CppPrinter, FloatImm1) {
  auto f = alloc<FloatImm>(10);
  STR_CHECK(f, "10.f");
}

TEST(CppPrinter, DoubleImm) {
  auto d = alloc<DoubleImm>(10);
  STR_CHECK(d, "10.0");
}

TEST(CppPrinter, DoubleImm1) {
  auto d = alloc<DoubleImm>(10.1);
  STR_CHECK(d, "10.1");
}

TEST(CppPrinter, HalfImm) {
  auto h = alloc<HalfImm>(10);
  STR_CHECK(h, "10");
}

TEST(CppPrinter, Add) {
  auto add = alloc<Add>(alloc<IntImm>(1), alloc<IntImm>(2));
  STR_CHECK(add, "1 + 2");
}

TEST(CppPrinter, AddExpr1) {
  auto add = alloc<Add>(
      alloc<Add>(alloc<IntImm>(0), alloc<IntImm>(1)),
      alloc<Sub>(alloc<IntImm>(2), alloc<IntImm>(3)));
  STR_CHECK(add, "(0 + 1) + (2 - 3)");
}

TEST(CppPrinter, AddExpr2) {
  auto add = alloc<Add>(
      alloc<Mul>(alloc<IntImm>(0), alloc<IntImm>(1)),
      alloc<Sub>(alloc<IntImm>(2), alloc<IntImm>(3)));
  STR_CHECK(add, "0 * 1 + (2 - 3)");
}

TEST(CppPrinter, AddExpr3) {
  auto add = alloc<Add>(
      alloc<Add>(alloc<IntImm>(0), alloc<IntImm>(1)),
      alloc<Div>(alloc<IntImm>(2), alloc<IntImm>(3)));
  STR_CHECK(add, "(0 + 1) + 2 / 3");
}

TEST(CppPrinter, Mod) {
  auto mod = alloc<Mod>(alloc<IntImm>(1), alloc<IntImm>(2));
  STR_CHECK(mod, "1 % 2");
}

TEST(CppPrinter, ModFloat) {
  auto mod = alloc<Mod>(alloc<FloatImm>(1), alloc<FloatImm>(2));
  STR_CHECK(mod, "std::fmod(1.f, 2.f)");
}

TEST(CppPrinter, Max) {
  auto max = alloc<Max>(alloc<IntImm>(1), alloc<IntImm>(2), false);
  STR_CHECK(max, "std::max(1, 2)");
}

TEST(CppPrinter, MaxFloat) {
  auto max = alloc<Max>(alloc<FloatImm>(1), alloc<FloatImm>(2), false);
  STR_CHECK(max, "std::max(1.f, 2.f)");
}

TEST(CppPrinter, MaxHalf) {
  auto max = alloc<Max>(alloc<HalfImm>(1), alloc<HalfImm>(2), false);
  STR_CHECK(max, "(1 < 2) ? 2 : 1");
}

TEST(CppPrinter, And) {
  auto v = alloc<And>(alloc<IntImm>(1), alloc<IntImm>(2));
  STR_CHECK(v, "1 & 2");
}

TEST(CppPrinter, CompareSelect) {
  auto cs = alloc<CompareSelect>(
      alloc<IntImm>(1),
      alloc<IntImm>(2),
      alloc<FloatImm>(1),
      alloc<FloatImm>(2),
      CompareSelectOperation::kLE);
  STR_CHECK(cs, "((1 <= 2) ? 1.f : 2.f)");
}

TEST(CppPrinter, IfThenElse) {
  auto cond = alloc<Add>(alloc<IntImm>(1), alloc<IntImm>(2));
  auto true_value = alloc<Sub>(alloc<IntImm>(0), alloc<IntImm>(1));
  auto false_value = alloc<Mul>(alloc<IntImm>(2), alloc<IntImm>(3));
  auto v = alloc<IfThenElse>(cond, true_value, false_value);
  STR_CHECK(v, "((1 + 2) ? 0 - 1 : 2 * 3)");
}

TEST(CppPrinter, AllocateFree) {
  BufHandle buf("x", {2, 3}, kInt);
  AllocatePtr alloc = Allocate::make(buf);
  FreePtr free = Free::make(buf);
  BlockPtr block = Block::make({alloc, free});

  const std::string pattern = R"(
   # CHECK: {
   # CHECK:   int* x = static_cast<int*>(malloc(24));
   # CHECK:   free(x);
   # CHECK: }
  )";
  FILE_CHECK(block, pattern);
}

TEST(CppPrinter, LoadStore) {
  BufHandle a("A", {2, 3}, kInt);
  BufHandle b("B", {3, 4}, kInt);
  auto store = b.store({2, 2}, a.load(1, 1));
  STR_CHECK(
      store, "B[(0 + 2 * (1 * 4)) + 2 * 1] = A[(0 + 1 * (1 * 3)) + 1 * 1];\n");
}

TEST(CppPrinter, Var) {
  auto var = alloc<Var>("x", kInt);
  STR_CHECK(var, "x");
}

TEST(CppPrinter, Cast) {
  auto cast = alloc<Cast>(kFloat, alloc<IntImm>(1));
  STR_CHECK(cast, "static_cast<float>(1)");
}

TEST(CppPrinter, BitCast) {
  auto cast = alloc<BitCast>(kInt, alloc<FloatImm>(20));
  STR_CHECK(cast, "std::bitcast<float, int>(20.f)");
}

TEST(CppPrinter, Let) {
  auto var = alloc<Var>("x", kFloat);
  auto val = alloc<FloatImm>(2);
  auto let = alloc<Let>(var, val);
  STR_CHECK(let, "float x = 2.f;\n");
}

TEST(CppPrinter, For) {
  constexpr int N = 1024;
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kInt);
  VarHandle i("i", kInt);
  auto f = For::make(i, 0, N, c.store({i}, Add::make(a.load(i), b.load(i))));
  const std::string pattern = R"(
   # CHECK: for (int i = 0; i < 1024; i++) {
   # CHECK:   C[i] = (A[i]) + (B[i]);
   # CHECK: }
  )";
  FILE_CHECK(f, pattern);
}

TEST(CppPrinter, Cond) {
  BufHandle x("X", {1}, kInt);
  auto cmp = CompareSelect::make(x.load(0), 10, CompareSelectOperation::kLT);
  auto cond =
      Cond::make(cmp, x.store({0}, x.load(0) + 1), x.store({0}, x.load(0) - 1));
  const std::string pattern = R"(
    # CHECK: if (((X[0] < 10) ? 1 : 0)) {
    # CHECK:   X[0] = (X[0]) + 1;
    # CHECK: } else {
    # CHECK:   X[0] = (X[0]) - 1;
    # CHECK: }
  )";
  FILE_CHECK(cond, pattern);
}

TEST(CppPrinter, Intrinsics) {
  const std::unordered_set<IntrinsicsOp, std::hash<int>> unsupported_ops{
      kRand, kSigmoid};
  for (const auto i : c10::irange(static_cast<uint32_t>(kMaxIntrinsicsOp))) {
    IntrinsicsOp op = static_cast<IntrinsicsOp>(i);
    if (unsupported_ops.count(op)) {
      continue;
    }

    if (Intrinsics::OpArgCount(op) == 1) {
      auto v = alloc<Intrinsics>(op, alloc<FloatImm>(2.0f));
      STR_CHECK(v, "std::" + v->func_name() + "(2.f)");
    } else {
      auto v =
          alloc<Intrinsics>(op, alloc<FloatImm>(1.0f), alloc<FloatImm>(2.0f));
      STR_CHECK(v, "std::" + v->func_name() + "(1.f, 2.f)");
    }
  }
}

TEST(CppPrinter, ExternalCall) {
  std::vector<ExprPtr> dims{alloc<IntImm>(2), alloc<IntImm>(2)};
  auto output = alloc<Buf>("out", dims, kFloat);
  auto buf_arg1 = alloc<Buf>("a", dims, kFloat);
  auto buf_arg2 = alloc<Buf>("b", dims, kFloat);
  auto scalar_arg = alloc<Add>(alloc<IntImm>(1), alloc<IntImm>(2));
  std::vector<BufPtr> buf_args{buf_arg1, buf_arg2};
  std::vector<ExprPtr> scalar_args{scalar_arg};
  auto call =
      alloc<ExternalCall>(output, "nnc_aten_matmul", buf_args, scalar_args);
  const std::string pattern = R"(
   # CHECK: {
   # CHECK:   void* buf_ptrs[]{out, a, b};
   # CHECK:   int64_t buf_ranks[]{2, 2, 2};
   # CHECK:   int64_t buf_dims[]{2, 2, 2, 2, 2, 2};
   # CHECK:   int8_t buf_dtypes[]{6, 6, 6};
   # CHECK:   int64_t extra_args[]{1 + 2};
   # CHECK:   nnc_aten_matmul(
   # CHECK:       3,
   # CHECK:       buf_ptrs,
   # CHECK:       buf_ranks,
   # CHECK:       buf_dims,
   # CHECK:       buf_dtypes,
   # CHECK:       1,
   # CHECK:       extra_args);
   # CHECK: }
  )";
  FILE_CHECK(call, pattern);
}

} // namespace jit
} // namespace torch
