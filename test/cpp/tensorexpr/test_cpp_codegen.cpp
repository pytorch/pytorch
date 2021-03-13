#include "test/cpp/tensorexpr/test_base.h"

#include <torch/csrc/jit/tensorexpr/cpp_codegen.h>
#include <torch/csrc/jit/tensorexpr/mem_arena.h>
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
  KernelScope kernel_scope;
  auto i = new IntImm(10);
  STR_CHECK(i, "10");
}

TEST(CppPrinter, FloatImm) {
  KernelScope kernel_scope;
  auto f = new FloatImm(10);
  STR_CHECK(f, "10.f");
}

TEST(CppPrinter, FloatImm1) {
  KernelScope kernel_scope;
  auto f = new FloatImm(10);
  STR_CHECK(f, "10.f");
}

TEST(CppPrinter, DoubleImm) {
  KernelScope kernel_scope;
  auto d = new DoubleImm(10);
  STR_CHECK(d, "10.0");
}

TEST(CppPrinter, DoubleImm1) {
  KernelScope kernel_scope;
  auto d = new DoubleImm(10.1);
  STR_CHECK(d, "10.1");
}

TEST(CppPrinter, HalfImm) {
  KernelScope kernel_scope;
  auto h = new HalfImm(10);
  STR_CHECK(h, "10");
}

TEST(CppPrinter, Add) {
  KernelScope kernel_scope;
  auto add = new Add(new IntImm(1), new IntImm(2));
  STR_CHECK(add, "1 + 2");
}

TEST(CppPrinter, AddExpr1) {
  KernelScope kernel_scope;
  Add* add = new Add(
      new Add(new IntImm(0), new IntImm(1)),
      new Sub(new IntImm(2), new IntImm(3)));
  STR_CHECK(add, "(0 + 1) + (2 - 3)");
}

TEST(CppPrinter, AddExpr2) {
  KernelScope kernel_scope;
  Add* add = new Add(
      new Mul(new IntImm(0), new IntImm(1)),
      new Sub(new IntImm(2), new IntImm(3)));
  STR_CHECK(add, "0 * 1 + (2 - 3)");
}

TEST(CppPrinter, AddExpr3) {
  KernelScope kernel_scope;
  Add* add = new Add(
      new Add(new IntImm(0), new IntImm(1)),
      new Div(new IntImm(2), new IntImm(3)));
  STR_CHECK(add, "(0 + 1) + 2 / 3");
}

TEST(CppPrinter, Mod) {
  KernelScope kernel_scope;
  auto mod = new Mod(new IntImm(1), new IntImm(2));
  STR_CHECK(mod, "1 % 2");
}

TEST(CppPrinter, ModFloat) {
  KernelScope kernel_scope;
  auto mod = new Mod(new FloatImm(1), new FloatImm(2));
  STR_CHECK(mod, "std::fmod(1.f, 2.f)");
}

TEST(CppPrinter, Max) {
  KernelScope kernel_scope;
  auto max = new Max(new IntImm(1), new IntImm(2), false);
  STR_CHECK(max, "std::max(1, 2)");
}

TEST(CppPrinter, MaxFloat) {
  KernelScope kernel_scope;
  auto max = new Max(new FloatImm(1), new FloatImm(2), false);
  STR_CHECK(max, "std::max(1.f, 2.f)");
}

TEST(CppPrinter, MaxHalf) {
  KernelScope kernel_scope;
  auto max = new Max(new HalfImm(1), new HalfImm(2), false);
  STR_CHECK(max, "(1 < 2) ? 2 : 1");
}

TEST(CppPrinter, And) {
  KernelScope kernel_scope;
  auto v = new And(new IntImm(1), new IntImm(2));
  STR_CHECK(v, "1 & 2");
}

TEST(CppPrinter, CompareSelect) {
  KernelScope kernel_scope;
  auto cs = new CompareSelect(
      new IntImm(1),
      new IntImm(2),
      new FloatImm(1),
      new FloatImm(2),
      CompareSelectOperation::kLE);
  STR_CHECK(cs, "((1 <= 2) ? 1.f : 2.f)");
}

TEST(CppPrinter, IfThenElse) {
  KernelScope kernel_scope;
  auto cond = new Add(new IntImm(1), new IntImm(2));
  auto true_value = new Sub(new IntImm(0), new IntImm(1));
  auto false_value = new Mul(new IntImm(2), new IntImm(3));
  auto v = new IfThenElse(cond, true_value, false_value);
  STR_CHECK(v, "((1 + 2) ? 0 - 1 : 2 * 3)");
}

TEST(CppPrinter, AllocateFree) {
  KernelScope kernel_scope;
  BufHandle buf("x", {2, 3}, kInt);
  Allocate* alloc = Allocate::make(buf);
  Free* free = Free::make(buf);
  Block* block = Block::make({alloc, free});

  const std::string pattern = R"(
   # CHECK: {
   # CHECK:   int* x = static_cast<int*>(malloc(24));
   # CHECK:   free(x);
   # CHECK: }
  )";
  FILE_CHECK(block, pattern);
}

TEST(CppPrinter, LoadStore) {
  KernelScope kernel_scope;
  Placeholder a(BufHandle("A", {2, 3}, kInt));
  Placeholder b(BufHandle("B", {3, 4}, kInt));
  auto store = b.store({2, 2}, a.load(1, 1));
  STR_CHECK(
      store, "B[(0 + 2 * (1 * 4)) + 2 * 1] = A[(0 + 1 * (1 * 3)) + 1 * 1];\n");
}

TEST(CppPrinter, Var) {
  KernelScope kernel_scope;
  auto var = new Var("x", kInt);
  STR_CHECK(var, "x");
}

TEST(CppPrinter, Cast) {
  KernelScope kernel_scope;
  auto cast = new Cast(kFloat, new IntImm(1));
  STR_CHECK(cast, "static_cast<float>(1)");
}

TEST(CppPrinter, BitCast) {
  KernelScope kernel_scope;
  auto cast = new BitCast(kInt, new FloatImm(20));
  STR_CHECK(cast, "std::bitcast<float, int>(20.f)");
}

TEST(CppPrinter, Let) {
  KernelScope kernel_scope;
  auto var = new Var("x", kFloat);
  auto val = new FloatImm(2);
  auto let = new Let(var, val);
  STR_CHECK(let, "float x = 2.f;\n");
}

TEST(CppPrinter, For) {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Placeholder a(BufHandle("A", {N}, kInt));
  Placeholder b(BufHandle("B", {N}, kInt));
  Placeholder c(BufHandle("C", {N}, kInt));
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
  KernelScope kernel_scope;
  Placeholder x(BufHandle("X", {1}, kInt));
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
  KernelScope kernel_scope;
  const std::unordered_set<IntrinsicsOp, std::hash<int>> unsupported_ops{
      kRand, kSigmoid};
  for (int i = 0; i < kMaxIntrinsicsOp; i++) {
    IntrinsicsOp op = static_cast<IntrinsicsOp>(i);
    if (unsupported_ops.count(op)) {
      continue;
    }

    if (Intrinsics::OpArgCount(op) == 1) {
      auto v = new Intrinsics(op, new FloatImm(2.0f));
      STR_CHECK(v, "std::" + v->func_name() + "(2.f)");
    } else {
      auto v = new Intrinsics(op, new FloatImm(1.0f), new FloatImm(2.0f));
      STR_CHECK(v, "std::" + v->func_name() + "(1.f, 2.f)");
    }
  }
}

TEST(CppPrinter, ExternalCall) {
  KernelScope kernel_scope;
  Buf* output = new Buf("out", {new IntImm(2), new IntImm(2)}, kFloat);
  Buf* buf_arg1 = new Buf("a", {new IntImm(2), new IntImm(2)}, kFloat);
  Buf* buf_arg2 = new Buf("b", {new IntImm(2), new IntImm(2)}, kFloat);
  Expr* scalar_arg = new Add(new IntImm(1), new IntImm(2));
  auto call = new ExternalCall(
      output, "nnc_aten_matmul", {buf_arg1, buf_arg2}, {scalar_arg});
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

TEST(CppPrinter, LoadStoreVecWithMask) {
  KernelScope kernel_scope;
  Placeholder a(BufHandle("A", {3}, kInt));
  Placeholder b(BufHandle("B", {3}, kInt));
  auto store = b.storeWithMask(
      {Ramp::make(0, 1, 3)},
      a.loadWithMask(
          {Ramp::make(0, 1, 3)}, Broadcast::make(IntImm::make(1), 3)),
      Broadcast::make(IntImm::make(1), 3));
  const std::string pattern = R"(
   # CHECK: B[0 + 0 * 1] = A[0 + 0 * 1];
   # CHECK: B[0 + 1 * 1] = A[0 + 1 * 1];
   # CHECK: B[0 + 2 * 1] = A[0 + 2 * 1];
  )";
  FILE_CHECK(store, pattern);
}

} // namespace jit
} // namespace torch
