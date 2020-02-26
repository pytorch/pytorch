#ifdef ENABLE_LLVM
#include "test/cpp/tensorexpr/test_base.h"

#include "test/cpp/tensorexpr/padded_buffer.h"
#include "test/cpp/tensorexpr/test_utils.h"
#include "torch/csrc/jit/tensorexpr/buffer.h"
#include "torch/csrc/jit/tensorexpr/eval.h"
#include "torch/csrc/jit/tensorexpr/function.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "torch/csrc/jit/tensorexpr/llvm_codegen.h"
#include "torch/csrc/jit/tensorexpr/schedule.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

#include <numeric>

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;
using namespace torch::jit::tensorexpr::schedule;

using LLVMExprEval = ExprEval<LLVMCodeGen>;

void testLLVMIntImmTest() {
  KernelScope kernel_scope;
  auto a = IntImm::make(2);
  LLVMExprEval cg(a);
  EXPECT_EQ(cg.value<int>(), 2);
}

void testLLVMFloatImmTest() {
  KernelScope kernel_scope;
  auto a = FloatImm::make(1.0);
  LLVMExprEval cg(a, {});
  EXPECT_EQ(cg.value<float>(), 1.0);
}

void testLLVMIntAddTest() {
  KernelScope kernel_scope;
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Add::make(a, b);
  LLVMExprEval cg(c);
  EXPECT_EQ(cg.value<int>(), 5);
}

void testLLVMIntSubTest() {
  KernelScope kernel_scope;
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Sub::make(a, b);
  LLVMExprEval cg(c);
  EXPECT_EQ(cg.value<int>(), -1);
}

void testLLVMIntMulTest() {
  KernelScope kernel_scope;
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Mul::make(a, b);
  LLVMExprEval cg(c);
  EXPECT_EQ(cg.value<int>(), 6);
}

void testLLVMIntDivTest() {
  KernelScope kernel_scope;
  auto a = IntImm::make(6);
  auto b = IntImm::make(3);
  auto c = Div::make(a, b);
  LLVMExprEval cg(c);
  EXPECT_EQ(cg.value<int>(), 2);
}

void testLLVMIntToFloatCastTest() {
  KernelScope kernel_scope;
  auto a = IntImm::make(2);
  auto b = Cast::make(kFloat32, a);
  LLVMExprEval cg(b, {});
  EXPECT_EQ(cg.value<float>(), 2.0);
}

void testLLVMFloatToIntCastTest() {
  KernelScope kernel_scope;
  auto a = FloatImm::make(2.0);
  auto b = Cast::make(kInt32, a);
  LLVMExprEval cg(b);
  EXPECT_EQ(cg.value<int>(), 2);
}

void testLLVMLetTest01() {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat32);
  ExprHandle value = ExprHandle(3.f);
  ExprHandle body = ExprHandle(2.f) + (x * ExprHandle(3.f) + ExprHandle(4.f));
  ExprHandle result = Let::make(x, ExprHandle(3.f), body);
  LLVMExprEval cg(result, {});
  EXPECT_EQ(cg.value<float>(), 2.f + (3.f * 3.f + 4.f));
}

void testLLVMLetTest02() {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat32);
  VarHandle y("y", kFloat32);
  ExprHandle value = ExprHandle(3.f);
  ExprHandle body = ExprHandle(2.f) + (x * ExprHandle(3.f) + ExprHandle(4.f) * y);
  ExprHandle e1 = Let::make(x, ExprHandle(3.f), body);
  ExprHandle e2 = Let::make(y, ExprHandle(6.f), e1);
  LLVMExprEval cg(e2, {});
  EXPECT_EQ(cg.value<float>(), 2.f + (3.f * 3.f + 4.f * 6.f));
}

void testLLVMBufferTest() {
  KernelScope kernel_scope;
  Buffer a(VarHandle("A", kHandle), kFloat32, {32});
  std::vector<int32_t> v(5);
  std::vector<void*> args({v.data()});
  auto rv = IntImm::make(0);
  LLVMExprEval cg(rv, {a});
  EXPECT_EQ(cg.value<int>(args), 0);
}

void testLLVMBlockTest() {
  KernelScope kernel_scope;
  Buffer a(VarHandle("A", kHandle), kInt32, {32});
  std::vector<int32_t> v = {1, 2};
  std::vector<void*> args({v.data()});

  auto block = Block::make({
      Store::make(a, IntImm::make(0), IntImm::make(3), IntImm::make(1)),
      Store::make(a, IntImm::make(1), IntImm::make(4), IntImm::make(1)),
      Store::make(a, IntImm::make(0), IntImm::make(4), IntImm::make(1)),
  });

  LLVMCodeGen cg(block, {a});
  EXPECT_EQ(cg.value<int>(args), 0);
  EXPECT_EQ(v[0], 4);
  EXPECT_EQ(v[1], 4);
}

void testLLVMLoadStoreTest() {
  KernelScope kernel_scope;
  Buffer a(VarHandle("A", kHandle), kInt32, {1});
  Buffer b(VarHandle("B", kHandle), kInt32, {1});
  std::vector<int32_t> a_buffer = {42};
  std::vector<int32_t> b_buffer = {-11};

  auto store = Store::make(
      b,
      IntImm::make(0),
      Load::make(a, IntImm::make(0), IntImm::make(1)),
      IntImm::make(1));
  LLVMCodeGen cg(store, {a, b});
  std::vector<void*> args({a_buffer.data(), b_buffer.data()});
  EXPECT_EQ(cg.value<int>(args), 0);
  EXPECT_EQ(a_buffer[0], 42);
  EXPECT_EQ(b_buffer[0], 42);
}

void testLLVMIfThenElseTest() {
  KernelScope kernel_scope;
  Buffer a(VarHandle("A", kHandle), kInt32, {1});
  Buffer b(VarHandle("B", kHandle), kInt32, {1});
  Buffer c(VarHandle("C", kHandle), kInt32, {1});
  std::vector<int32_t> a_buffer = {42};
  std::vector<int32_t> b_buffer = {-11};
  std::vector<int32_t> c_buffer = {1};

  auto store = Store::make(
      b,
      IntImm::make(0),
      IfThenElse::make(
          Load::make(c, IntImm::make(0), IntImm::make(1)), // cond
          Load::make(a, IntImm::make(0), IntImm::make(1)), // then
          IntImm::make(0)), // else
      IntImm::make(1));
  LLVMCodeGen cg(store, {a, b, c});
  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  EXPECT_EQ(cg.value<int>(args), 0);
  EXPECT_EQ(a_buffer[0], 42);
  EXPECT_EQ(b_buffer[0], 42);
}

void testLLVMVecLoadStoreTest() {
  KernelScope kernel_scope;
  Buffer a(VarHandle("A", kHandle), kInt32, {1});
  Buffer b(VarHandle("B", kHandle), kInt32, {1});
  std::vector<int32_t> a_buffer = {1, 1, 1, 1};
  std::vector<int32_t> b_buffer = {2, 2, 2, 2};

  auto store = Store::make(
      b,
      Ramp::make(0, 1, 4),
      Load::make(a, Ramp::make(0, 1, 4), Broadcast::make(IntImm::make(1), 4)),
      Broadcast::make(IntImm::make(1), 4));
  LLVMCodeGen cg(store, {a, b});
  std::vector<void*> args({a_buffer.data(), b_buffer.data()});
  EXPECT_EQ(cg.value<int>(args), 0);
  EXPECT_EQ(a_buffer[0], 1);
  EXPECT_EQ(a_buffer[1], 1);
  EXPECT_EQ(a_buffer[2], 1);
  EXPECT_EQ(a_buffer[3], 1);
  EXPECT_EQ(b_buffer[0], 1);
  EXPECT_EQ(b_buffer[1], 1);
  EXPECT_EQ(b_buffer[2], 1);
  EXPECT_EQ(b_buffer[3], 1);
}

void testLLVMMemcpyTest() {
  KernelScope kernel_scope;
  constexpr int N = 32;
  Buffer a(VarHandle("A", kHandle), kInt32, {N});
  Buffer b(VarHandle("B", kHandle), kInt32, {N});
  std::vector<int32_t> a_buffer(N, 42);
  std::vector<int32_t> b_buffer(N, 0);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto expr =
      For::make(i, 0, N, Store::make(b, i, Load::make(a, i, mask), mask));

  LLVMCodeGen cg(expr, {a, b});

  std::vector<void*> args({a_buffer.data(), b_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  assertAllEqual(a_buffer, 42);
  assertAllEqual(b_buffer, 42);
}

void testLLVMBzeroTest() {
  KernelScope kernel_scope;
  constexpr int N = 32;
  Buffer b(VarHandle("B", kHandle), kInt32, {N});
  std::vector<int32_t> b_buffer(N, 11);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto expr = For::make(i, 0, N, Store::make(b, i, IntImm::make(0), mask));

  LLVMCodeGen cg(expr, {b});

  std::vector<void*> args({b_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(b_buffer.size(), N);
  assertAllEqual(b_buffer, 0);
}

void testLLVMElemwiseAdd() {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Buffer a(VarHandle("A", kHandle), kInt32, {N});
  Buffer b(VarHandle("B", kHandle), kInt32, {N});
  Buffer c(VarHandle("C", kHandle), kInt32, {N});
  std::vector<int32_t> a_buffer(N, 41);
  std::vector<int32_t> b_buffer(N, 1);
  std::vector<int32_t> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Add::make(Load::make(a, i, mask), Load::make(b, i, mask)),
          mask));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41);
  assertAllEqual(b_buffer, 1);
  assertAllEqual(c_buffer, 42);
}

void testLLVMElemwiseAddFloat() {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Buffer a(VarHandle("A", kHandle), kFloat32, {N});
  Buffer b(VarHandle("B", kHandle), kFloat32, {N});
  Buffer c(VarHandle("C", kHandle), kFloat32, {N});
  std::vector<float> a_buffer(N, 41);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto expr = For::make(
      i,
      0,
      N,
      Store::make(c, i, Load::make(a, i, mask) + Load::make(b, i, mask), mask));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41.0f);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 42.0f);
}

void testLLVMElemwiseLog10Float() {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Buffer a(VarHandle("A", kHandle), kFloat32, {N});
  Buffer b(VarHandle("B", kHandle), kFloat32, {N});
  std::vector<float> a_buffer(N, 10.0f);
  std::vector<float> b_buffer(N, 2.0f);

  auto mask = Broadcast::make(IntImm::make(1), 4);
  VarHandle i("i", kInt32);
  auto expr = For::make(
      i,
      0,
      N / 4,
      Store::make(
          b,
          Ramp::make(i * 4, 1, 4),
          log10(Load::make(a, Ramp::make(i * 4, 1, 4), mask)),
          mask));

  LLVMCodeGen cg(expr, {a, b});

  std::vector<void*> args({a_buffer.data(), b_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  assertAllEqual(a_buffer, 10.0f);
  assertAllEqual(b_buffer, 1.0f);
}

void testLLVMElemwiseMaxInt() {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Buffer a(VarHandle("A", kHandle), kInt32, {N});
  Buffer b(VarHandle("B", kHandle), kInt32, {N});
  Buffer c(VarHandle("C", kHandle), kInt32, {N});
  std::vector<int> a_buffer(N, 41);
  std::vector<int> b_buffer(N, 1);
  std::vector<int> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Max::make(Load::make(a, i, mask), Load::make(b, i, mask), false),
          mask));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41);
  assertAllEqual(b_buffer, 1);
  assertAllEqual(c_buffer, 41);
}

void testLLVMElemwiseMinInt() {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Buffer a(VarHandle("A", kHandle), kInt32, {N});
  Buffer b(VarHandle("B", kHandle), kInt32, {N});
  Buffer c(VarHandle("C", kHandle), kInt32, {N});
  std::vector<int> a_buffer(N, 41);
  std::vector<int> b_buffer(N, 1);
  std::vector<int> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Min::make(Load::make(a, i, mask), Load::make(b, i, mask), false),
          mask));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41);
  assertAllEqual(b_buffer, 1);
  assertAllEqual(c_buffer, 1);
}

void testLLVMElemwiseMaxNumFloat() {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Buffer a(VarHandle("A", kHandle), kFloat32, {N});
  Buffer b(VarHandle("B", kHandle), kFloat32, {N});
  Buffer c(VarHandle("C", kHandle), kFloat32, {N});
  std::vector<float> a_buffer(N, 41);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Max::make(Load::make(a, i, mask), Load::make(b, i, mask), false),
          mask));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41.0f);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 41.0f);
}

void testLLVMElemwiseMaxNumNaNFloat() {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Buffer a(VarHandle("A", kHandle), kFloat32, {N});
  Buffer b(VarHandle("B", kHandle), kFloat32, {N});
  Buffer c(VarHandle("C", kHandle), kFloat32, {N});
  std::vector<float> a_buffer(N, NAN);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Max::make(Load::make(a, i, mask), Load::make(b, i, mask), false),
          mask));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 1.0f);
}

void testLLVMElemwiseMinNumFloat() {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Buffer a(VarHandle("A", kHandle), kFloat32, {N});
  Buffer b(VarHandle("B", kHandle), kFloat32, {N});
  Buffer c(VarHandle("C", kHandle), kFloat32, {N});
  std::vector<float> a_buffer(N, 41);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Min::make(Load::make(a, i, mask), Load::make(b, i, mask), false),
          mask));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41.0f);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 1.0f);
}

void testLLVMElemwiseMinNumNaNFloat() {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Buffer a(VarHandle("A", kHandle), kFloat32, {N});
  Buffer b(VarHandle("B", kHandle), kFloat32, {N});
  Buffer c(VarHandle("C", kHandle), kFloat32, {N});
  std::vector<float> a_buffer(N, NAN);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Min::make(Load::make(a, i, mask), Load::make(b, i, mask), false),
          mask));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 1.0f);
}

#if 1 // LLVM doesn't currently have implementations for maximum/minimum on x86
void testLLVMElemwiseMaximumFloat() {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Buffer a(VarHandle("A", kHandle), kFloat32, {N});
  Buffer b(VarHandle("B", kHandle), kFloat32, {N});
  Buffer c(VarHandle("C", kHandle), kFloat32, {N});
  std::vector<float> a_buffer(N, 41);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Max::make(Load::make(a, i, mask), Load::make(b, i, mask), true),
          mask));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41.0f);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 41.0f);
}

void testLLVMElemwiseMaximumNaNFloat() {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Buffer a(VarHandle("A", kHandle), kFloat32, {N});
  Buffer b(VarHandle("B", kHandle), kFloat32, {N});
  Buffer c(VarHandle("C", kHandle), kFloat32, {N});
  std::vector<float> a_buffer(N, NAN);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Max::make(Load::make(a, i, mask), Load::make(b, i, mask), true),
          mask));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  for (int i = 0; i < N; ++i) {
    ASSERT_TRUE(std::isnan(a_buffer[i]));
    ASSERT_TRUE(std::isnan(c_buffer[i]));
  }
}

void testLLVMElemwiseMinimumFloat() {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Buffer a(VarHandle("A", kHandle), kFloat32, {N});
  Buffer b(VarHandle("B", kHandle), kFloat32, {N});
  Buffer c(VarHandle("C", kHandle), kFloat32, {N});
  std::vector<float> a_buffer(N, 41);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Min::make(Load::make(a, i, mask), Load::make(b, i, mask), true),
          mask));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41.0f);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 1.0f);
}

void testLLVMElemwiseMinimumNaNFloat() {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Buffer a(VarHandle("A", kHandle), kFloat32, {N});
  Buffer b(VarHandle("B", kHandle), kFloat32, {N});
  Buffer c(VarHandle("C", kHandle), kFloat32, {N});
  std::vector<float> a_buffer(N, NAN);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Min::make(Load::make(a, i, mask), Load::make(b, i, mask), true),
          mask));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  for (int i = 0; i < N; ++i) {
    ASSERT_TRUE(std::isnan(a_buffer[i]));
    ASSERT_TRUE(std::isnan(c_buffer[i]));
  }
}
#endif

void testLLVMCompareSelectIntEQ() {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Buffer a(VarHandle("A", kHandle), kInt32, {N});
  Buffer b(VarHandle("B", kHandle), kInt32, {N});
  Buffer c(VarHandle("C", kHandle), kInt32, {N});
  std::vector<int> a_buffer(N, 1);
  std::vector<int> b_buffer(N, 1);
  std::vector<int> c_buffer(N, 0);
  std::vector<int> c_ref(N, 1);

  for (int i = 0; i < N / 2; i++) {
    b_buffer[i] = 0;
    c_ref[i] = 0;
  }

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          CompareSelect::make(
              Load::make(a, i, mask),
              Load::make(b, i, mask),
              CompareSelectOperation::kEQ),
          mask));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);

  assertAllEqual(a_buffer, 1);
  for (int i = 0; i < N; i++) {
    ASSERT_EQ(c_ref[i], c_buffer[i]);
  }
}

void testLLVMCompareSelectFloatEQ() {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Buffer a(VarHandle("A", kHandle), kFloat32, {N});
  Buffer b(VarHandle("B", kHandle), kFloat32, {N});
  Buffer c(VarHandle("C", kHandle), kInt32, {N});
  std::vector<float> a_buffer(N, 1.0f);
  std::vector<float> b_buffer(N, 1.0f);
  std::vector<int> c_buffer(N, 0);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          CompareSelect::make(
              Load::make(a, i, mask),
              Load::make(b, i, mask),
              CompareSelectOperation::kEQ),
          mask));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);

  assertAllEqual(a_buffer, 1.0f);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 1);
}

void testLLVMStoreFloat() {
  KernelScope kernel_scope;
  Buffer result(VarHandle("result", kHandle), kFloat32, {1});
  std::vector<float> result_buffer = {0.0f};
  auto expr = Store::make(
      result, IntImm::make(0), FloatImm::make(3.14f), IntImm::make(1));
  LLVMCodeGen cg(expr, {result});
  std::vector<void*> args({result_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);
  EXPECT_EQ(result_buffer[0], 3.14f);
}

void testLLVMSimpleMath01() {
  KernelScope kernel_scope;
  const int N = 1024;
  Tensor* tensor = Compute(
      "f", {{N, "i"}}, [](const VarHandle& i) { return cast<float>(i * i + 1); });
  Schedule sch = Schedule::make({tensor});
  Stmt* stmt = sch.Lower();
  Buffer f_buf(VarHandle(tensor->function()->func_var()), kFloat32, {N});
  LLVMCodeGen cg(stmt, {f_buf});

  PaddedBuffer<float> f_v(N, "f_v");
  std::vector<void*> args({f_v.data()});
  int value = cg.value<int>(args);
  ASSERT_EQ(value, 0);
  PaddedBuffer<float> f_ref(N, "f_ref");
  for (int i = 0; i < N; i++) {
    f_ref(i) = i * i + 1;
  }
  ExpectAllNear(f_v, f_ref, 1e-5);
}

void testLLVMComputeMul() {
  KernelScope kernel_scope;
  const int N = 1024;
  Buffer a(VarHandle("a", kHandle), kFloat32, {N});
  Buffer b(VarHandle("b", kHandle), kFloat32, {N});
  Tensor* c = Compute("c", {{N, "i"}}, [&](const VarHandle& i) {
    return Load::make(a, i, 1) * Load::make(b, i, 1);
  });

  Buffer c_buf(VarHandle(c->function()->func_var()), kFloat32, {N});
  Schedule sch = Schedule::make({c});
  Stmt* s = sch.Lower();

  LLVMCodeGen cg(s, {a, b, c_buf});

  std::vector<float> a_vec(N, 21.0f);
  std::vector<float> b_vec(N, 2.0f);
  std::vector<float> c_vec(N, 0.0f);
  std::vector<void*> args({a_vec.data(), b_vec.data(), c_vec.data()});
  ASSERT_EQ(cg.value<int>(args), 0);
  assertAllEqual(c_vec, 42.0f);
}

void testLLVMBroadcastAdd() {
  KernelScope kernel_scope;
  const int M = 32;
  const int N = 1024;
  Buffer a(VarHandle("a", kHandle), kFloat32, {M, N});
  Buffer b(VarHandle("b", kHandle), kFloat32, {N});
  Tensor* c =
      Compute("c", {{M, "i"}, {N, "j"}}, [&](const VarHandle& i, const VarHandle& j) {
        ExprHandle mask(1);
        return Load::make(a, i * N + j, mask) + Load::make(b, j, mask);
      });

  Buffer c_buf(VarHandle(c->function()->func_var()), kFloat32, {M, N});
  Schedule sch = Schedule::make({c});
  Stmt* s = sch.Lower();

  LLVMCodeGen cg(s, {a, b, c_buf});

  std::vector<float> av(M * N);
  std::iota(av.begin(), av.end(), 0);
  std::vector<float> bv(N);
  std::iota(bv.begin(), bv.end(), 0);
  std::vector<float> cv(M * N, 0);
  std::vector<void*> args({av.data(), bv.data(), cv.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      ASSERT_EQ(cv[i * N + j], av[i * N + j] + bv[j]);
    }
  }
}

void testLLVMDynamicShapeAdd() {
  KernelScope kernel_scope;
  auto testWithSize = [](int32_t size) {
    VarHandle n("n", kInt32);
    Buffer a(VarHandle("a", kHandle), kFloat32, {n});
    Buffer b(VarHandle("b", kHandle), kFloat32, {n});
    Buffer c(VarHandle("c", kHandle), kFloat32, {n});
    VarHandle i("i", kInt32);
    Stmt* s = For::make(i, 0, n, Store::make(c, i, a(i) + b(i), 1));
    std::vector<float> aData(size, 1.0f);
    std::vector<float> bData(size, 2.0f);
    std::vector<float> cData(size, 0.0f);
    LLVMCodeGen cg(s, {a, b, c, n});
    std::vector<void*> args({aData.data(), bData.data(), cData.data(), &size});
    cg.value<float>(args);
    ExpectAllNear(cData, std::vector<float>(size, 3.0f), 1e-7);
  };
  testWithSize(1);
  testWithSize(16);
  testWithSize(37);
}

void testLLVMBindDynamicShapeAdd() {
  KernelScope kernel_scope;
  auto testWithSize = [](int32_t size) {
    VarHandle n("n", kInt32);
    Buffer a(VarHandle("a", kHandle), kFloat32, {n});
    Buffer b(VarHandle("b", kHandle), kFloat32, {n});
    Buffer c(VarHandle("c", kHandle), kFloat32, {n});
    VarHandle i("i", kInt32);
    Stmt* s = For::make(i, 0, n, Store::make(c, i, a(i) + b(i), 1));
    std::vector<float> aData(size, 1.0f);
    std::vector<float> bData(size, 2.0f);
    std::vector<float> cData(size, 0.0f);
    LLVMCodeGen cg(s, {a, b, c, n});
    cg.call({aData, bData, cData, size});
    ExpectAllNear(cData, std::vector<float>(size, 3.0f), 1e-7);
  };
  testWithSize(1);
  testWithSize(16);
  testWithSize(37);
}

void testLLVMTensorDynamicShapeAdd() {
  KernelScope kernel_scope;
  auto testWithSize = [](int32_t size) {
    VarHandle n("n", kInt32);
    Buffer a(VarHandle("a", kHandle), kFloat32, {n});
    Buffer b(VarHandle("b", kHandle), kFloat32, {n});
    Tensor* c =
        Compute("c", {{n, "n"}}, [&](const VarHandle& i) { return a(i) + b(i); });
    Schedule sch = Schedule::make({c});
    Stmt* s = sch.Lower();
    LLVMCodeGen cg(s, {a, b, c, n});
    std::vector<float> aData(size, 1.0f);
    std::vector<float> bData(size, 2.0f);
    std::vector<float> cData(size, 0.0f);
    cg.call({aData, bData, cData, size});
    ExpectAllNear(cData, std::vector<float>(size, 3.0f), 1e-7);
  };
  testWithSize(1);
  testWithSize(16);
  testWithSize(37);
}

void testLLVMDynamicShape2D() {
  KernelScope kernel_scope;
  auto testWithSize = [](int32_t M, int32_t N) {
    VarHandle m("m", kInt32);
    VarHandle n("n", kInt32);
    Buffer a(VarHandle("a", kHandle), kFloat32, {m, n});
    Buffer b(VarHandle("b", kHandle), kFloat32, {m, n});
    Tensor* c =
        Compute("c", {{m, "m"}, {n, "n"}}, [&](const VarHandle& i, const VarHandle& j) {
          return a(i, j) + b(i, j);
        });
    auto sch = torch::jit::tensorexpr::schedule::Schedule::make({c});
    Stmt* s = sch.Lower();
    LLVMCodeGen cg(s, {a, b, c, m, n});
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

} // namespace jit
} // namespace torch

#endif // ENABLE_LLVM
