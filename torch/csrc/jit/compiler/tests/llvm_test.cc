#include "torch/csrc/jit/compiler/include/ir.h"
#include "torch/csrc/jit/compiler/include/ir_printer.h"
#include "torch/csrc/jit/compiler/include/llvm_codegen.h"
#include "torch/csrc/jit/compiler/include/schedule.h"
#include "torch/csrc/jit/compiler/include/tensor.h"
#include "torch/csrc/jit/compiler/tests/test_utils.h"

#include <gtest/gtest.h>

#include <numeric>

using namespace torch::jit::compiler;
using namespace torch::jit::compiler::schedule;

template <typename T>
static void assertAllEqual(const std::vector<T>& vec, const T& val) {
  for (auto const& elt : vec) {
    ASSERT_EQ(elt, val);
  }
}

TEST(LLVMTest, IntImmTest) {
  auto a = IntImm::make(2);
  LLVMCodeGen cg;
  a.accept(&cg);
  EXPECT_EQ(cg.value<int>(), 2);
}

TEST(LLVMTest, FloatImmTest) {
  auto a = FloatImm::make(1.0);
  LLVMCodeGen cg({}, kFloat32);
  a.accept(&cg);
  EXPECT_EQ(cg.value<float>(), 1.0);
}

TEST(LLVMTest, IntAddTest) {
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Add::make(a, b);
  LLVMCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value<int>(), 5);
}

TEST(LLVMTest, IntSubTest) {
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Sub::make(a, b);
  LLVMCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value<int>(), -1);
}

TEST(LLVMTest, IntMulTest) {
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Mul::make(a, b);
  LLVMCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value<int>(), 6);
}

TEST(LLVMTest, IntDivTest) {
  auto a = IntImm::make(6);
  auto b = IntImm::make(3);
  auto c = Div::make(a, b);
  LLVMCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value<int>(), 2);
}

TEST(LLVMTest, IntToFloatCastTest) {
  auto a = IntImm::make(2);
  auto b = Cast::make(kFloat32, a);
  LLVMCodeGen cg({}, kFloat32);
  b.accept(&cg);
  EXPECT_EQ(cg.value<float>(), 2.0);
}

TEST(LLVMTest, FloatToIntCastTest) {
  auto a = FloatImm::make(2.0);
  auto b = Cast::make(kInt32, a);
  LLVMCodeGen cg;
  b.accept(&cg);
  EXPECT_EQ(cg.value<int>(), 2);
}

TEST(LLVMTest, BufferTest) {
  Buffer a(Var("A", kHandle), kFloat32, {32});
  LLVMCodeGen cg({&a});
  std::vector<int32_t> v(5);
  std::vector<void*> args({v.data()});
  auto rv = IntImm::make(0);
  rv.accept(&cg);
  EXPECT_EQ(cg.value<int>(args), 0);
}

TEST(LLVMTest, LoadStoreTest) {
  Buffer a(Var("A", kHandle), kInt32, {1});
  Buffer b(Var("B", kHandle), kInt32, {1});
  std::vector<int32_t> a_buffer = {42};
  std::vector<int32_t> b_buffer = {-11};

  LLVMCodeGen cg({&a, &b});
  auto store = Store::make(
      b,
      IntImm::make(0),
      Load::make(a, IntImm::make(0), IntImm::make(1)),
      IntImm::make(1));
  store.accept(&cg);
  std::vector<void*> args({a_buffer.data(), b_buffer.data()});
  EXPECT_EQ(cg.value<int>(args), 0);
  EXPECT_EQ(a_buffer[0], 42);
  EXPECT_EQ(b_buffer[0], 42);
}

TEST(LLVMTest, MemcpyTest) {
  constexpr int N = 32;
  Buffer a(Var("A", kHandle), kInt32, {N});
  Buffer b(Var("B", kHandle), kInt32, {N});
  std::vector<int32_t> a_buffer(N, 42);
  std::vector<int32_t> b_buffer(N, 0);

  auto mask = IntImm::make(1);
  Var i("i", kInt32);
  auto memcpy_expr =
      For::make(i, 0, N, Store::make(b, i, Load::make(a, i, mask), mask));

  LLVMCodeGen cg({&a, &b});
  memcpy_expr.accept(&cg);

  std::vector<void*> args({a_buffer.data(), b_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  assertAllEqual(a_buffer, 42);
  assertAllEqual(b_buffer, 42);
}

TEST(LLVMTest, BzeroTest) {
  constexpr int N = 32;
  Buffer b(Var("B", kHandle), kInt32, {N});
  std::vector<int32_t> b_buffer(N, 11);

  auto mask = IntImm::make(1);
  Var i("i", kInt32);
  auto memcpy_expr =
      For::make(i, 0, N, Store::make(b, i, IntImm::make(0), mask));

  LLVMCodeGen cg({&b});
  memcpy_expr.accept(&cg);

  std::vector<void*> args({b_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(b_buffer.size(), N);
  assertAllEqual(b_buffer, 0);
}

TEST(LLVMTest, ElemwiseAdd) {
  constexpr int N = 1024;
  Buffer a(Var("A", kHandle), kInt32, {N});
  Buffer b(Var("B", kHandle), kInt32, {N});
  Buffer c(Var("C", kHandle), kInt32, {N});
  std::vector<int32_t> a_buffer(N, 41);
  std::vector<int32_t> b_buffer(N, 1);
  std::vector<int32_t> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  Var i("i", kInt32);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Add::make(Load::make(a, i, mask), Load::make(b, i, mask)),
          mask));

  LLVMCodeGen cg({&a, &b, &c});
  memcpy_expr.accept(&cg);

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41);
  assertAllEqual(b_buffer, 1);
  assertAllEqual(c_buffer, 42);
}

TEST(LLVMTest, ElemwiseAddFloat) {
  constexpr int N = 1024;
  Buffer a(Var("A", kHandle), kFloat32, {N});
  Buffer b(Var("B", kHandle), kFloat32, {N});
  Buffer c(Var("C", kHandle), kFloat32, {N});
  std::vector<float> a_buffer(N, 41);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  Var i("i", kInt32);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      Store::make(c, i, Load::make(a, i, mask) + Load::make(b, i, mask), mask));

  LLVMCodeGen cg({&a, &b, &c});
  memcpy_expr.accept(&cg);

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41.0f);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 42.0f);
}

TEST(LLVMTest, StoreFloat) {
  Buffer result(Var("result", kHandle), kFloat32, {1});
  std::vector<float> result_buffer = {0.0f};
  auto expr = Store::make(
      result, IntImm::make(0), FloatImm::make(3.14f), IntImm::make(1));
  LLVMCodeGen cg({&result});
  expr.accept(&cg);
  std::vector<void*> args({result_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);
  EXPECT_EQ(result_buffer[0], 3.14f);
}

TEST(LLVMTest, SimpleMath01) {
  const int N = 1024;
  Tensor tensor = Compute(
      "f", {{N, "i"}}, [](const Var& i) { return cast<float>(i * i + 1); });
  Schedule sch = Schedule::make({tensor});
  Stmt stmt = sch.Lower();
  Buffer f_buf(tensor.function().func_var(), kFloat32, {N});
  LLVMCodeGen cg({&f_buf});
  stmt.accept(&cg);

  int kPaddingSize = 8;
  float kPaddingValue = 0.1357;
  std::vector<float> f_vec(N + 2 * kPaddingSize, kPaddingValue);
  std::vector<void*> args({f_vec.data() + kPaddingSize});
  int value = cg.value<int>(args);
  ASSERT_EQ(value, 0);
  std::vector<float> f_ref(N + 2 * kPaddingSize, kPaddingValue);
  for (int i = 0; i < N; i++) {
    f_ref[i + kPaddingSize] = i * i + 1;
  }
  for (int i = 0; i < f_ref.size(); ++i) {
    EXPECT_NEAR(f_vec[i], f_ref[i], 1e-5) << "element index: " << i;
  }
}

TEST(LLVMTest, ComputeMul) {
  const int N = 1024;
  Buffer a(Var("a", kHandle), kFloat32, {N});
  Buffer b(Var("b", kHandle), kFloat32, {N});
  Tensor c = Compute("c", {{N, "i"}}, [&](const Var& i) {
    return Load::make(a, i, 1) * Load::make(b, i, 1);
  });

  Buffer c_buf(c.function().func_var(), kFloat32, {N});
  Schedule sch = Schedule::make({c});
  Stmt s = sch.Lower();

  LLVMCodeGen cg({&a, &b, &c_buf});
  s.accept(&cg);

  std::vector<float> a_vec(N, 21.0f);
  std::vector<float> b_vec(N, 2.0f);
  std::vector<float> c_vec(N, 0.0f);
  std::vector<void*> args({a_vec.data(), b_vec.data(), c_vec.data()});
  ASSERT_EQ(cg.value<int>(args), 0);
  assertAllEqual(c_vec, 42.0f);
}

TEST(LLVMTest, BroadcastAdd) {
  const int M = 32;
  const int N = 1024;
  Buffer a(Var("a", kHandle), kFloat32, {M, N});
  Buffer b(Var("b", kHandle), kFloat32, {N});
  Tensor c =
      Compute("c", {{M, "i"}, {N, "j"}}, [&](const Var& i, const Var& j) {
        Expr mask(1);
        return Load::make(a, i * N + j, mask) + Load::make(b, j, mask);
      });

  Buffer c_buf(c.function().func_var(), kFloat32, {M, N});
  Schedule sch = Schedule::make({c});
  Stmt s = sch.Lower();

  LLVMCodeGen cg({&a, &b, &c_buf});
  s.accept(&cg);

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
