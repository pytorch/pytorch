#include "torch/csrc/jit/compiler/include/ir.h"
#include "torch/csrc/jit/compiler/include/llvm_codegen.h"

#include <gtest/gtest.h>

using namespace torch::jit::compiler;

template<typename T>
static void assertAllEqual(const std::vector<T> &vec, const T &val) {
  for (auto const &elt : vec) {
    ASSERT_EQ(elt, val);
  }
}

TEST(LLVMTest, IntImmTest) {
  auto a = IntImm::make(2);
  LLVMCodeGen cg;
  a.accept(&cg);
  EXPECT_EQ(cg.value(), 2);
}

TEST(LLVMTest, IntAddTest) {
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Add::make(a, b);
  LLVMCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value(), 5);
}

TEST(LLVMTest, IntSubTest) {
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Sub::make(a, b);
  LLVMCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value(), -1);
}

TEST(LLVMTest, IntMulTest) {
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Mul::make(a, b);
  LLVMCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value(), 6);
}

TEST(LLVMTest, IntDivTest) {
  auto a = IntImm::make(6);
  auto b = IntImm::make(3);
  auto c = Div::make(a, b);
  LLVMCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value(), 2);
}

TEST(LLVMTest, BufferTest) {
  Buffer a(Var("A", kHandle), kFloat32, {32});
  LLVMCodeGen cg({&a});
  std::vector<int32_t> v(5);
  std::vector<void *> args({v.data()});
  auto rv = IntImm::make(0);
  rv.accept(&cg);
  EXPECT_EQ(cg.value(args), 0);
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
  std::vector<void *> args({a_buffer.data(), b_buffer.data()});
  EXPECT_EQ(cg.value(args), 0);
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
  auto memcpy_expr = For::make(
    i, 0, N,
    Store::make(b, i, Load::make(a, i, mask), mask));

  LLVMCodeGen cg({&a, &b});
  memcpy_expr.accept(&cg);

  std::vector<void *> args({a_buffer.data(), b_buffer.data()});
  ASSERT_EQ(cg.value(args), 0);

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
  auto memcpy_expr = For::make(
    i, 0, N,
    Store::make(b, i, IntImm::make(0), mask));

  LLVMCodeGen cg({&b});
  memcpy_expr.accept(&cg);

  std::vector<void *> args({b_buffer.data()});
  ASSERT_EQ(cg.value(args), 0);

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
    i, 0, N,
    Store::make(
      c, i,
      Add::make(
        Load::make(a, i, mask),
        Load::make(b, i, mask)),
      mask));

  LLVMCodeGen cg({&a, &b, &c});
  memcpy_expr.accept(&cg);

  std::vector<void *> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41);
  assertAllEqual(b_buffer, 1);
  assertAllEqual(c_buffer, 42);
}
