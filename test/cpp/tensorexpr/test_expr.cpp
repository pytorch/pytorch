#include "test/cpp/tensorexpr/test_base.h"

#include "test/cpp/tensorexpr/test_utils.h"
#include "torch/csrc/jit/tensorexpr/buffer.h"
#include "torch/csrc/jit/tensorexpr/eval.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;

using SimpleIRExprEval = ExprEval<SimpleIREvaluator>;

void testExprBasicValueTest() {
  KernelScope kernel_scope;
  Expr a = IntImm::make(2), b = IntImm::make(3);
  Expr c = Add::make(a, b);
  SimpleIRExprEval eval(c);
  EXPECT_EQ(eval.value<int>(), 5);
}

void testExprBasicValueTest02() {
  KernelScope kernel_scope;
  Expr a(2.0f);
  Expr b(3.0f);
  Expr c(4.0f);
  Expr d(5.0f);
  Expr f = (a + b) - (c + d);
  SimpleIRExprEval eval(f);
  EXPECT_EQ(eval.value<float>(), -4.0f);
}

void testExprLetTest01() {
  KernelScope kernel_scope;
  Var x("x", kFloat32);
  Expr value = Expr(3.f);
  Expr body = Expr(2.f) + (x * Expr(3.f) + Expr(4.f));
  Expr result = Let::make(x, Expr(3.f), body);
  SimpleIRExprEval eval(result);
  EXPECT_EQ(eval.value<float>(), 2 + (3 * 3 + 4));
}

void testExprLetTest02() {
  KernelScope kernel_scope;
  Var x("x", kFloat32);
  Var y("y", kFloat32);
  Expr value = Expr(3.f);
  Expr body = Expr(2.f) + (x * Expr(3.f) + Expr(4.f) * y);
  Expr e1 = Let::make(x, Expr(3.f), body);
  Expr e2 = Let::make(y, Expr(6.f), e1);
  SimpleIRExprEval eval(e2);
  EXPECT_EQ(eval.value<float>(), 2 + (3 * 3 + 4 * 6));
}

static Expr test_01(const Expr& expr) {
  return expr;
}

void testExprVectorAdd01() {
  KernelScope kernel_scope;
  const int kVectorSize = 8;
  const int kVectorCount = 128;
  const int kTotalSize = kVectorSize * kVectorCount;

  Buffer a_buf(Var("A", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer c_buf(Var("C", kHandle), kFloat32, {Expr(kTotalSize)});

  /*
  Build the following:
    for (int index = 0; index < kVectorCount; index++) {
      store(c_buf, ramp(index * 8, 1, 8),
            load(a_buf, ramp(index * 8, 1, 8) +
            load(b_buf, ramp(index * 8, 1, 8))))
    }
  */
  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      Ramp::make(index * kVectorSize, 1, kVectorSize),
      Broadcast::make(1, kVectorSize));
  Expr load_b = Load::make(
      b_buf,
      Ramp::make(index * kVectorSize, 1, kVectorSize),
      Broadcast::make(1, kVectorSize));
  Expr value = load_a + load_b;
  Stmt store_c = Store::make(
      c_buf,
      Ramp::make(index * kVectorSize, 1, kVectorSize),
      value,
      Broadcast::make(1, kVectorSize));
  Stmt stmt = For::make(index, 0, kVectorCount, store_c);

  EXPECT_EQ(load_a.dtype(), Dtype(kFloat32, kVectorSize));
  EXPECT_EQ(load_b.dtype(), Dtype(kFloat32, kVectorSize));
  EXPECT_EQ(value.dtype(), Dtype(kFloat32, kVectorSize));
}

void testExprCompareSelectEQ() {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Buffer a(Var("A", kHandle), kInt32, {N});
  Buffer b(Var("B", kHandle), kInt32, {N});
  Buffer c(Var("C", kHandle), kInt32, {N});
  std::vector<int> a_buffer(N, 1);
  std::vector<int> b_buffer(N, 1);
  std::vector<int> c_buffer(N, 0);
  std::vector<int> c_ref(N, 0);

  auto mask = IntImm::make(1);
  Var i("i", kInt32);
  auto memcpy_expr = For::make(
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

  SimpleIREvaluator ir_eval(memcpy_expr, a, b, c);
  ir_eval(a_buffer, b_buffer, c_buffer);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);

  assertAllEqual(a_buffer, 1);
  assertAllEqual(b_buffer, 1);
  assertAllEqual(c_buffer, 1);
}

void testExprSubstitute01() {
  KernelScope kernel_scope;
  Expr x = Variable::make("x", kFloat32);
  Expr y = Variable::make("y", kFloat32);
  Expr e = (x - 1.0f) * (x + y + 2.0f);

  Expr z = Variable::make("z", kFloat32);
  Expr e2 = Substitute(&e, {{x, z + 1.0f}});
  Expr e2_ref = ((z + 1.0f) - 1.0f) * ((z + 1.0f) + y + 2.0f);
  std::ostringstream oss;
  oss << e2;
  std::string e2_str = oss.str();

  oss.str("");
  oss << e2_ref;
  std::string e2_ref_str = oss.str();
  ASSERT_EQ(e2_str, e2_ref_str);
}

void testExprDynamicShapeAdd() {
  KernelScope kernel_scope;
  auto testWithSize = [](int32_t size) {
    Var n("n", kInt32);
    Buffer a(Var("a", kHandle), kFloat32, {n});
    Buffer b(Var("b", kHandle), kFloat32, {n});
    Buffer c(Var("c", kHandle), kFloat32, {n});
    Var i("i", kInt32);
    Stmt s = For::make(i, 0, n, Store::make(c, i, a(i) + b(i), 1));
    std::vector<float> aData(size, 1.0f);
    std::vector<float> bData(size, 2.0f);
    std::vector<float> cData(size, 0.0f);
    SimpleIREvaluator(s, a, b, c, n)(aData, bData, cData, size);
    ExpectAllNear(cData, std::vector<float>(size, 3.0f), 1e-7);
  };
  testWithSize(1);
  testWithSize(16);
  testWithSize(37);
}

void testIfThenElse01() {
  KernelScope kernel_scope;
  Expr v = ifThenElse(Expr(1), Expr(1.0f), Expr(2.0f));

  std::ostringstream oss;
  oss << v;
  ASSERT_EQ(oss.str(), "IfThenElse(1, 1, 2)");

  SimpleIRExprEval eval(v);
  ASSERT_EQ(eval.value<float>(), 1.0f);
}

void testIfThenElse02() {
  KernelScope kernel_scope;
  Expr v = ifThenElse(Expr(0), Expr(1.0f), Expr(2.0f));

  std::ostringstream oss;
  oss << v;
  ASSERT_EQ(oss.str(), "IfThenElse(0, 1, 2)");

  SimpleIRExprEval eval(v);
  ASSERT_EQ(eval.value<float>(), 2.0f);
}

} // namespace jit
} // namespace torch
