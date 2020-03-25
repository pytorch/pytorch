#include "test/cpp/tensorexpr/test_base.h"

#include "test/cpp/tensorexpr/padded_buffer.h"
#include "test/cpp/tensorexpr/test_utils.h"
#include "torch/csrc/jit/tensorexpr/buffer.h"
#include "torch/csrc/jit/tensorexpr/eval.h"
#include "torch/csrc/jit/tensorexpr/function.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "torch/csrc/jit/tensorexpr/loopnest.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

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
  ExprHandle a = IntImm::make(2), b = IntImm::make(3);
  ExprHandle c = Add::make(a, b);
  SimpleIRExprEval eval(c);
  EXPECT_EQ(eval.value<int>(), 5);
}

void testExprBasicValueTest02() {
  KernelScope kernel_scope;
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle c(4.0f);
  ExprHandle d(5.0f);
  ExprHandle f = (a + b) - (c + d);
  SimpleIRExprEval eval(f);
  EXPECT_EQ(eval.value<float>(), -4.0f);
}

void testExprLetTest01() {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  ExprHandle value = ExprHandle(3.f);
  ExprHandle body = ExprHandle(2.f) + (x * ExprHandle(3.f) + ExprHandle(4.f));
  ExprHandle result = Let::make(x, ExprHandle(3.f), body);
  SimpleIRExprEval eval(result);
  EXPECT_EQ(eval.value<float>(), 2 + (3 * 3 + 4));
}

void testExprLetTest02() {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  VarHandle y("y", kFloat);
  ExprHandle value = ExprHandle(3.f);
  ExprHandle body =
      ExprHandle(2.f) + (x * ExprHandle(3.f) + ExprHandle(4.f) * y);
  ExprHandle e1 = Let::make(x, ExprHandle(3.f), body);
  ExprHandle e2 = Let::make(y, ExprHandle(6.f), e1);
  SimpleIRExprEval eval(e2);
  EXPECT_EQ(eval.value<float>(), 2 + (3 * 3 + 4 * 6));
}

void testExprLetStmtTest01() {
  KernelScope kernel_scope;
  Buffer a_buf("a", kFloat, {1});
  Buffer b_buf("b", kFloat, {1});

  ExprHandle load_a = Load::make(a_buf, 0, 1);
  VarHandle var = VarHandle("v", kFloat);
  Stmt* store_b = Store::make(b_buf, 0, var, 1);
  Stmt* let_store = LetStmt::make(var, load_a, store_b);
  SimpleIREvaluator eval(let_store, a_buf, b_buf);

  PaddedBuffer<float> a_v(1);
  PaddedBuffer<float> b_v(1);
  PaddedBuffer<float> b_ref(1);

  a_v(0) = 23;
  b_ref(0) = a_v(0);
  eval(a_v, b_v);

  ExpectAllNear(b_v, b_ref, 1e-5);
}

static ExprHandle test_01(const ExprHandle& expr) {
  return expr;
}

void testExprIntTest() {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  ExprHandle value = ExprHandle(3);
  ExprHandle body = ExprHandle(2) + (x * ExprHandle(3) + ExprHandle(4));
  ExprHandle result = Let::make(x, ExprHandle(3), body);
  SimpleIRExprEval eval(result);
  EXPECT_EQ(eval.value<int>(), 2 + (3 * 3 + 4));
}

void testExprFloatTest() {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  ExprHandle value = ExprHandle((float)3);
  ExprHandle body =
      ExprHandle((float)2) + (x * ExprHandle((float)3) + ExprHandle((float)4));
  ExprHandle result = Let::make(x, ExprHandle((float)3), body);
  SimpleIRExprEval eval(result);
  EXPECT_EQ(eval.value<float>(), 2 + (3 * 3 + 4));
}

void testExprByteTest() {
  KernelScope kernel_scope;
  VarHandle x("x", kByte);
  ExprHandle value = ExprHandle((uint8_t)3);
  ExprHandle body = ExprHandle((uint8_t)2) +
      (x * ExprHandle((uint8_t)3) + ExprHandle((uint8_t)4));
  ExprHandle result = Let::make(x, ExprHandle((uint8_t)3), body);
  SimpleIRExprEval eval(result);
  EXPECT_EQ(eval.value<uint8_t>(), 2 + (3 * 3 + 4));
}

void testExprCharTest() {
  KernelScope kernel_scope;
  VarHandle x("x", kChar);
  ExprHandle value = ExprHandle((int8_t)3);
  ExprHandle body = ExprHandle((int8_t)2) +
      (x * ExprHandle((int8_t)3) + ExprHandle((int8_t)4));
  ExprHandle result = Let::make(x, ExprHandle((int8_t)3), body);
  SimpleIRExprEval eval(result);
  EXPECT_EQ(eval.value<int8_t>(), 2 + (3 * 3 + 4));
}

void testExprShortTest() {
  KernelScope kernel_scope;
  VarHandle x("x", kShort);
  ExprHandle value = ExprHandle((int16_t)3);
  ExprHandle body = ExprHandle((int16_t)2) +
      (x * ExprHandle((int16_t)3) + ExprHandle((int16_t)4));
  ExprHandle result = Let::make(x, ExprHandle((int16_t)3), body);
  SimpleIRExprEval eval(result);
  EXPECT_EQ(eval.value<int16_t>(), 2 + (3 * 3 + 4));
}

void testExprLongTest() {
  KernelScope kernel_scope;
  VarHandle x("x", kLong);
  ExprHandle value = ExprHandle((int64_t)3);
  ExprHandle body = ExprHandle((int64_t)2) +
      (x * ExprHandle((int64_t)3) + ExprHandle((int64_t)4));
  ExprHandle result = Let::make(x, ExprHandle((int64_t)3), body);
  SimpleIRExprEval eval(result);
  EXPECT_EQ(eval.value<int64_t>(), 2 + (3 * 3 + 4));
}

void testExprHalfTest() {
  KernelScope kernel_scope;
  VarHandle x("x", kHalf);
  ExprHandle value = ExprHandle((at::Half)3);
  ExprHandle body = ExprHandle((at::Half)2) +
      (x * ExprHandle((at::Half)3) + ExprHandle((at::Half)4));
  ExprHandle result = Let::make(x, ExprHandle((at::Half)3), body);
  SimpleIRExprEval eval(result);
  EXPECT_EQ(eval.value<at::Half>(), 2 + (3 * 3 + 4));
}

void testExprDoubleTest() {
  KernelScope kernel_scope;
  VarHandle x("x", kDouble);
  ExprHandle value = ExprHandle((double)3);
  ExprHandle body = ExprHandle((double)2) +
      (x * ExprHandle((double)3) + ExprHandle((double)4));
  ExprHandle result = Let::make(x, ExprHandle((double)3), body);
  SimpleIRExprEval eval(result);
  EXPECT_EQ(eval.value<double>(), 2 + (3 * 3 + 4));
}
void testExprVectorAdd01() {
  KernelScope kernel_scope;
  const int kVectorSize = 8;
  const int kVectorCount = 128;
  const int kTotalSize = kVectorSize * kVectorCount;

  Buffer a_buf(VarHandle("A", kHandle), kFloat, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat, {ExprHandle(kTotalSize)});
  Buffer c_buf(VarHandle("C", kHandle), kFloat, {ExprHandle(kTotalSize)});

  /*
  Build the following:
    for (int index = 0; index < kVectorCount; index++) {
      store(c_buf, ramp(index * 8, 1, 8),
            load(a_buf, ramp(index * 8, 1, 8) +
            load(b_buf, ramp(index * 8, 1, 8))))
    }
  */
  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = Load::make(
      a_buf,
      Ramp::make(index * kVectorSize, 1, kVectorSize),
      Broadcast::make(1, kVectorSize));
  ExprHandle load_b = Load::make(
      b_buf,
      Ramp::make(index * kVectorSize, 1, kVectorSize),
      Broadcast::make(1, kVectorSize));
  ExprHandle value = load_a + load_b;
  Stmt* store_c = Store::make(
      c_buf,
      Ramp::make(index * kVectorSize, 1, kVectorSize),
      value,
      Broadcast::make(1, kVectorSize));
  Stmt* stmt = For::make(index, 0, kVectorCount, store_c);

  EXPECT_EQ(load_a.dtype(), Dtype(kFloat, kVectorSize));
  EXPECT_EQ(load_b.dtype(), Dtype(kFloat, kVectorSize));
  EXPECT_EQ(value.dtype(), Dtype(kFloat, kVectorSize));

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> c_ref(kTotalSize);
  for (int i = 0; i < kTotalSize; i++) {
    a_v(i) = i * i;
    b_v(i) = i * i * 4;
    c_ref(i) = a_v(i) + b_v(i);
  }
  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);
  ExpectAllNear(c_v, c_ref, 1e-5);
}

void testExprCompareSelectEQ() {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Buffer a(VarHandle("A", kHandle), kInt, {N});
  Buffer b(VarHandle("B", kHandle), kInt, {N});
  Buffer c(VarHandle("C", kHandle), kInt, {N});
  std::vector<int> a_buffer(N, 1);
  std::vector<int> b_buffer(N, 1);
  std::vector<int> c_buffer(N, 0);
  std::vector<int> c_ref(N, 0);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt);
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
  ExprHandle x = Var::make("x", kFloat);
  ExprHandle y = Var::make("y", kFloat);
  ExprHandle e = (x - 1.0f) * (x + y + 2.0f);

  ExprHandle z = Var::make("z", kFloat);
  ExprHandle e2(Substitute(e.node(), {{x, z + 1.0f}}));
  ExprHandle e2_ref = ((z + 1.0f) - 1.0f) * ((z + 1.0f) + y + 2.0f);
  std::ostringstream oss;
  oss << e2;
  std::string e2_str = oss.str();

  oss.str("");
  oss << e2_ref;
  std::string e2_ref_str = oss.str();
  ASSERT_EQ(e2_str, e2_ref_str);
}

void testExprMath01() {
  KernelScope kernel_scope;
  ExprHandle v = sin(ExprHandle(1.0f));

  std::ostringstream oss;
  oss << v;
  ASSERT_EQ(oss.str(), "sin(1.f)");

  SimpleIRExprEval eval(v);
  float v_ref = std::sin(1.0f);
  float res = eval.value<float>();
  ASSERT_NEAR(res, v_ref, 1e-6);
}

void testExprUnaryMath01() {
  KernelScope kernel_scope;
  struct TestConfig {
    std::function<ExprHandle(const ExprHandle&)> func;
    std::function<float(float)> ref_func;
  };

  std::vector<TestConfig> test_configs = {
      {[](const ExprHandle& v) { return sin(v); },
       [](float v) { return std::sin(v); }},
      {[](const ExprHandle& v) { return sin(v); },
       [](float v) { return std::sin(v); }},
      {[](const ExprHandle& v) { return tan(v); },
       [](float v) { return std::tan(v); }},
      {[](const ExprHandle& v) { return asin(v); },
       [](float v) { return std::asin(v); }},
      {[](const ExprHandle& v) { return acos(v); },
       [](float v) { return std::acos(v); }},
      {[](const ExprHandle& v) { return atan(v); },
       [](float v) { return std::atan(v); }},
      {[](const ExprHandle& v) { return sinh(v); },
       [](float v) { return std::sinh(v); }},
      {[](const ExprHandle& v) { return cosh(v); },
       [](float v) { return std::cosh(v); }},
      {[](const ExprHandle& v) { return tanh(v); },
       [](float v) { return std::tanh(v); }},
      {[](const ExprHandle& v) { return exp(v); },
       [](float v) { return std::exp(v); }},
      {[](const ExprHandle& v) { return fabs(v); },
       [](float v) { return std::fabs(v); }},
      {[](const ExprHandle& v) { return log(v); },
       [](float v) { return std::log(v); }},
      {[](const ExprHandle& v) { return log2(v); },
       [](float v) { return std::log2(v); }},
      {[](const ExprHandle& v) { return log10(v); },
       [](float v) { return std::log10(v); }},
      {[](const ExprHandle& v) { return erf(v); },
       [](float v) { return std::erf(v); }},
      {[](const ExprHandle& v) { return sqrt(v); },
       [](float v) { return std::sqrt(v); }},
      {[](const ExprHandle& v) { return rsqrt(v); },
       [](float v) { return 1.0f / std::sqrt(v); }},
      {[](const ExprHandle& v) { return ceil(v); },
       [](float v) { return std::ceil(v); }},
      {[](const ExprHandle& v) { return floor(v); },
       [](float v) { return std::floor(v); }},
      {[](const ExprHandle& v) { return round(v); },
       [](float v) { return std::round(v); }},
      {[](const ExprHandle& v) { return trunc(v); },
       [](float v) { return std::trunc(v); }},
  };

  for (const TestConfig& test_config : test_configs) {
    const float input_v = 0.8765f;
    ExprHandle v = test_config.func(ExprHandle(input_v));
    float v_ref = test_config.ref_func(input_v);
    SimpleIRExprEval eval(v);
    EXPECT_NEAR(eval.value<float>(), v_ref, 1e-6) << "fail: " << v;
  }
}

void testExprBinaryMath01() {
  KernelScope kernel_scope;
  struct TestConfig {
    std::function<ExprHandle(const ExprHandle&, const ExprHandle&)> func;
    std::function<float(float, float)> ref_func;
  };

  std::vector<TestConfig> test_configs = {
      {[](const ExprHandle& v1, const ExprHandle& v2) { return pow(v1, v2); },
       [](float v1, float v2) { return std::pow(v1, v2); }},
      {[](const ExprHandle& v1, const ExprHandle& v2) { return fmod(v1, v2); },
       [](float v1, float v2) { return std::fmod(v1, v2); }},
  };

  for (const TestConfig& test_config : test_configs) {
    const float v1 = 0.8765f;
    float v2 = 1.2345f;
    ExprHandle v_expr = test_config.func(ExprHandle(v1), ExprHandle(v2));
    float v_ref = test_config.ref_func(v1, v2);
    SimpleIRExprEval eval(v_expr);
    EXPECT_NEAR(eval.value<float>(), v_ref, 1e-6) << "fail: " << v_expr;
  }
}

void testExprBitwiseOps() {
  KernelScope kernel_scope;
  ExprHandle a(59);
  ExprHandle b(11);
  ExprHandle c(101);
  ExprHandle d(2);
  ExprHandle f = (((a ^ (b << 1)) & c) >> 2) | d;

  SimpleIRExprEval eval(f);
  EXPECT_EQ(eval.value<int>(), 11);
}

void testExprDynamicShapeAdd() {
  KernelScope kernel_scope;
  auto testWithSize = [](int32_t size) {
    VarHandle n("n", kInt);
    Buffer a(VarHandle("a", kHandle), kFloat, {n});
    Buffer b(VarHandle("b", kHandle), kFloat, {n});
    Buffer c(VarHandle("c", kHandle), kFloat, {n});
    VarHandle i("i", kInt);
    Stmt* s = For::make(i, 0, n, Store::make(c, i, a(i) + b(i), 1));
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

void testCond01() {
  KernelScope kernel_scope;
  const int N = 16;
  PaddedBuffer<float> a_v(N);
  Buffer a_buf("a", kFloat, {N});
  VarHandle index = VarHandle("index", kInt);
  Stmt* assign_x2 =
      Store::make(VarHandle(a_buf.data()), index, cast<float>(index) * 2, 1);
  Stmt* assign_x3 =
      Store::make(VarHandle(a_buf.data()), index, cast<float>(index) * 3, 1);
  ExprHandle even_cond = CompareSelect::make(Mod::make(index, 2), 0, kEQ);
  Stmt* assign = Cond::make(even_cond, assign_x2, assign_x3);
  Stmt* for_stmt = For::make(index, 0, N, assign);
  SimpleIREvaluator(for_stmt, a_buf)(a_v);

  PaddedBuffer<float> a_ref(N);
  for (int i = 0; i < N; i++) {
    if (i % 2 == 0) {
      a_ref(i) = i * 2;
    } else {
      a_ref(i) = i * 3;
    }
  }
  ExpectAllNear(a_v, a_ref, 1e-5);
}

void testIfThenElse01() {
  KernelScope kernel_scope;
  ExprHandle v = ifThenElse(ExprHandle(1), ExprHandle(1.0f), ExprHandle(2.0f));

  std::ostringstream oss;
  oss << v;
  ASSERT_EQ(oss.str(), "IfThenElse(1, 1.f, 2.f)");

  SimpleIRExprEval eval(v);
  ASSERT_EQ(eval.value<float>(), 1.0f);
}

void testIfThenElse02() {
  KernelScope kernel_scope;
  ExprHandle v = ifThenElse(ExprHandle(0), ExprHandle(1.0f), ExprHandle(2.0f));

  std::ostringstream oss;
  oss << v;
  ASSERT_EQ(oss.str(), "IfThenElse(0, 1.f, 2.f)");

  SimpleIRExprEval eval(v);
  ASSERT_EQ(eval.value<float>(), 2.0f);
}

void testStmtClone() {
  KernelScope kernel_scope;
  const int N = 16;

  Buffer a_buf("a", kInt, {N});
  VarHandle index = VarHandle("index", kInt);
  Stmt* body =
      Store::make(VarHandle(a_buf.data()), index, 5, 1);
  Stmt* loop = For::make(index, 0, N, body);

  Stmt* cloned_loop = Stmt::clone(loop);
  std::vector<int> orig_loop_results(N);
  std::vector<int> cloned_loop_results(N);
  SimpleIREvaluator(loop, a_buf)(orig_loop_results);
  SimpleIREvaluator(cloned_loop, a_buf)(cloned_loop_results);

  assertAllEqual(orig_loop_results, 5);
  assertAllEqual(cloned_loop_results, 5);

  // Let's add another assign to the body in the cloned loop and verify that the
  // original statement hasn't changed while the cloned one has.
  Stmt* body_addition = Store::make(VarHandle(a_buf.data()), index, 33, 1);
  Block* cloned_body =
      static_cast<Block*>(static_cast<const For*>(cloned_loop)->body());
  cloned_body->append_stmt(body_addition);

  std::vector<int> orig_loop_results_after_mutation(N);
  std::vector<int> cloned_loop_results_after_mutation(N);
  SimpleIREvaluator(loop, a_buf)(orig_loop_results_after_mutation);
  SimpleIREvaluator(cloned_loop, a_buf)(cloned_loop_results_after_mutation);

  assertAllEqual(orig_loop_results_after_mutation, 5);
  assertAllEqual(cloned_loop_results_after_mutation, 33);
}

} // namespace jit
} // namespace torch
