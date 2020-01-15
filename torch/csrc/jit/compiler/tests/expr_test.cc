#include <sstream>
#include <stdexcept>

#include <gtest/gtest.h>

#include "torch/csrc/jit/compiler/include/ir_printer.h"
#include "torch/csrc/jit/compiler/tests/test_utils.h"

using namespace torch::jit::compiler;

TEST(ExprTest, BasicValueTest) {
  Expr a = IntImm::make(2), b = IntImm::make(3);
  Expr c = Add::make(a, b);
  SimpleIREvaluator eval;
  c.accept(&eval);
  EXPECT_EQ(eval.value().as<int>(), 5);
}

TEST(ExprTest, BasicValueTest02) {
  Expr a(2.0f);
  Expr b(3.0f);
  Expr c(4.0f);
  Expr d(5.0f);
  Expr f = (a + b) - (c + d);
  SimpleIREvaluator eval;
  f.accept(&eval);
  EXPECT_EQ(eval.value().as<float>(), -4.0f);
}

TEST(ExprTest, LetTest01) {
  Var x("x", kFloat32);
  Expr value = Expr(3.f);
  Expr body = Expr(2.f) + (x * Expr(3.f) + Expr(4.f));
  Expr result = Let::make(x, Expr(3.f), body);
  SimpleIREvaluator eval;
  result.accept(&eval);
  EXPECT_EQ(eval.value().as<float>(), 2 + (3 * 3 + 4));
}

TEST(ExprTest, LetTest02) {
  Var x("x", kFloat32);
  Var y("y", kFloat32);
  Expr value = Expr(3.f);
  Expr body = Expr(2.f) + (x * Expr(3.f) + Expr(4.f) * y);
  Expr e1 = Let::make(x, Expr(3.f), body);
  Expr e2 = Let::make(y, Expr(6.f), e1);
  SimpleIREvaluator eval;
  e2.accept(&eval);
  EXPECT_EQ(eval.value().as<float>(), 2 + (3 * 3 + 4 * 6));
}

TEST(ExprTest, Tensor01) {
  Tensor tensor =
      Compute("f", {{3, "x"}, {4, "y"}}, [](const Var& x, const Var& y) {
        return Expr(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  std::vector<float> result;
  SimpleTensorEvaluator<float> tensor_eval;
  tensor_eval.evaluate(tensor, &result);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      float reference_v = 1 + i * i + j * j;
      int index = i * 4 + j;
      EXPECT_EQ(result[index], reference_v);
    }
  }
}

TEST(ExprTest, VectorAdd01) {
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

  SimpleIREvaluator ir_eval;
  SimpleIREvaluator::BufferMapping buffer_mapping;
  const int kPadding = 8;
  float kPaddingValue = 0.1357;
  std::vector<float> a_v(kTotalSize + 2 * kPadding, kPaddingValue);
  std::vector<float> b_v(kTotalSize + 2 * kPadding, kPaddingValue);
  std::vector<float> c_v(kTotalSize + 2 * kPadding, kPaddingValue);
  std::vector<float> c_ref(kTotalSize + 2 * kPadding, kPaddingValue);
  for (int i = 0; i < kTotalSize; i++) {
    a_v[i + kPadding] = i * i;
    b_v[i + kPadding] = i * i * 4;
    c_ref[i + kPadding] = a_v[i + kPadding] + b_v[i + kPadding];
  }
  buffer_mapping[a_buf.data().node()] = &a_v[kPadding];
  buffer_mapping[b_buf.data().node()] = &b_v[kPadding];
  buffer_mapping[c_buf.data().node()] = &c_v[kPadding];
  ir_eval.SetBufferMapping(buffer_mapping);
  stmt.accept(&ir_eval);
  for (int i = 0; i < c_v.size(); ++i) {
    ASSERT_NEAR(c_v[i], c_ref[i], 1e-5) << "i: " << i;
  }
}

TEST(ExprTest, Substitute01) {
  {
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
  // TODO: move this to a test fixture and enable for all tests.
  ASSERT_EQ(RefCounted::CheckNoLiveRefCount(), true);
}
