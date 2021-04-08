#include <gtest/gtest.h>

#include "torch/csrc/jit/tensorexpr/eval.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;

TEST(Type, Test01) {
  KernelScope kernel_scope;
  {
    Dtype dt1 = kInt;
    ASSERT_EQ(dt1, kInt);
  }
  {
    Dtype dt2_a(kInt, 8);
    Dtype dt2_b(kInt, 4);
    Dtype dt2_c(ScalarType::Int, 8);
    ASSERT_EQ(dt2_a, dt2_c);
    ASSERT_NE(dt2_a, dt2_b);
  }
  {
    ASSERT_EQ(kInt, ToDtype<int>());
    ASSERT_EQ(kFloat, ToDtype<float>());
    ASSERT_EQ(kByte, ToDtype<uint8_t>());
    ASSERT_EQ(kChar, ToDtype<int8_t>());
    ASSERT_EQ(kShort, ToDtype<int16_t>());
    ASSERT_EQ(kLong, ToDtype<int64_t>());
    ASSERT_EQ(kHalf, ToDtype<at::Half>());
    ASSERT_EQ(kDouble, ToDtype<double>());
    ASSERT_EQ(kBool, ToDtype<bool>());
  }
  {
    Dtype int32x8(kInt, 8);
    Dtype float32x8(kFloat, 8);
    ASSERT_NE(int32x8, float32x8);
    ASSERT_EQ(float32x8, BinaryOpDtype(int32x8, float32x8));
    ASSERT_EQ(float32x8, BinaryOpDtype(float32x8, int32x8));
    ASSERT_EQ(int32x8, BinaryOpDtype(int32x8, int32x8));
    ASSERT_EQ(float32x8, BinaryOpDtype(float32x8, float32x8));
  }
}

TEST(Type, BitCasting) {
  {
    KernelScope kernel_scope;
    VarHandle x("x", kFloat);
    ExprHandle y = bitcast<int32_t>(x);
    ASSERT_EQ(y.dtype(), kInt);
  }
  {
    KernelScope kernel_scope;
    VarHandle x("x", kInt);
    ExprHandle y = bitcast<float>(x);
    ASSERT_EQ(y.dtype(), kFloat);
  }
  {
    KernelScope kernel_scope;
    VarHandle x("x", kShort);
    ExprHandle y = bitcast<at::Half>(x);
    ASSERT_EQ(y.dtype(), kHalf);
  }
  {
    KernelScope kernel_scope;
    VarHandle x("x", kHalf);
    ExprHandle y = bitcast<int16_t>(x);
    ASSERT_EQ(y.dtype(), kShort);
  }

  constexpr int16_t ref16 = 1337;
  constexpr int32_t ref32 = 1337;
  constexpr int64_t ref64 = 1337;
  at::Half reff16 = 1337.0f;
  constexpr float reff32 = 1337.0f;
  constexpr double reff64 = 1337.0f;
  using SimpleIRExprEval = ExprEval<SimpleIREvaluator>;
  // this is broken
  /*{
    KernelScope kernel_scope;
    at::Half k_;
    at::Half* k = &k_;
    *reinterpret_cast<int16_t*>(k) = ref16;
    auto a = HalfImm::make(*k);
    auto b = BitCast::make(kShort, a);
    SimpleIRExprEval cg(b);
    ASSERT_EQ(cg.value<int16_t>(), ref16);
  }*/

  {
    KernelScope kernel_scope;
    float k = raw_bitcast<float>(ref32);
    auto a = FloatImm::make(k);
    auto b = BitCast::make(kInt, a);
    SimpleIRExprEval cg(b);
    ASSERT_EQ(cg.value<int32_t>(), ref32);
  }

  {
    KernelScope kernel_scope;
    double k = raw_bitcast<double>(ref64);
    auto a = DoubleImm::make(k);
    auto b = BitCast::make(kLong, a);
    SimpleIRExprEval cg(b);
    ASSERT_EQ(cg.value<int64_t>(), ref64);
  }

  {
    KernelScope kernel_scope;
    int64_t k = raw_bitcast<int64_t>(reff64);
    auto a = LongImm::make(k);
    auto b = BitCast::make(kDouble, a);
    SimpleIRExprEval cg(b);
    ASSERT_EQ(cg.value<double>(), reff64);
  }

  {
    KernelScope kernel_scope;
    int32_t k = raw_bitcast<int32_t>(reff32);
    auto a = IntImm::make(k);
    auto b = BitCast::make(kFloat, a);
    SimpleIRExprEval cg(b);
    ASSERT_EQ(cg.value<float>(), reff32);
  }

  // This segfaults :(
  /*{
    KernelScope kernel_scope;
    VarHandle x("x", kDouble);
    ASSERT_ANY_THROW(ExprHandle y = bitcast<int32_t>(x));
  }
  {
    KernelScope kernel_scope;
    VarHandle x("x", kFloat);
    ASSERT_ANY_THROW(ExprHandle y = bitcast<int64_t>(x));
  }
  {
    KernelScope kernel_scope;
    VarHandle x("x", kLong);
    ASSERT_ANY_THROW(ExprHandle y = bitcast<float>(x));
  }
  {
    KernelScope kernel_scope;
    VarHandle x("x", kShort);
    ASSERT_ANY_THROW(ExprHandle y = bitcast<float>(x));
  }
  {
    KernelScope kernel_scope;
    VarHandle x("x", kInt);
    ASSERT_ANY_THROW(ExprHandle y = bitcast<at::Half>(x));
  }*/
}

TEST(Type, Propagation) {
  // Same types:
  {
    KernelScope kernel_scope;
    VarHandle x("x", kFloat);
    VarHandle y("y", kFloat);
    ExprHandle body = FloatImm::make(2.f) +
        (x * FloatImm::make(3.f) + FloatImm::make(4.f) * y);
    ASSERT_EQ(body.dtype(), kFloat);
  }
  // Int to bigger int:
  {
    KernelScope kernel_scope;
    VarHandle x("x", kShort);
    VarHandle y("y", kLong);
    ExprHandle body =
        ShortImm::make(2.f) + (x * ShortImm::make(3) + ShortImm::make(4) * y);
    ASSERT_EQ(body.dtype(), kLong);
  }
  // Float to bigger float:
  {
    KernelScope kernel_scope;
    VarHandle x("x", kHalf);
    VarHandle y("y", kDouble);
    ExprHandle body =
        HalfImm::make(2.f) + (x * HalfImm::make(3) + HalfImm::make(4) * y);
    ASSERT_EQ(body.dtype(), kDouble);
  }
  // Int to Float:
  {
    KernelScope kernel_scope;
    VarHandle x("x", kFloat);
    VarHandle y("y", kInt);
    ExprHandle body =
        IntImm::make(2) + (x * IntImm::make(3) + IntImm::make(4) * y);
    ASSERT_EQ(body.dtype(), kFloat);
  }
  // Smaller float, bigger Int:
  {
    KernelScope kernel_scope;
    VarHandle x("x", kHalf);
    VarHandle y("y", kLong);
    ExprHandle body =
        HalfImm::make(2) + (x * HalfImm::make(3) + HalfImm::make(4) * y);
    ASSERT_EQ(body.dtype(), kHalf);
  }
  // Bigger float, smaller Int:
  {
    KernelScope kernel_scope;
    VarHandle x("x", kChar);
    VarHandle y("y", kDouble);
    ExprHandle body =
        CharImm::make(2) + (x * CharImm::make(3) + CharImm::make(4) * y);
    ASSERT_EQ(body.dtype(), kDouble);
  }
  // Sign change char/byte upgrades to short:
  {
    KernelScope kernel_scope;
    VarHandle x("x", kChar);
    VarHandle y("y", kByte);
    ExprHandle body =
        CharImm::make(2) + (x * CharImm::make(3) + CharImm::make(4) * y);
    ASSERT_EQ(body.dtype(), kShort);
  }
}
} // namespace jit
} // namespace torch
