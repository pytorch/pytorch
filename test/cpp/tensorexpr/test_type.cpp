#include "test/cpp/tensorexpr/test_base.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;

void testTypeTest01() {
  KernelScope kernel_scope;
  {
    Dtype dt1 = kInt;
    EXPECT_EQ(dt1, kInt);
  }
  {
    Dtype dt2_a(kInt, 8);
    Dtype dt2_b(kInt, 4);
    Dtype dt2_c(ScalarType::Int, 8);
    EXPECT_EQ(dt2_a, dt2_c);
    EXPECT_NE(dt2_a, dt2_b);
  }
  {
    EXPECT_EQ(kInt, ToDtype<int>());
    EXPECT_EQ(kFloat, ToDtype<float>());
    EXPECT_EQ(kByte, ToDtype<uint8_t>());
    EXPECT_EQ(kChar, ToDtype<int8_t>());
    EXPECT_EQ(kShort, ToDtype<int16_t>());
    EXPECT_EQ(kLong, ToDtype<int64_t>());
    EXPECT_EQ(kHalf, ToDtype<at::Half>());
    EXPECT_EQ(kDouble, ToDtype<double>());
    EXPECT_EQ(kBool, ToDtype<bool>());
  }
  {
    Dtype int32x8(kInt, 8);
    Dtype float32x8(kFloat, 8);
    EXPECT_NE(int32x8, float32x8);
    EXPECT_EQ(float32x8, BinaryOpDtype(int32x8, float32x8));
    EXPECT_EQ(float32x8, BinaryOpDtype(float32x8, int32x8));
    EXPECT_EQ(int32x8, BinaryOpDtype(int32x8, int32x8));
    EXPECT_EQ(float32x8, BinaryOpDtype(float32x8, float32x8));
  }
}

void testTypePropagation() {
  // Same types:
  {
    KernelScope kernel_scope;
    VarHandle x("x", kFloat);
    VarHandle y("y", kFloat);
    ExprHandle body =
        ExprHandle(2.f) + (x * ExprHandle(3.f) + ExprHandle(4.f) * y);
    ExprHandle e1 = Let::make(x, ExprHandle(3.f), body);
    ExprHandle e2 = Let::make(y, ExprHandle(6.f), e1);
    EXPECT_EQ(e2.dtype(), kFloat);
  }
  // Int to bigger int:
  {
    KernelScope kernel_scope;
    VarHandle x("x", kShort);
    VarHandle y("y", kLong);
    ExprHandle body = ExprHandle((short)2.f) +
        (x * ExprHandle((short)3) + ExprHandle((short)4) * y);
    ExprHandle e1 = Let::make(x, ExprHandle((short)3), body);
    ExprHandle e2 = Let::make(y, ExprHandle((long)6), e1);
    EXPECT_EQ(e2.dtype(), kLong);
  }
  // Float to bigger float:
  {
    KernelScope kernel_scope;
    VarHandle x("x", kHalf);
    VarHandle y("y", kDouble);
    ExprHandle body = ExprHandle((at::Half)2.f) +
        (x * ExprHandle((at::Half)3) + ExprHandle((at::Half)4) * y);
    ExprHandle e1 = Let::make(x, ExprHandle((at::Half)3), body);
    ExprHandle e2 = Let::make(y, ExprHandle((double)6), e1);
    EXPECT_EQ(e2.dtype(), kDouble);
  }
  // Int to Float:
  {
    KernelScope kernel_scope;
    VarHandle x("x", kFloat);
    VarHandle y("y", kInt);
    ExprHandle body = ExprHandle(2) + (x * ExprHandle(3) + ExprHandle(4) * y);
    ExprHandle e1 = Let::make(x, ExprHandle(3.f), body);
    ExprHandle e2 = Let::make(y, ExprHandle(6), e1);
    EXPECT_EQ(e2.dtype(), kFloat);
  }
  // Smaller float, bigger Int:
  {
    KernelScope kernel_scope;
    VarHandle x("x", kHalf);
    VarHandle y("y", kLong);
    ExprHandle body = ExprHandle((at::Half)2) +
        (x * ExprHandle((at::Half)3) + ExprHandle((at::Half)4) * y);
    ExprHandle e1 = Let::make(x, ExprHandle((at::Half)3), body);
    ExprHandle e2 = Let::make(y, ExprHandle(6l), e1);
    EXPECT_EQ(e2.dtype(), kHalf);
  }
  // Bigger float, smaller Int:
  {
    KernelScope kernel_scope;
    VarHandle x("x", kChar);
    VarHandle y("y", kDouble);
    ExprHandle body = ExprHandle((char)2) +
        (x * ExprHandle((char)3) + ExprHandle((char)4) * y);
    ExprHandle e1 = Let::make(x, ExprHandle((char)3), body);
    ExprHandle e2 = Let::make(y, ExprHandle((double)6), e1);
    EXPECT_EQ(e2.dtype(), kDouble);
  }
  // Sign change char/byte upgrades to short:
  {
    KernelScope kernel_scope;
    VarHandle x("x", kChar);
    VarHandle y("y", kByte);
    ExprHandle body = ExprHandle((int8_t)2) +
        (x * ExprHandle((int8_t)3) + ExprHandle((int8_t)4) * y);
    ExprHandle e1 = Let::make(x, ExprHandle((int8_t)3), body);
    ExprHandle e2 = Let::make(y, ExprHandle((uint8_t)6), e1);
    EXPECT_EQ(e2.dtype(), kShort);
  }
}
} // namespace jit
} // namespace torch
