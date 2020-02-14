#include "test/cpp/tensorexpr/test_base.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;

void testTypeTest01() {
  KernelScope kernel_scope;
  {
    Dtype dt1 = kInt32;
    EXPECT_EQ(dt1, kInt32);
  }
  {
    Dtype dt2_a(kInt32, 8);
    Dtype dt2_b(kInt32, 4);
    Dtype dt2_c(kInt32, 8);
    EXPECT_EQ(dt2_a, dt2_c);
    EXPECT_NE(dt2_a, dt2_b);
  }
  {
    EXPECT_EQ(kInt32, ToDtype<int>());
    EXPECT_EQ(kFloat32, ToDtype<float>());
  }
  {
    Dtype int32x8(kInt32, 8);
    Dtype float32x8(kFloat32, 8);
    EXPECT_NE(int32x8, float32x8);
    EXPECT_EQ(float32x8, BinaryOpDtype(int32x8, float32x8));
    EXPECT_EQ(float32x8, BinaryOpDtype(float32x8, int32x8));
    EXPECT_EQ(int32x8, BinaryOpDtype(int32x8, int32x8));
    EXPECT_EQ(float32x8, BinaryOpDtype(float32x8, float32x8));
  }
}
} // namespace jit
} // namespace torch
