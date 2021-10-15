#include <gtest/gtest.h>

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensor_core/csrc/lowering_context.h"
#include "lazy_tensor_core/csrc/ops/arithmetic_ir_ops.h"
#include "lazy_tensor_core/csrc/ops/ops.h"
#include "lazy_tensor_core/csrc/ops/scalar.h"
#include "lazy_tensor_core/csrc/ops/select.h"
#include "lazy_tensor_core/csrc/ops/unselect.h"
#include "lazy_tensors/primitive_types.h"

namespace torch_lazy_tensors {
namespace cpp_test {

using lazy_tensors::PrimitiveType;

TEST(IrTest, TestScalarCreate) {
  ir::NodePtr scalar = ir::ops::ScalarOp(1.0, PrimitiveType::F32);
  ASSERT_TRUE(scalar != nullptr);
}

TEST(IrTest, TestHash) {
  ir::NodePtr scalar1 = ir::ops::ScalarOp(1.0, PrimitiveType::F32);
  ir::NodePtr scalar2 = ir::ops::ScalarOp(2.0, PrimitiveType::F32);
  ir::Value add1 = scalar1 + scalar2;

  ir::NodePtr scalar3 = ir::ops::ScalarOp(1.0, PrimitiveType::F32);
  ir::NodePtr scalar4 = ir::ops::ScalarOp(2.0, PrimitiveType::F32);
  ir::Value add2 = scalar1 + scalar2;

  ir::NodePtr scalar5 = ir::ops::ScalarOp(11.0, PrimitiveType::F32);
  ir::NodePtr scalar6 = ir::ops::ScalarOp(22.0, PrimitiveType::F32);
  ir::Value add3 = scalar5 + scalar6;

  EXPECT_EQ(add1->hash(), add2->hash());
  EXPECT_NE(add1->hash(), add3->hash());

  ir::Value sub = scalar1 - scalar2;

  EXPECT_NE(add1->hash(), sub->hash());
}

}  // namespace cpp_test
}  // namespace torch_lazy_tensors
