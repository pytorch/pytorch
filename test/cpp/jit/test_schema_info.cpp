#include <gtest/gtest.h>
#include <torch/csrc/utils/schema_info.h>

namespace torch {
namespace utils {
namespace {
TEST(FunctionSchemaIsMutableTest, Basic) {
  c10::FunctionSchema schema =
      torch::jit::getOperatorForLiteral(
          "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))")
          ->schema();
  ASSERT_TRUE(schema.is_mutable(0));
  ASSERT_TRUE(schema.is_mutable("self"));
  ASSERT_FALSE(schema.is_mutable(1));
  ASSERT_FALSE(schema.is_mutable("other"));
  ASSERT_FALSE(schema.is_mutable(2));
  ASSERT_FALSE(schema.is_mutable("alpha"));
}

TEST(FunctionSchemaIsMutableTest, InvalidArgument) {
  c10::FunctionSchema schema =
      torch::jit::getOperatorForLiteral(
          "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))")
          ->schema();
  ASSERT_THROW(schema.is_mutable(4), c10::Error);
  ASSERT_THROW(schema.is_mutable("named_argument"), c10::Error);
}

TEST(FunctionSchemaAreAliasingTest, Basic) {
  c10::FunctionSchema schema =
      torch::jit::getOperatorForLiteral(
          "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))")
          ->schema();
  ASSERT_TRUE(schema.areAliasing({c10::input, 0}, {c10::output, 0}));
  ASSERT_FALSE(schema.areAliasing({c10::input, 1}, {c10::output, 0}));
  ASSERT_FALSE(schema.areAliasing({c10::input, 1}, {c10::input, 0}));
}

TEST(FunctionSchemaAreAliasingTest, InvalidArgument) {
  c10::FunctionSchema schema =
      torch::jit::getOperatorForLiteral(
          "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))")
          ->schema();
  ASSERT_THROW(
      schema.areAliasing({c10::input, 15}, {c10::output, 0}), c10::Error);
  ASSERT_THROW(
      schema.areAliasing({c10::input, 0}, {c10::output, 15}), c10::Error);
}

TEST(FunctionSchemaAreAliasingTest, Wildcard) {
  c10::FunctionSchema schema =
      torch::jit::getOperatorForLiteral(
          "aten::split.Tensor(Tensor(a -> *) self, int split_size, int dim=0) -> Tensor(a)[]")
          ->schema();
  ASSERT_TRUE(schema.areAliasing({c10::input, 0}, {c10::output, 0}, true));
  ASSERT_FALSE(schema.areAliasing({c10::input, 0}, {c10::output, 0}, false));
}
} // namespace
} // namespace utils
} // namespace torch
