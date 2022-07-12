#include <gtest/gtest.h>
#include <torch/csrc/utils/schema_info.h>

namespace torch {
namespace utils {
TEST(FunctionSchemaIsMutableTest, Basic) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_TRUE(schema.is_mutable(0));
  ASSERT_TRUE(schema.is_mutable("self"));
  ASSERT_FALSE(schema.is_mutable(1));
  ASSERT_FALSE(schema.is_mutable("other"));
  ASSERT_FALSE(schema.is_mutable(2));
  ASSERT_FALSE(schema.is_mutable("alpha"));
}

TEST(FunctionSchemaIsMutableTest, InvalidArgument) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_THROW(schema.is_mutable(4), c10::Error);
  ASSERT_THROW(schema.is_mutable("named_argument"), c10::Error);
}

TEST(FunctionSchemaMayAliasTest, Basic) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_TRUE(schema.may_alias(
      {c10::SchemaArgType::input, 0}, {c10::SchemaArgType::output, 0}));
  ASSERT_FALSE(schema.may_alias(
      {c10::SchemaArgType::input, 1}, {c10::SchemaArgType::output, 0}));
  ASSERT_FALSE(schema.may_alias(
      {c10::SchemaArgType::input, 1}, {c10::SchemaArgType::input, 0}));
}

TEST(FunctionSchemaMayAliasTest, InvalidArgument) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_THROW(
      schema.may_alias(
          {c10::SchemaArgType::input, 15}, {c10::SchemaArgType::output, 0}),
      c10::Error);
  ASSERT_THROW(
      schema.may_alias(
          {c10::SchemaArgType::input, 0}, {c10::SchemaArgType::output, 15}),
      c10::Error);
}

TEST(FunctionSchemaMayAliasTest, Wildcard) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::test.Tensor(Tensor(*) self) -> (Tensor(*), Tensor)");
  ASSERT_TRUE(schema.may_alias(
      {c10::SchemaArgType::output, 0}, {c10::SchemaArgType::input, 0}));
  ASSERT_FALSE(schema.may_alias(
      {c10::SchemaArgType::output, 1}, {c10::SchemaArgType::input, 0}));
}
} // namespace utils
} // namespace torch
