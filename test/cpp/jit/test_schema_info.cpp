#include <gtest/gtest.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/utils/schema_info.h>

namespace torch {
namespace utils {

const c10::FunctionSchema getSchema(const char* name) {
  return torch::jit::getOperatorForLiteral(name)->schema();
}

TEST(SchemaInfoIsMutableTest, Basic) {
  auto schema = getSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  SchemaInfo schema_info(schema);
  ASSERT_TRUE(schema_info.isMutating(0));
  ASSERT_TRUE(schema_info.isMutating("self"));
  ASSERT_FALSE(schema_info.isMutating(1));
  ASSERT_FALSE(schema_info.isMutating("other"));
}

TEST(SchemaInfoIsMutableTest, InvalidArgument) {
  auto schema = getSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  SchemaInfo schema_info(schema);
  ASSERT_THROW(schema_info.isMutating(-1), c10::Error);
  ASSERT_THROW(schema_info.isMutating(4), c10::Error);
}

TEST(SchemaInfoAreAliasingTest, Basic) {
  auto schema = getSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  SchemaInfo schema_info(schema);

  ASSERT_TRUE(schema_info.areAliasing(
      {SchemaArgType::input, 0}, {SchemaArgType::output, 0}));
  ASSERT_FALSE(schema_info.areAliasing(
      {SchemaArgType::input, 1}, {SchemaArgType::output, 0}));
  ASSERT_FALSE(schema_info.areAliasing(
      {SchemaArgType::input, 1}, {SchemaArgType::input, 0}));
}

TEST(SchemaInfoAreAliasingTest, InvalidArgument) {
  auto schema = getSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  SchemaInfo schema_info(schema);
  ASSERT_THROW(
      schema_info.areAliasing(
          {SchemaArgType::input, -1}, {SchemaArgType::output, 0}),
      c10::Error);
  ASSERT_THROW(
      schema_info.areAliasing(
          {SchemaArgType::input, 0}, {SchemaArgType::output, -1}),
      c10::Error);
}

TEST(SchemaInfoAreAliasingTest, Wildcard) {
  auto schema = getSchema(
      "aten::split.Tensor(Tensor(a -> *) self, int split_size, int dim=0) -> Tensor(a)[]");
  SchemaInfo schema_info(schema);
  ASSERT_TRUE(schema_info.areAliasing(
      {SchemaArgType::input, 0}, {SchemaArgType::output, 0}));
}
} // namespace utils
} // namespace torch
