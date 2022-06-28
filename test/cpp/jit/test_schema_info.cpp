#include <gtest/gtest.h>
#include <torch/csrc/utils/schema_info.h>

namespace torch {
namespace utils {
namespace {

TEST(SchemaInfoHasSideEffectsTest, Basic) {
  SchemaInfo no_side_effects_schema_info(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  SchemaInfo side_effects_schema_info(
      "aten::warn(str message, int stacklevel=2) -> ()");
  ASSERT_TRUE(side_effects_schema_info.hasSideEffects());
  ASSERT_FALSE(no_side_effects_schema_info.hasSideEffects());
}

TEST(SchemaInfoIsDeterministicTest, Basic) {
  SchemaInfo deterministic_schema_info(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  SchemaInfo nondeterministic_schema_info(
      "aten::dropout(Tensor input, float p, bool train) -> Tensor");
  ASSERT_TRUE(deterministic_schema_info.isDeterministic());
  ASSERT_FALSE(nondeterministic_schema_info.isDeterministic());
}

TEST(SchemaInfoIsMutableTest, Basic) {
  SchemaInfo schema_info(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_TRUE(schema_info.isMutating(0));
  ASSERT_TRUE(schema_info.isMutating("self"));
  ASSERT_FALSE(schema_info.isMutating(1));
  ASSERT_FALSE(schema_info.isMutating("other"));
  ASSERT_FALSE(schema_info.isMutating(2));
  ASSERT_FALSE(schema_info.isMutating("alpha"));
}

TEST(SchemaInfoIsMutableTest, InvalidArgument) {
  SchemaInfo schema_info(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_THROW(schema_info.isMutating(-1), c10::Error);
  ASSERT_THROW(schema_info.isMutating(4), c10::Error);
  ASSERT_THROW(schema_info.isMutating("named_argument"), c10::Error);
}

TEST(SchemaInfoAreAliasingTest, Basic) {
  SchemaInfo schema_info(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_TRUE(schema_info.areAliasing(
      {SchemaArgType::input, 0}, {SchemaArgType::output, 0}));
  ASSERT_FALSE(schema_info.areAliasing(
      {SchemaArgType::input, 1}, {SchemaArgType::output, 0}));
  ASSERT_FALSE(schema_info.areAliasing(
      {SchemaArgType::input, 1}, {SchemaArgType::input, 0}));
}

TEST(SchemaInfoAreAliasingTest, InvalidArgument) {
  SchemaInfo schema_info(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
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
  SchemaInfo schema_info(
      "aten::split.Tensor(Tensor(a -> *) self, int split_size, int dim=0) -> Tensor(a)[]");
  ASSERT_TRUE(schema_info.areAliasing(
      {SchemaArgType::input, 0}, {SchemaArgType::output, 0}));
}
} // namespace
} // namespace utils
} // namespace torch
