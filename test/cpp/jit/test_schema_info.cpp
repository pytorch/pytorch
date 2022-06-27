#include <gtest/gtest.h>
#include <torch/csrc/utils/schema_info.h>

namespace torch {
namespace utils {

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
      "aten::bernoulli(Tensor self, *, Generator? generator) -> Tensor");
  ASSERT_TRUE(deterministic_schema_info.isDeterministic());
  ASSERT_FALSE(nondeterministic_schema_info.isDeterministic());
}

TEST(SchemaInfoIsDeterministicTest, Dropout) {
  SchemaInfo droupout_schema_info(
      "aten::dropout(Tensor input, float p, bool train) -> Tensor");
  ASSERT_FALSE(droupout_schema_info.isDeterministic());
  droupout_schema_info.addArgumentValue("train", false);
  ASSERT_TRUE(droupout_schema_info.isDeterministic());
}

TEST(SchemaInfoIsMutableTest, Basic) {
  SchemaInfo schema_info(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_TRUE(schema_info.isMutating(0));
  ASSERT_TRUE(schema_info.isMutating("self"));
  ASSERT_FALSE(schema_info.isMutating(1));
  ASSERT_FALSE(schema_info.isMutating("other"));
}

TEST(SchemaInfoIsMutableTest, InvalidArgument) {
  SchemaInfo schema_info(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_THROW(schema_info.isMutating(-1), c10::Error);
  ASSERT_THROW(schema_info.isMutating(4), c10::Error);
}

TEST(SchemaInfoIsMutableTest, InstanceNorm) {
  SchemaInfo schema_info(
      "aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor");
  ASSERT_FALSE(schema_info.isMutating("running_mean"));
  ASSERT_FALSE(schema_info.isMutating("running_var"));
  schema_info.addArgumentValue("use_input_stats", true);
  ASSERT_TRUE(schema_info.isMutating("running_mean"));
  ASSERT_TRUE(schema_info.isMutating("running_var"));
}

TEST(SchemaInfoIsMutableTest, BatchNorm) {
  SchemaInfo schema_info(
      "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor");
  ASSERT_FALSE(schema_info.isMutating("running_mean"));
  ASSERT_FALSE(schema_info.isMutating("running_var"));
  schema_info.addArgumentValue("training", true);
  ASSERT_TRUE(schema_info.isMutating("running_mean"));
  ASSERT_TRUE(schema_info.isMutating("running_var"));
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
} // namespace utils
} // namespace torch
