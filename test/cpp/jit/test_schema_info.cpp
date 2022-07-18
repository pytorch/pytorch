#include <gtest/gtest.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/utils/schema_info.h>

namespace torch {
namespace utils {
TEST(SchemaInfoHasSideEffectsTest, Basic) {
  SchemaInfo no_side_effects_schema_info(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  SchemaInfo side_effects_schema_info(
      "aten::warn(str message, int stacklevel=2) -> ()");
  ASSERT_TRUE(side_effects_schema_info.has_side_effects());
  ASSERT_FALSE(no_side_effects_schema_info.has_side_effects());
}

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

TEST(SchemaInfoIsMutableTest, Basic) {
  SchemaInfo schema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_TRUE(schema.is_mutable(0));
  ASSERT_TRUE(schema.is_mutable("self"));
  ASSERT_FALSE(schema.is_mutable(1));
  ASSERT_FALSE(schema.is_mutable("other"));
  ASSERT_FALSE(schema.is_mutable(2));
  ASSERT_FALSE(schema.is_mutable("alpha"));
}

TEST(SchemaInfoIsMutableTest, InvalidArgument) {
  SchemaInfo schema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_THROW(schema.is_mutable(4), c10::Error);
  ASSERT_THROW(schema.is_mutable("named_argument"), c10::Error);
}

TEST(SchemaInfoIsMutableTest, AliasingInputs) {
  SchemaInfo schema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_TRUE(schema.is_mutable(0));
  ASSERT_TRUE(schema.is_mutable("self"));
  ASSERT_FALSE(schema.is_mutable(1));
  ASSERT_FALSE(schema.is_mutable("other"));
  at::Tensor input = at::randn({3, 3});
  schema.addArgumentValue("self", input);
  schema.addArgumentValue("other", input);
  ASSERT_TRUE(schema.is_mutable(1));
  ASSERT_TRUE(schema.is_mutable("other"));
}

TEST(SchemaInfoIsMutableTest, InstanceNorm) {
  SchemaInfo schema_info(
      "aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor");
  ASSERT_FALSE(schema_info.is_mutable("running_mean"));
  ASSERT_FALSE(schema_info.is_mutable("running_var"));
  schema_info.addArgumentValue("use_input_stats", true);
  ASSERT_TRUE(schema_info.is_mutable("running_mean"));
  ASSERT_TRUE(schema_info.is_mutable("running_var"));
}

TEST(SchemaInfoIsMutableTest, BatchNorm) {
  SchemaInfo schema_info(
      "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor");
  ASSERT_FALSE(schema_info.is_mutable("running_mean"));
  ASSERT_FALSE(schema_info.is_mutable("running_var"));
  schema_info.addArgumentValue("training", true);
  ASSERT_TRUE(schema_info.is_mutable("running_mean"));
  ASSERT_TRUE(schema_info.is_mutable("running_var"));
}

TEST(SchemaInfoIsNonDeterministicTest, Basic) {
  SchemaInfo deterministic_schema_info(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  SchemaInfo nondeterministic_schema_info(
      "aten::bernoulli(Tensor self, *, Generator? generator) -> Tensor");
  ASSERT_FALSE(deterministic_schema_info.is_nondeterministic());
  ASSERT_TRUE(nondeterministic_schema_info.is_nondeterministic());
}

TEST(SchemaInfoIsNonDeterministicTest, Dropout) {
  SchemaInfo droupout_schema_info(
      "aten::dropout(Tensor input, float p, bool train) -> Tensor");
  ASSERT_TRUE(droupout_schema_info.is_nondeterministic());
  droupout_schema_info.addArgumentValue("train", false);
  ASSERT_FALSE(droupout_schema_info.is_nondeterministic());
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

TEST(SchemaInfoMayAliasTest, AliasingInputs) {
  SchemaInfo schema(
      "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor");
  ASSERT_FALSE(schema.may_alias(
      {c10::SchemaArgType::input, 0}, {c10::SchemaArgType::input, 1}));
  at::Tensor input = at::randn({3, 3});
  schema.addArgumentValue("self", input);
  schema.addArgumentValue("other", input);
  ASSERT_TRUE(schema.may_alias(
      {c10::SchemaArgType::input, 0}, {c10::SchemaArgType::input, 1}));
}

TEST(SchemaInfoMayAliasTest, AliasingOutputs) {
  SchemaInfo schema(
      "aten::aminmax.out(Tensor self, *, int? dim=None, bool keepdim=False, Tensor(a!) min, Tensor(b!) max) -> (Tensor(a!) min, Tensor(b!) max)");
  ASSERT_FALSE(schema.may_alias(
      {c10::SchemaArgType::output, 0}, {c10::SchemaArgType::output, 1}));
  at::Tensor input = at::randn({3, 3});
  schema.addArgumentValue("min", input);
  schema.addArgumentValue("max", input);
  ASSERT_TRUE(schema.may_alias(
      {c10::SchemaArgType::output, 0}, {c10::SchemaArgType::output, 1}));
}

TEST(SchemaInfoMayAliasTest, AliasingInputOutput) {
  SchemaInfo schema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_TRUE(schema.may_alias(
      {c10::SchemaArgType::input, 0}, {c10::SchemaArgType::output, 0}));
  ASSERT_FALSE(schema.may_alias(
      {c10::SchemaArgType::input, 1}, {c10::SchemaArgType::output, 0}));
  at::Tensor input = at::randn({3, 3});
  schema.addArgumentValue("self", input);
  schema.addArgumentValue("other", input);
  ASSERT_TRUE(schema.may_alias(
      {c10::SchemaArgType::input, 0}, {c10::SchemaArgType::output, 0}));
  ASSERT_TRUE(schema.may_alias(
      {c10::SchemaArgType::input, 1}, {c10::SchemaArgType::output, 0}));
}

TEST(FunctionSchemaMayContainAliasTest, Basic) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_TRUE(schema.may_contain_alias(
      {c10::SchemaArgType::input, 0}, {c10::SchemaArgType::output, 0}));
  ASSERT_FALSE(schema.may_contain_alias(
      {c10::SchemaArgType::input, 1}, {c10::SchemaArgType::output, 0}));
  ASSERT_FALSE(schema.may_contain_alias(
      {c10::SchemaArgType::input, 1}, {c10::SchemaArgType::input, 0}));
}

TEST(FunctionSchemaMayContainAliasTest, Wildcard) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::test.Tensor(Tensor(*) self) -> (Tensor[], Tensor)");
  ASSERT_FALSE(schema.may_alias(
      {c10::SchemaArgType::output, 0}, {c10::SchemaArgType::input, 0}));
  ASSERT_TRUE(schema.may_contain_alias(
      {c10::SchemaArgType::output, 0}, {c10::SchemaArgType::input, 0}));
  ASSERT_TRUE(schema.may_contain_alias(
      {c10::SchemaArgType::output, 0}, {c10::SchemaArgType::input, 0}, false));
  ASSERT_FALSE(schema.may_contain_alias(
      {c10::SchemaArgType::input, 0}, {c10::SchemaArgType::output, 0}, false));
  ASSERT_FALSE(schema.may_alias(
      {c10::SchemaArgType::output, 1}, {c10::SchemaArgType::input, 0}));
}

TEST(FunctionSchemaMayContainAliasTest, InputAndOutputContainers) {
  c10::FunctionSchema schema =
      torch::jit::parseSchema("aten::test.Tensor(Tensor[] self) -> Tensor[]");
  ASSERT_FALSE(schema.may_alias(
      {c10::SchemaArgType::output, 0}, {c10::SchemaArgType::input, 0}));
  ASSERT_TRUE(schema.may_contain_alias(
      {c10::SchemaArgType::output, 0}, {c10::SchemaArgType::input, 0}));
  ASSERT_TRUE(schema.may_contain_alias(
      {c10::SchemaArgType::output, 0}, {c10::SchemaArgType::input, 0}, false));
  ASSERT_TRUE(schema.may_contain_alias(
      {c10::SchemaArgType::input, 0}, {c10::SchemaArgType::output, 0}, false));
}
} // namespace utils
} // namespace torch
