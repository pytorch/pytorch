#include <gtest/gtest.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/utils/schema_info.h>

namespace torch {
namespace utils {
using c10::SchemaArgType;

TEST(FunctionSchemaIsAliasingTest, Basic) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::test.Tensor(Tensor(a) self, Tensor(b!) other, Tensor more_other) -> (Tensor(a), Tensor(b!))");
  ASSERT_TRUE(schema.is_aliasing({SchemaArgType::output, 0}));
  ASSERT_TRUE(schema.is_aliasing({SchemaArgType::output, 1}));
  ASSERT_TRUE(schema.is_aliasing({SchemaArgType::input, 0}));
  ASSERT_TRUE(schema.is_aliasing({SchemaArgType::input, 1}));
  ASSERT_FALSE(schema.is_aliasing({SchemaArgType::input, 2}));
}

TEST(FunctionSchemaIsAliasingTest, InvalidArgument) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_THROW(schema.is_aliasing({SchemaArgType::input, 4}), c10::Error);
  ASSERT_THROW(schema.is_aliasing({SchemaArgType::output, 4}), c10::Error);
}

TEST(FunctionSchemaIsMutableTest, Basic) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_TRUE(schema.is_mutable({SchemaArgType::output, 0}));
  ASSERT_TRUE(schema.is_mutable({SchemaArgType::input, 0}));
  ASSERT_TRUE(schema.is_mutable("self"));
  ASSERT_FALSE(schema.is_mutable({SchemaArgType::input, 1}));
  ASSERT_FALSE(schema.is_mutable("other"));
  ASSERT_FALSE(schema.is_mutable({SchemaArgType::input, 2}));
  ASSERT_FALSE(schema.is_mutable("alpha"));
}

TEST(FunctionSchemaIsMutableTest, InvalidArgument) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_THROW(schema.is_mutable({SchemaArgType::input, 4}), c10::Error);
  ASSERT_THROW(schema.is_mutable({SchemaArgType::output, 4}), c10::Error);
  ASSERT_THROW(schema.is_mutable("named_argument"), c10::Error);
}

TEST(SchemaInfoIsMutableTest, Basic) {
  SchemaInfo schema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_TRUE(schema.is_mutable({SchemaArgType::input, 0}));
  ASSERT_TRUE(schema.is_mutable("self"));
  ASSERT_FALSE(schema.is_mutable({SchemaArgType::input, 1}));
  ASSERT_FALSE(schema.is_mutable("other"));
  ASSERT_FALSE(schema.is_mutable({SchemaArgType::input, 2}));
  ASSERT_FALSE(schema.is_mutable("alpha"));
}

TEST(SchemaInfoIsMutableTest, InvalidArgument) {
  SchemaInfo schema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_THROW(schema.is_mutable({SchemaArgType::input, 4}), c10::Error);
  ASSERT_THROW(schema.is_mutable("named_argument"), c10::Error);
}

TEST(SchemaInfoIsMutableTest, AliasingInputs) {
  SchemaInfo schema(
      "aten::test.Tensor(Tensor(a!) self, Tensor(b) other, *, Scalar alpha=1) -> (Tensor(a!), Tensor(b))");
  ASSERT_TRUE(schema.is_mutable({SchemaArgType::input, 0}));
  ASSERT_TRUE(schema.is_mutable({SchemaArgType::output, 0}));
  ASSERT_TRUE(schema.is_mutable("self"));
  ASSERT_FALSE(schema.is_mutable({SchemaArgType::input, 1}));
  ASSERT_FALSE(schema.is_mutable({SchemaArgType::output, 1}));
  ASSERT_FALSE(schema.is_mutable("other"));
  at::Tensor input = at::randn({3, 3});
  schema.addArgumentValue("self", input);
  schema.addArgumentValue("other", input);
  ASSERT_TRUE(schema.is_mutable({SchemaArgType::input, 1}));
  ASSERT_TRUE(schema.is_mutable({SchemaArgType::output, 1}));
  ASSERT_TRUE(schema.is_mutable("other"));
}

TEST(SchemaInfoIsMutableTest, InstanceNorm) {
  SchemaInfo schema_info(
      "aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor");
  ASSERT_TRUE(schema_info.is_mutable("running_mean"));
  ASSERT_TRUE(schema_info.is_mutable("running_var"));
  schema_info.addArgumentValue("use_input_stats", false);
  ASSERT_FALSE(schema_info.is_mutable("running_mean"));
  ASSERT_FALSE(schema_info.is_mutable("running_var"));
}

TEST(SchemaInfoIsMutableTest, BatchNorm) {
  SchemaInfo schema_info(
      "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor");
  ASSERT_TRUE(schema_info.is_mutable("running_mean"));
  ASSERT_TRUE(schema_info.is_mutable("running_var"));
  schema_info.addArgumentValue("training", false);
  ASSERT_FALSE(schema_info.is_mutable("running_mean"));
  ASSERT_FALSE(schema_info.is_mutable("running_var"));
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
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output, 0}));
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::output, 0}));
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::input, 0}));
}

TEST(FunctionSchemaMayAliasTest, InvalidArgument) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_THROW(
      schema.may_alias({SchemaArgType::input, 15}, {SchemaArgType::output, 0}),
      c10::Error);
  ASSERT_THROW(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output, 15}),
      c10::Error);
}

TEST(FunctionSchemaMayAliasTest, Wildcard) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::test.Tensor(Tensor(*) self) -> (Tensor(*), Tensor)");
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::output, 0}, {SchemaArgType::input, 0}));
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::output, 1}, {SchemaArgType::input, 0}));
}

TEST(SchemaInfoMayAliasTest, AliasingInputs) {
  SchemaInfo schema(
      "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor");
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  at::Tensor input = at::randn({3, 3});
  schema.addArgumentValue("self", input);
  schema.addArgumentValue("other", input);
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
}

TEST(SchemaInfoMayAliasTest, AliasingOutputs) {
  SchemaInfo schema(
      "aten::aminmax.out(Tensor self, *, int? dim=None, bool keepdim=False, Tensor(a!) min, Tensor(b!) max) -> (Tensor(a!) min, Tensor(b!) max)");
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::output, 0}, {SchemaArgType::output, 1}));
  at::Tensor input = at::randn({3, 3});
  schema.addArgumentValue("min", input);
  schema.addArgumentValue("max", input);
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::output, 0}, {SchemaArgType::output, 1}));
}

TEST(SchemaInfoMayAliasTest, AliasingInputOutput) {
  SchemaInfo schema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output, 0}));
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::output, 0}));
  at::Tensor input = at::randn({3, 3});
  schema.addArgumentValue("self", input);
  schema.addArgumentValue("other", input);
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output, 0}));
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::output, 0}));
}

TEST(SchemaInfoMayAliasTest, MultipleWildcardInputs) {
  SchemaInfo schema(
      "aten::test.Tensor(Tensor(a) a, Tensor(*) b, Tensor(*) c) -> (Tensor(a), Tensor(*))");
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output, 0}));
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::output, 1}));
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 2}, {SchemaArgType::output, 1}));
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input, 2}));
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output, 1}));
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::output, 0}));
  at::Tensor input = at::randn({3, 3});
  schema.addArgumentValue("a", input);
  schema.addArgumentValue("b", input);
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output, 0}));
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::output, 1}));
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 2}, {SchemaArgType::output, 1}));
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input, 2}));
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output, 1}));
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::output, 0}));
}

TEST(SchemaInfoMayAliasTest, MultipleNonWildcardInputs) {
  SchemaInfo schema(
      "aten::test.Tensor(Tensor(a) a, Tensor(a) b, Tensor(*) c, Tensor(b) d) -> (Tensor(a), Tensor(*))");
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input, 2}));
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 2}, {SchemaArgType::input, 1}));
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 2}, {SchemaArgType::output, 0}));
}

TEST(SchemaInfoMayAliasTest, MultipleNonWildcardOutputs) {
  SchemaInfo schema(
      "aten::test.Tensor(Tensor(a) a, Tensor(*) b) -> (Tensor(a), Tensor(a))");
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::output, 0}, {SchemaArgType::output, 1}));
  ASSERT_TRUE(
      schema.may_alias({SchemaArgType::output, 0}, {SchemaArgType::input, 1}));
}

TEST(SchemaInfoMayAliasTest, MismatchingTypes) {
  SchemaInfo schema("aten::test.Tensor(Tensor(a) a) -> int(a)");
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output, 0}));
}

TEST(FunctionSchemaMayContainAliasTest, Basic) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::output, 0}));
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::input, 1}, {SchemaArgType::output, 0}));
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::input, 1}, {SchemaArgType::input, 0}));
}

TEST(FunctionSchemaMayContainAliasTest, Wildcard) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::test.Tensor(Tensor(*) self) -> (Tensor[], Tensor)");
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::output, 0}, {SchemaArgType::input, 0}));
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 0}));
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 0}, false));
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::output, 0}, false));
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::output, 1}, {SchemaArgType::input, 0}));
}

TEST(FunctionSchemaMayContainAliasTest, InputAndOutputContainers) {
  c10::FunctionSchema schema =
      torch::jit::parseSchema("aten::test.Tensor(Tensor[] self) -> Tensor[]");
  ASSERT_FALSE(
      schema.may_alias({SchemaArgType::output, 0}, {SchemaArgType::input, 0}));
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 0}));
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 0}, false));
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::output, 0}, false));
}

TEST(SchemaInfoMayContainAliasTest, ContainAliasInputsEqual) {
  SchemaInfo schema(
      "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor");
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  at::Tensor input = at::randn({3, 3});
  schema.addArgumentValue("self", input);
  schema.addArgumentValue("other", input);
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 1}, false));
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 1}, {SchemaArgType::input, 0}, false));
}

TEST(SchemaInfoMayContainAliasTest, ContainAliasInputsContained) {
  SchemaInfo schema(
      "aten::test.Tensor(Tensor[] self, Tensor other, *, Scalar alpha=1) -> Tensor");
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  at::Tensor input = at::randn({3, 3});
  schema.addArgumentValue("self", c10::List<at::Tensor>({input}));
  schema.addArgumentValue("other", input);
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 1}, false));
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::input, 1}, {SchemaArgType::input, 0}, false));
}

TEST(SchemaInfoMayContainAliasTest, ContainAliasOutputs) {
  SchemaInfo schema(
      "aten::aminmax.out(Tensor self, *, int? dim=None, bool keepdim=False, Tensor(a!) min, Tensor(b!) max) -> (Tensor(a!) min, Tensor(b!) max)");
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::output, 1}));
  at::Tensor input = at::randn({3, 3});
  schema.addArgumentValue("min", input);
  schema.addArgumentValue("max", input);
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::output, 1}));
}

TEST(SchemaInfoMayContainAliasTest, ContainAliasInputOutput) {
  SchemaInfo schema(
      "aten::test.tensor(Tensor(a) self, Tensor[] other) -> Tensor(a)");
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 1}));
  at::Tensor input = at::randn({3, 3});
  schema.addArgumentValue("other", c10::List<at::Tensor>({input}));
  schema.addArgumentValue("self", input);
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 1}));
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 1}, false));
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 1}, {SchemaArgType::output, 0}, false));
}

TEST(SchemaInfoMayContainAliasTest, InputAndOutputContainers) {
  SchemaInfo schema(
      "aten::test.tensor(Tensor self, Tensor[] other) -> Tensor[]");
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 1}));
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 0}));
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  at::Tensor input = at::randn({3, 3});
  schema.addArgumentValue("other", c10::List<at::Tensor>({input}));
  schema.addArgumentValue("self", input);
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 1}));
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::output, 0}, {SchemaArgType::input, 0}));
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
}

TEST(SchemaInfoMayContainAliasTest, Wildcard) {
  SchemaInfo schema(
      "aten::test.tensor(Tensor a, Tensor[] b, Tensor(*) c) -> Tensor[]");
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 2}));
  ASSERT_FALSE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 2}, {SchemaArgType::input, 1}));
  at::Tensor input = at::randn({3, 3});
  schema.addArgumentValue("b", c10::List<at::Tensor>({input}));
  schema.addArgumentValue("a", input);
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 2}));
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
  ASSERT_TRUE(schema.may_contain_alias(
      {SchemaArgType::input, 2}, {SchemaArgType::input, 1}));
}
} // namespace utils
} // namespace torch
