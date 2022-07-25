#include <ATen/core/dispatch/Dispatcher.h>
#include <gtest/gtest.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/utils/alias_map.h>
#include <torch/csrc/utils/schema_info.h>
#include <chrono>

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

TEST(AliasMapGenerationTest, Basic) {
  AliasMap alias_map(torch::jit::parseSchema(
      "aten::test.Tensor(Tensor(a) self, Tensor(b!) other, Tensor more_other) -> (Tensor(a), Tensor(b!))"));
  std::vector<std::unordered_set<size_t>> expected_alias_map = {{0}, {1}};
  std::unordered_set<size_t> expected_aliasing_inputs = {0, 1};
  std::unordered_set<size_t> expected_aliasing_outputs = {0, 1};
  std::unordered_set<size_t> expected_const_aliasing_inputs = {0};
  std::unordered_set<size_t> expected_const_aliasing_outputs = {0};
  std::unordered_set<size_t> expected_mutating_inputs = {1};
  std::unordered_set<size_t> expected_mutating_outputs = {1};
  ASSERT_EQ(expected_alias_map, alias_map.alias_map());
  ASSERT_EQ(expected_aliasing_inputs, alias_map.aliasing_inputs());
  ASSERT_EQ(expected_aliasing_outputs, alias_map.aliasing_outputs());
  ASSERT_EQ(expected_const_aliasing_inputs, alias_map.const_aliasing_inputs());
  ASSERT_EQ(
      expected_const_aliasing_outputs, alias_map.const_aliasing_outputs());
  ASSERT_EQ(expected_mutating_inputs, alias_map.mutating_inputs());
  ASSERT_EQ(expected_mutating_outputs, alias_map.mutating_outputs());
}

TEST(AliasMapGenerationTest, AlreadyBoundAlias) {
  AliasMap alias_map(torch::jit::parseSchema(
      "aten::test.Tensor(Tensor(a) self, Tensor(b!) other, Tensor(a) more_other) -> (Tensor(a), Tensor(b!))"));
  std::vector<std::unordered_set<size_t>> expected_alias_map = {{0}, {1}};
  std::unordered_set<size_t> expected_aliasing_inputs = {0, 1, 2};
  std::unordered_set<size_t> expected_aliasing_outputs = {0, 1};
  std::unordered_set<size_t> expected_const_aliasing_inputs = {0, 2};
  std::unordered_set<size_t> expected_const_aliasing_outputs = {0};
  std::unordered_set<size_t> expected_mutating_inputs = {1};
  std::unordered_set<size_t> expected_mutating_outputs = {1};
  ASSERT_EQ(expected_alias_map, alias_map.alias_map());
  ASSERT_EQ(expected_aliasing_inputs, alias_map.aliasing_inputs());
  ASSERT_EQ(expected_aliasing_outputs, alias_map.aliasing_outputs());
  ASSERT_EQ(expected_const_aliasing_inputs, alias_map.const_aliasing_inputs());
  ASSERT_EQ(
      expected_const_aliasing_outputs, alias_map.const_aliasing_outputs());
  ASSERT_EQ(expected_mutating_inputs, alias_map.mutating_inputs());
  ASSERT_EQ(expected_mutating_outputs, alias_map.mutating_outputs());
}

TEST(AliasMapSpeedTest, SpeedTest) {
  const auto approach1 = [](const c10::OperatorHandle& op) {
    const auto& schema = op.schema();
    const auto& op_name = schema.operator_name().name;
    const auto num_arguments = schema.arguments().size();
    const auto num_returns = schema.returns().size();
    // Keep track of which outputs are output of in-place modification
    // so we can rebase_history if necessary
    std::vector<bool> is_inplace_output(num_returns, false);
    bool any_is_inplace_output = false;
    std::vector<bool> is_aliased_output(num_returns, false);
    int aliased_output_idx = -1;

    for (const auto i : c10::irange(num_returns)) {
      if (schema.is_aliasing({c10::SchemaArgType::output, i})) {
        if (schema.is_mutable({c10::SchemaArgType::output, i})) {
          is_inplace_output[i] = true;
          any_is_inplace_output = true;
        } else {
          TORCH_CHECK(
              aliased_output_idx == -1,
              "Expected only a single output in the operator schema to have a non-write alias annotation (i.e., 'Tensor(a)'). "
              "Non-composite functions where multiple outputs are aliased with inputs aren't supported."
              "Please rewrite your function as a composite function.");
          aliased_output_idx = i;
        }
        is_aliased_output[i] = true;
      }
    }

    int aliased_input_idx = -1;
    for (const auto i : c10::irange(num_arguments)) {
      if (schema.is_aliasing({c10::SchemaArgType::input, i}) &&
          !schema.is_mutable({c10::SchemaArgType::input, i})) {
        TORCH_CHECK(
            aliased_input_idx == -1,
            "Expected only a single input in the operator schema to have a non-write alias annotation (i.e., 'Tensor(a)'). "
            "Non-composite functions where multiple inputs are aliased with outputs aren't supported. "
            "Please rewrite your function as a composite function.");
        aliased_input_idx = i;
      }
    }
  };

  const auto approach2 = [](const c10::OperatorHandle& op) {
    AliasMap m(op.schema());
    TORCH_CHECK(
        m.const_aliasing_inputs().size() <= 1,
        "Expected only a single input in the operator schema to have a non-write alias annotation (i.e., 'Tensor(a)'). "
        "Non-composite functions where multiple inputs are aliased with outputs aren't supported."
        "Please rewrite your function as a composite function.");
    TORCH_CHECK(
        m.const_aliasing_outputs().size() <= 1,
        "Expected only a single output in the operator schema to have a non-write alias annotation (i.e., 'Tensor(a)'). "
        "Non-composite functions where multiple outputs are aliased with inputs aren't supported."
        "Please rewrite your function as a composite function.");
  };

  std::unordered_map<c10::OperatorName, AliasMap> mm;
  const auto approach3 = [&mm](const c10::OperatorHandle& op) {
    if (mm.count(op.operator_name())) {
      const AliasMap& m = mm[op.operator_name()];
      TORCH_CHECK(
          m.const_aliasing_inputs().size() <= 1,
          "Expected only a single input in the operator schema to have a non-write alias annotation (i.e., 'Tensor(a)'). "
          "Non-composite functions where multiple inputs are aliased with outputs aren't supported."
          "Please rewrite your function as a composite function.");
      TORCH_CHECK(
          m.const_aliasing_outputs().size() <= 1,
          "Expected only a single output in the operator schema to have a non-write alias annotation (i.e., 'Tensor(a)'). "
          "Non-composite functions where multiple outputs are aliased with inputs aren't supported."
          "Please rewrite your function as a composite function.");
    } else {
      AliasMap m(op.schema());
      TORCH_CHECK(
          m.const_aliasing_inputs().size() <= 1,
          "Expected only a single input in the operator schema to have a non-write alias annotation (i.e., 'Tensor(a)'). "
          "Non-composite functions where multiple inputs are aliased with outputs aren't supported."
          "Please rewrite your function as a composite function.");
      TORCH_CHECK(
          m.const_aliasing_outputs().size() <= 1,
          "Expected only a single output in the operator schema to have a non-write alias annotation (i.e., 'Tensor(a)'). "
          "Non-composite functions where multiple outputs are aliased with inputs aren't supported."
          "Please rewrite your function as a composite function.");
      mm[op.operator_name()] = m;
    }
  };
  const std::vector<c10::OperatorName> names = {
      {"aten::normal", "Tensor_float_out"},
      {"aten::bartlett_window", "periodic"},
      {"aten::normal", "Tensor_float_out"},
      {"aten::bartlett_window", "periodic"},
      {"aten::normal", "Tensor_float_out"},
      {"aten::bartlett_window", "periodic"},
      {"aten::normal", "Tensor_float_out"}};
  for (const auto& name : names) {
    std::cout << name << '\n';
    const auto& handle = *c10::Dispatcher::singleton().findOp(name);
    auto start = std::chrono::high_resolution_clock::now();
    approach2(handle);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "approach 1 elapsed: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(
                     end - start)
                     .count()
              << " nanoseconds.\n";
    start = std::chrono::high_resolution_clock::now();
    approach2(handle);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "approach 2 elapsed: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(
                     end - start)
                     .count()
              << " nanoseconds.\n";

    start = std::chrono::high_resolution_clock::now();
    approach3(handle);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "approach 3 elapsed: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(
                     end - start)
                     .count()
              << " nanoseconds.\n";
  }
  ASSERT_FALSE(true);
}

} // namespace utils
} // namespace torch
