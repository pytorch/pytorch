#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>
#include <torch/jit.h>

namespace torch {
namespace jit {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(CustomOperatorTest, InferredSchema) {
  torch::RegisterOperators reg(
      "foo::bar", [](double a, at::Tensor b) { return a + b; });
  auto& ops = getAllOperatorsFor(Symbol::fromQualString("foo::bar"));
  ASSERT_EQ(ops.size(), 1);

  auto& op = ops.front();
  ASSERT_EQ(op->schema().name(), "foo::bar");

  ASSERT_EQ(op->schema().arguments().size(), 2);
  ASSERT_EQ(op->schema().arguments()[0].name(), "_0");
  ASSERT_EQ(op->schema().arguments()[0].type()->kind(), TypeKind::FloatType);
  ASSERT_EQ(op->schema().arguments()[1].name(), "_1");
  ASSERT_EQ(op->schema().arguments()[1].type()->kind(), TypeKind::TensorType);

  ASSERT_EQ(op->schema().returns()[0].type()->kind(), TypeKind::TensorType);

  Stack stack;
  push(stack, 2.0f, at::ones(5));
  op->getOperation()(&stack);
  at::Tensor output;
  pop(stack, output);

  ASSERT_TRUE(output.allclose(at::full(5, 3.0f)));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(CustomOperatorTest, ExplicitSchema) {
  torch::RegisterOperators reg(
      "foo::bar_with_schema(float a, Tensor b) -> Tensor",
      [](double a, at::Tensor b) { return a + b; });

  auto& ops =
      getAllOperatorsFor(Symbol::fromQualString("foo::bar_with_schema"));
  ASSERT_EQ(ops.size(), 1);

  auto& op = ops.front();
  ASSERT_EQ(op->schema().name(), "foo::bar_with_schema");

  ASSERT_EQ(op->schema().arguments().size(), 2);
  ASSERT_EQ(op->schema().arguments()[0].name(), "a");
  ASSERT_EQ(op->schema().arguments()[0].type()->kind(), TypeKind::FloatType);
  ASSERT_EQ(op->schema().arguments()[1].name(), "b");
  ASSERT_EQ(op->schema().arguments()[1].type()->kind(), TypeKind::TensorType);

  ASSERT_EQ(op->schema().returns().size(), 1);
  ASSERT_EQ(op->schema().returns()[0].type()->kind(), TypeKind::TensorType);

  Stack stack;
  push(stack, 2.0f, at::ones(5));
  op->getOperation()(&stack);
  at::Tensor output;
  pop(stack, output);

  ASSERT_TRUE(output.allclose(at::full(5, 3.0f)));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(CustomOperatorTest, ListParameters) {
  // Check that lists work well.
  torch::RegisterOperators reg(
      "foo::lists(int[] ints, float[] floats, complex[] complexdoubles, Tensor[] tensors) -> float[]",
      [](torch::List<int64_t> ints,
         torch::List<double> floats,
         torch::List<c10::complex<double>> complexdoubles,
         torch::List<at::Tensor> tensors) { return floats; });

  auto& ops = getAllOperatorsFor(Symbol::fromQualString("foo::lists"));
  ASSERT_EQ(ops.size(), 1);

  auto& op = ops.front();
  ASSERT_EQ(op->schema().name(), "foo::lists");

  ASSERT_EQ(op->schema().arguments().size(), 4);
  ASSERT_EQ(op->schema().arguments()[0].name(), "ints");
  ASSERT_TRUE(
      op->schema().arguments()[0].type()->isSubtypeOf(ListType::ofInts()));
  ASSERT_EQ(op->schema().arguments()[1].name(), "floats");
  ASSERT_TRUE(
      op->schema().arguments()[1].type()->isSubtypeOf(ListType::ofFloats()));
  ASSERT_EQ(op->schema().arguments()[2].name(), "complexdoubles");
  ASSERT_TRUE(op->schema().arguments()[2].type()->isSubtypeOf(
      ListType::ofComplexDoubles()));
  ASSERT_EQ(op->schema().arguments()[3].name(), "tensors");
  ASSERT_TRUE(
      op->schema().arguments()[3].type()->isSubtypeOf(ListType::ofTensors()));

  ASSERT_EQ(op->schema().returns().size(), 1);
  ASSERT_TRUE(
      op->schema().returns()[0].type()->isSubtypeOf(ListType::ofFloats()));

  Stack stack;
  push(stack, c10::List<int64_t>({1, 2}));
  push(stack, c10::List<double>({1.0, 2.0}));
  push(
      stack,
      c10::List<c10::complex<double>>(
          {c10::complex<double>(2.4, -5.5), c10::complex<double>(-1.3, 2)}));
  push(stack, c10::List<at::Tensor>({at::ones(5)}));
  op->getOperation()(&stack);
  c10::List<double> output;
  pop(stack, output);

  ASSERT_EQ(output.size(), 2);
  ASSERT_EQ(output.get(0), 1.0);
  ASSERT_EQ(output.get(1), 2.0);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(CustomOperatorTest, ListParameters2) {
  torch::RegisterOperators reg(
      "foo::lists2(Tensor[] tensors) -> Tensor[]",
      [](torch::List<at::Tensor> tensors) { return tensors; });

  auto& ops = getAllOperatorsFor(Symbol::fromQualString("foo::lists2"));
  ASSERT_EQ(ops.size(), 1);

  auto& op = ops.front();
  ASSERT_EQ(op->schema().name(), "foo::lists2");

  ASSERT_EQ(op->schema().arguments().size(), 1);
  ASSERT_EQ(op->schema().arguments()[0].name(), "tensors");
  ASSERT_TRUE(
      op->schema().arguments()[0].type()->isSubtypeOf(ListType::ofTensors()));

  ASSERT_EQ(op->schema().returns().size(), 1);
  ASSERT_TRUE(
      op->schema().returns()[0].type()->isSubtypeOf(ListType::ofTensors()));

  Stack stack;
  push(stack, c10::List<at::Tensor>({at::ones(5)}));
  op->getOperation()(&stack);
  c10::List<at::Tensor> output;
  pop(stack, output);

  ASSERT_EQ(output.size(), 1);
  ASSERT_TRUE(output.get(0).allclose(at::ones(5)));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(CustomOperatorTest, Aliasing) {
  torch::RegisterOperators reg(
      "foo::aliasing", [](at::Tensor a, at::Tensor b) -> at::Tensor {
        a.add_(b);
        return a;
      });
  getAllOperatorsFor(Symbol::fromQualString("foo::aliasing"));

  {
    auto graph = std::make_shared<Graph>();
    parseIR(
        R"IR(
graph(%x: Tensor, %y: Tensor):
  %ret : Tensor = foo::aliasing(%x, %y)
  return (%ret)
  )IR",
        graph.get());

    auto opNode = *graph->block()->nodes().begin();

    AliasDb aliasDb(graph);
    for (const auto input : opNode->inputs()) {
      // The custom op writes to all its inputs
      ASSERT_TRUE(aliasDb.writesToAlias(opNode, {input}));
      // The output should be a wildcard and thus alias all inputs
      ASSERT_TRUE(aliasDb.mayAlias(opNode->output(), input));
    }
  }
  {
    // DCE should not remove a custom op
    auto graph = std::make_shared<Graph>();
    const auto text = R"IR(
graph(%x: Tensor, %y: Tensor):
  # CHECK: foo::aliasing
  %ret : Tensor = foo::aliasing(%x, %y)
  return (%x)
  )IR";
    parseIR(text, graph.get());
    EliminateDeadCode(graph);

    testing::FileCheck().run(text, *graph);
  }
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
static constexpr char op_list[] = "foofoo::bar.template;foo::another";
#define TORCH_SELECTIVE_NAME_IN_SCHEMA(l, n)                                   \
  torch::detail::SelectiveStr<c10::impl::op_allowlist_contains_name_in_schema( \
      l, n)>(n)

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TestCustomOperator, OperatorGeneratorUndeclared) {
  // Try to register an op name that does not exist in op_list.
  // Expected: the op name is not registered.
  torch::jit::RegisterOperators reg({OperatorGenerator(
      TORCH_SELECTIVE_NAME_IN_SCHEMA(
          op_list, "foofoo::not_exist(float a, Tensor b) -> Tensor"),
      [](Stack* stack) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        double a;
        at::Tensor b;
        pop(stack, a, b);
        push(stack, a + b);
      },
      aliasAnalysisFromSchema())});

  auto& ops = getAllOperatorsFor(Symbol::fromQualString("foofoo::not_exist"));
  ASSERT_EQ(ops.size(), 0);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TestCustomOperator, OperatorGeneratorBasic) {
  // The operator should be successfully registered since its name is in the
  // whitelist.
  torch::jit::RegisterOperators reg({OperatorGenerator(
      TORCH_SELECTIVE_NAME_IN_SCHEMA(
          op_list, "foofoo::bar.template(float a, Tensor b) -> Tensor"),
      [](Stack* stack) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        double a;
        at::Tensor b;
        pop(stack, a, b);
        push(stack, a + b);
      },
      aliasAnalysisFromSchema())});

  auto& ops = getAllOperatorsFor(Symbol::fromQualString("foofoo::bar"));
  ASSERT_EQ(ops.size(), 1);

  auto& op = ops.front();
  ASSERT_EQ(op->schema().name(), "foofoo::bar");

  ASSERT_EQ(op->schema().arguments().size(), 2);
  ASSERT_EQ(op->schema().arguments()[0].name(), "a");
  ASSERT_EQ(op->schema().arguments()[0].type()->kind(), TypeKind::FloatType);
  ASSERT_EQ(op->schema().arguments()[1].name(), "b");
  ASSERT_EQ(op->schema().arguments()[1].type()->kind(), TypeKind::TensorType);

  ASSERT_EQ(op->schema().returns().size(), 1);
  ASSERT_EQ(op->schema().returns()[0].type()->kind(), TypeKind::TensorType);

  Stack stack;
  push(stack, 2.0f, at::ones(5));
  op->getOperation()(&stack);
  at::Tensor output;
  pop(stack, output);

  ASSERT_TRUE(output.allclose(at::full(5, 3.0f)));
}

} // namespace jit
} // namespace torch
