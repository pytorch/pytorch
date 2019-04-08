#pragma once

#include "test/cpp/jit/test_base.h"
#include "test/cpp/jit/test_utils.h"

#include "torch/csrc/jit/custom_operator.h"
#include "torch/csrc/jit/irparser.h"
#include "torch/csrc/jit/passes/alias_analysis.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace torch {
namespace jit {
namespace test {

void testCustomOperators() {
  {
    RegisterOperators reg({createOperator(
        "foo::bar", [](double a, at::Tensor b) { return a + b; })});
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
    push(stack, 2.0f, autograd::make_variable(at::ones(5)));
    op->getOperation()(stack);
    at::Tensor output;
    pop(stack, output);

    ASSERT_TRUE(output.allclose(autograd::make_variable(at::full(5, 3.0f))));
  }
  {
    RegisterOperators reg({createOperator(
        "foo::bar_with_schema(float a, Tensor b) -> Tensor",
        [](double a, at::Tensor b) { return a + b; })});

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
    push(stack, 2.0f, autograd::make_variable(at::ones(5)));
    op->getOperation()(stack);
    at::Tensor output;
    pop(stack, output);

    ASSERT_TRUE(output.allclose(autograd::make_variable(at::full(5, 3.0f))));
  }
  {
    // Check that lists work well.
    RegisterOperators reg({createOperator(
        "foo::lists(int[] ints, float[] floats, Tensor[] tensors) -> float[]",
        [](const std::vector<int64_t>& ints,
           const std::vector<double>& floats,
           std::vector<at::Tensor> tensors) { return floats; })});

    auto& ops = getAllOperatorsFor(Symbol::fromQualString("foo::lists"));
    ASSERT_EQ(ops.size(), 1);

    auto& op = ops.front();
    ASSERT_EQ(op->schema().name(), "foo::lists");

    ASSERT_EQ(op->schema().arguments().size(), 3);
    ASSERT_EQ(op->schema().arguments()[0].name(), "ints");
    ASSERT_TRUE(
        op->schema().arguments()[0].type()->isSubtypeOf(ListType::ofInts()));
    ASSERT_EQ(op->schema().arguments()[1].name(), "floats");
    ASSERT_TRUE(
        op->schema().arguments()[1].type()->isSubtypeOf(ListType::ofFloats()));
    ASSERT_EQ(op->schema().arguments()[2].name(), "tensors");
    ASSERT_TRUE(
        op->schema().arguments()[2].type()->isSubtypeOf(ListType::ofTensors()));

    ASSERT_EQ(op->schema().returns().size(), 1);
    ASSERT_TRUE(
        op->schema().returns()[0].type()->isSubtypeOf(ListType::ofFloats()));

    Stack stack;
    push(stack, std::vector<int64_t>{1, 2});
    push(stack, std::vector<double>{1.0, 2.0});
    push(stack, std::vector<at::Tensor>{autograd::make_variable(at::ones(5))});
    op->getOperation()(stack);
    std::vector<double> output;
    pop(stack, output);

    ASSERT_EQ(output.size(), 2);
    ASSERT_EQ(output[0], 1.0);
    ASSERT_EQ(output[1], 2.0);
  }
  {
    RegisterOperators reg(
        "foo::lists2(Tensor[] tensors) -> Tensor[]",
        [](std::vector<at::Tensor> tensors) { return tensors; });

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
    push(stack, std::vector<at::Tensor>{autograd::make_variable(at::ones(5))});
    op->getOperation()(stack);
    std::vector<at::Tensor> output;
    pop(stack, output);

    ASSERT_EQ(output.size(), 1);
    ASSERT_TRUE(output[0].allclose(autograd::make_variable(at::ones(5))));
  }
  {
    auto op = createOperator(
        "traced::op(float a, Tensor b) -> Tensor",
        [](double a, at::Tensor b) { return a + b; });

    std::shared_ptr<tracer::TracingState> state;
    std::tie(state, std::ignore) = tracer::enter({});

    Stack stack;
    push(stack, 2.0f, autograd::make_variable(at::ones(5)));
    op.getOperation()(stack);
    at::Tensor output = autograd::make_variable(at::empty({}));
    pop(stack, output);

    tracer::exit({IValue(output)});

    std::string op_name("traced::op");
    bool contains_traced_op = false;
    for (const auto& node : state->graph->nodes()) {
      if (std::string(node->kind().toQualString()) == op_name) {
        contains_traced_op = true;
        break;
      }
    }
    ASSERT_TRUE(contains_traced_op);
  }
  {
    ASSERT_THROWS_WITH(
        createOperator(
            "foo::bar_with_bad_schema(Tensor a) -> Tensor",
            [](double a, at::Tensor b) { return a + b; }),
        "Inferred 2 argument(s) for operator implementation, "
        "but the provided schema specified 1 argument(s).");
    ASSERT_THROWS_WITH(
        createOperator(
            "foo::bar_with_bad_schema(Tensor a) -> Tensor",
            [](double a) { return a; }),
        "Inferred type for argument #0 was float, "
        "but the provided schema specified type Tensor "
        "for the argument in that position");
    ASSERT_THROWS_WITH(
        createOperator(
            "foo::bar_with_bad_schema(float a) -> (float, float)",
            [](double a) { return a; }),
        "Inferred 1 return value(s) for operator implementation, "
        "but the provided schema specified 2 return value(s).");
    ASSERT_THROWS_WITH(
        createOperator(
            "foo::bar_with_bad_schema(float a) -> Tensor",
            [](double a) { return a; }),
        "Inferred type for return value #0 was float, "
        "but the provided schema specified type Tensor "
        "for the return value in that position");
  }
  {
    // vector<double> is not supported yet.
    auto op = createOperator(
        "traced::op(float[] f) -> int",
        [](const std::vector<double>& f) -> int64_t { return f.size(); });

    std::shared_ptr<tracer::TracingState> state;
    std::tie(state, std::ignore) = tracer::enter({});

    Stack stack;
    push(stack, std::vector<double>{1.0});

    ASSERT_THROWS_WITH(
        op.getOperation()(stack),
        "Tracing float lists currently not supported!");

    tracer::abandon();
  }
  {
    // Try to create an op using a reserved namespace
    ASSERT_THROWS_WITH(
        createOperator(
            "aten::op(float[] f) -> int",
            [](const std::vector<double>& f) -> int64_t { return f.size(); }),
        "Tried to register a custom operator to a reserved namespace");
  }
}

void testCustomOperatorAliasing() {
  RegisterOperators reg({createOperator(
      "foo::aliasing", [](at::Tensor a, at::Tensor b) -> at::Tensor {
        a.add_(b);
        return a;
      })});
  auto& ops = getAllOperatorsFor(Symbol::fromQualString("foo::aliasing"));

  {
    auto graph = std::make_shared<Graph>();
    script::parseIR(
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
    script::parseIR(text, graph.get());
    EliminateDeadCode(graph);

    testing::FileCheck().run(text, *graph);
  }
}
} // namespace test
} // namespace jit
} // namespace torch
