#include <gtest/gtest.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/ops.h>

using namespace torch::jit;

namespace {

StaticModule makeStaticModuleFromScript(const std::string& script) {
  Module m("module");
  m.define(script);
  return StaticModule(m);
}

} // namespace

/**
 * Test that StaticModule::value_group groups values of the graph into
 * 1) Inputs/Constants and their aliases 2) Outputs and their aliases.
 */
TEST(StaticModule, ValueGroup) {
  const std::string src = R"IR(
    graph(%input0 : Tensor, %input1 : Tensor):
      # Constants.
      %0 : int = prim::Constant[value=1]()
      # Internal values.
      %1 : Tensor = aten::add(%input0, %input1, %0)
      # This includes aliases of output.
      %2 : Tensor = aten::add(%input0, %1, %0)
      # This includes output.
      %3 : (Tensor) = prim::TupleConstruct(%2)
      return (%3)
    )IR";
  auto input_graph = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(src, input_graph.get());
  torch::jit::StaticModule sm(input_graph);
  const Graph& graph = sm.graph();
  std::vector<const Node*> nodes(graph.nodes().begin(), graph.nodes().end());
  const auto& value_group = sm.value_group();

  std::vector<const Value*> expected_input_aliases{graph.inputs()[0], graph.inputs()[1], nodes[0]->output()};
  for (auto* value : expected_input_aliases) {
    EXPECT_TRUE(value_group.isExternalAlias(value));
  }

  std::vector<const Value*> expected_output_aliases{graph.outputs()[0], nodes[2]->output()};
  for (auto* value : expected_output_aliases) {
    EXPECT_TRUE(value_group.isOutputAlias(value));
  }
  EXPECT_FALSE(value_group.isAlwaysAlive(nodes[1]->output()));
  EXPECT_TRUE(value_group.isAlwaysAlive(graph.inputs()[0]));
  EXPECT_TRUE(value_group.isAlwaysAlive(graph.inputs()[1]));
  EXPECT_TRUE(value_group.isAlwaysAlive(graph.outputs()[0]));
}

TEST(StaticModule, IsOptimizableContainerType_NonOptimizableInputs) {
  // Cannot use out variants for list/tuple construction here because
  // inputs are not produced by nodes with out variants.
  const std::string src = R"JIT(
        def forward(self, a, b):
            a_alias = a.view(a.size())
            non_optimizable_list = [a_alias]
            non_optimizable_tuple = (b, )
            return non_optimizable_list, non_optimizable_tuple
    )JIT";

  auto sm = makeStaticModuleFromScript(src);
  const auto& graph = sm.graph();

  for (const Node* n : graph.nodes()) {
    EXPECT_FALSE(sm.is_optimizable_container_type(n));
  }
}

TEST(StaticModule, IsOptimizableContainerType_WrongType) {
  // Cannot use out variants for list/tuple construction here because
  // types are not Tensors
  const std::string src = R"JIT(
        def forward(self, x: int, y: int):
            a = 1 + x
            b = 2 + y
            non_optimizable_list = [a]
            non_optimizable_tuple = (b, )
            return non_optimizable_list, non_optimizable_tuple
    )JIT";

  auto sm = makeStaticModuleFromScript(src);
  const auto& graph = sm.graph();

  for (const Node* n : graph.nodes()) {
    EXPECT_FALSE(sm.is_optimizable_container_type(n));
  }
}

TEST(StaticModule, IsOptimizableContainerType_CanUseOutVariant) {
  // This container should be optimizable since aten::add has an
  // out variant the container contains Tensors.
  const std::string src = R"JIT(
        def forward(self, x):
            a = torch.relu(x)
            optimizable_list = [a]
            return optimizable_list
    )JIT";
  auto sm = makeStaticModuleFromScript(src);
  const auto& graph = sm.graph();

  for (const Node* n : graph.nodes()) {
    if (n->kind() == prim::ListConstruct) {
      EXPECT_TRUE(sm.is_optimizable_container_type(n));
    } else {
      EXPECT_FALSE(sm.is_optimizable_container_type(n));
    }
  }
}
