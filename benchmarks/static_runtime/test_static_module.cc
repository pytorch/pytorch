#include <gtest/gtest.h>
#include <torch/csrc/jit/ir/ir.h>
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
 * Test that StaticModule correctly initializes the set of always alive values
 * containing: 1) Inputs/Outputs 2) Constants 3) Aliases of (1) and (2)
 */
TEST(StaticModule, ExternalValues) {
  const std::string src = R"JIT(
        def forward(self, a, b):
            a_alias = a.view(a.size())
            inputs_list = [a, b]
            return a_alias + b + 1, inputs_list
    )JIT";
  auto sm = makeStaticModuleFromScript(src);

  const auto& external_values = sm.external_values();
  const auto& graph = sm.graph();

  auto value_is_always_alive = [&external_values](const Value* v) {
    return external_values.find(v) != external_values.end();
  };

  for (const Value* v : graph.inputs()) {
    EXPECT_TRUE(value_is_always_alive(v));
  }
  for (const Value* v : graph.outputs()) {
    EXPECT_TRUE(value_is_always_alive(v));
  }

  for (const Node* n : graph.nodes()) {
    if (n->kind() == prim::Constant ||
        // In the graph above, a view on an input is created.
        n->kind() == aten::view ||
        // We also create a list of the inputs.
        n->kind() == prim::ListConstruct) {
      EXPECT_TRUE(value_is_always_alive(n->output()));
    }
  }
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
