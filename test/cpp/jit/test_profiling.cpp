#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>
#include <torch/torch.h>

namespace torch {
namespace jit {
class JitProfilingFixtureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    prev_exec = getExecutorMode();
    getExecutorMode() = true;
    prev_profiling = getProfilingMode();
    getProfilingMode() = true;
    prev_inline_autodiff = getAutodiffSubgraphInlining();
    debugSetAutodiffSubgraphInlining(false);
  }
  void TearDown() override {
    getExecutorMode() = prev_exec;
    getProfilingMode() = prev_profiling;
    debugSetAutodiffSubgraphInlining(prev_inline_autodiff);
  }

  bool prev_exec;
  bool prev_profiling;
  bool prev_inline_autodiff;
};

TEST_F(JitProfilingFixtureTest, RemoveUnusedLinearGradientComputations) {
  auto graph = std::make_shared<Graph>();
  const std::string input =
      R"IR(
graph(%inp.1 : Tensor,
      %weight.1 : Tensor,
      %bias.1 : Tensor):
  %6 : Tensor = aten::linear(%inp.1, %weight.1, %bias.1)
  return (%6))IR";
  parseIR(input, graph.get());

  auto inp = torch::randn({10, 10}).requires_grad_(false);
  auto weight = torch::randn({10, 10}).requires_grad_(true);
  auto bias = torch::randn({1, 10}).requires_grad_(true);
  auto stack = createStack({inp, weight, bias});

  ProfilingGraphExecutorImpl executor(graph, "linear");

  // initial run to profile requires_grad information
  auto plan = executor.getPlanFor(stack, 20);
  InterpreterState is{plan.code};
  is.run(stack);

  auto optimized_plan = executor.getPlanFor(stack, 20);
  DepthFirstGraphNodeIterator it(optimized_plan.graph);
  Node* diff_graph_node = nullptr;

  while ((diff_graph_node = it.next()) != nullptr) {
    if (diff_graph_node->kind() == prim::DifferentiableGraph) {
      break;
    }
  }
  ASSERT_NE(nullptr, diff_graph_node);

  auto backward_graph = diff_graph_node->g(attr::ReverseSubgraph);
  std::cerr << *backward_graph << std::endl;

  // "input" does not need gradient, so it shouldn't be in the backwards graph
  testing::FileCheck()
      .check_not("%grad_input")
      ->check_not("%grad_inp")
      ->run(*backward_graph);

  // "bias" and "weight" need gradients
  testing::FileCheck().check("%grad_bias")->run(*backward_graph);
  testing::FileCheck().check("%grad_weight")->run(*backward_graph);
}
} // namespace jit
} // namespace torch
