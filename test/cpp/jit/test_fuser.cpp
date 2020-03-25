#include "test/cpp/jit/test_base.h"

#include <torch/csrc/jit/passes/canonicalize.h>
#include "ATen/core/interned_strings.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/runtime/argument_spec.h"
#include "torch/csrc/jit/ir/attributes.h"
#include "torch/csrc/jit/runtime/autodiff.h"
#include "torch/csrc/jit/frontend/code_template.h"
#include "torch/csrc/jit/runtime/custom_operator.h"
#include "torch/csrc/jit/codegen/fuser/interface.h"
#include "torch/csrc/jit/serialization/import.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/runtime/interpreter.h"
#include "torch/csrc/jit/ir/alias_analysis.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/graph_fuser.h"
#include "torch/csrc/jit/passes/lower_grad_of.h"
#include "torch/csrc/jit/passes/lower_tuples.h"
#include "torch/csrc/jit/passes/requires_grad_analysis.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/passes/utils/subgraph_utils.h"
#include "torch/csrc/jit/runtime/symbolic_script.h"
#include "torch/csrc/jit/frontend/tracer.h"



#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/variable.h"

#include <torch/csrc/jit/testing/file_check.h>
#include "ATen/core/ivalue.h"
#include "torch/csrc/jit/runtime/graph_executor.h"
#include "torch/csrc/jit/frontend/ir_emitter.h"
#include "torch/csrc/jit/api/module.h"

#include "onnx/onnx_pb.h"

#include <ATen/ATen.h>

#include <c10/util/Exception.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace torch {
namespace jit {

void testFusion() {
  auto testSimple = [&] {
    const auto graph_string = R"IR(
      graph(%0 : Tensor,
            %1 : Tensor):
        %2 : Tensor = aten::mul(%0, %1)
        return (%2))IR";
    Graph graph;
    torch::jit::parseIR(graph_string, &graph);

    auto a = at::rand({3, 4}, at::kCUDA);
    auto b = at::rand({4, 3}, at::kCUDA).transpose(0, 1);
    auto o = at::zeros({3, 4}, at::kCUDA);
    auto outputs = debugLaunchGraph(graph, {a, b});
    ASSERT_EQ(outputs.size(), 1);
    auto o2 = a * b;
    float max_diff = (o2 - outputs[0]).abs().max().item<double>();
    // std::cout << "max diff: " << max_diff << "\n";
    ASSERT_EQ(max_diff, 0);
  };
  testSimple();

  auto testOne = [&](int ti, int tj) {
    const auto graph_string = R"IR(
      graph(%0 : Tensor,
            %1 : Tensor,
            %2 : Tensor,
            %3 : Tensor,
            %4 : Tensor):
        %5 : Tensor = aten::sigmoid(%4)
        %6 : Tensor = aten::sigmoid(%3)
        %7 : Tensor = aten::tanh(%2)
        %8 : Tensor = aten::sigmoid(%1)
        %9 : Tensor = aten::mul(%6, %0)
        %10 : Tensor = aten::mul(%5, %7)
        %11 : int = prim::Constant[value=1]()
        %12 : Tensor = aten::add(%9, %10, %11)
        %13 : Tensor = aten::tanh(%12)
        %14 : Tensor = aten::mul(%8, %13)
        return (%14, %12))IR";
    Graph graph;
    torch::jit::parseIR(graph_string, &graph);

    graph.lint();

    std::vector<at::Tensor> inputs;
    // We want to generate input/output tensors with dimension 128x128x32, but
    // with different internal strides.  To do this, we generate a tensor
    // with the "wrong" dimensions, and then use transpose to get an
    // appropriately sized view.
    for (size_t i = 0; i < graph.inputs().size(); i++) {
      std::vector<int64_t> dims = {128, 128, 32};
      std::swap(dims[ti], dims[tj]);
      inputs.push_back(at::rand(dims, at::kCUDA).transpose(ti, tj));
    }

    auto t22 = inputs[4].sigmoid();
    auto t20 = inputs[3].sigmoid();
    auto t18 = inputs[2].tanh();
    auto t16 = inputs[1].sigmoid();
    auto t14 = t20 * inputs[0];
    auto t11 = t22 * t18;
    auto out1 = t14 + t11;
    auto t5 = out1.tanh();
    auto out0 = t16 * t5;

    auto outputs = debugLaunchGraph(graph, inputs);
    ASSERT_EQ(outputs.size(), graph.outputs().size());
    ASSERT_TRUE(out0.is_same_size(outputs.front()));
    float max_diff = (outputs.front() - out0).abs().max().item<double>();
    ASSERT_TRUE(max_diff < 1e-6);
  };
  testOne(0, 0);
  testOne(0, 1);
  testOne(1, 2);
  testOne(0, 2);

  const auto graph_string0 = R"IR(
    graph(%0 : Tensor,
          %1 : Tensor):
      %2 : Tensor = aten::mul(%0, %1)
      %3 : Tensor = prim::FusedConcat[dim=0](%0, %2)
      return (%2, %3))IR";
  const auto graph_string1 = R"IR(
    graph(%0 : Tensor,
          %1 : Tensor):
      %2 : Tensor = aten::mul(%0, %1)
      %3 : Tensor = prim::FusedConcat[dim=1](%0, %2)
      return (%2, %3))IR";
  const auto graph_string2 = R"IR(
    graph(%0 : Tensor,
          %1 : Tensor):
      %2 : Tensor = aten::mul(%0, %1)
      %3 : Tensor = prim::FusedConcat[dim=2](%0, %2)
      return (%2, %3))IR";

  auto a = at::rand({3, 4, 5}, at::kCUDA);
  auto b = at::rand({4, 3, 5}, at::kCUDA).transpose(0, 1);
  const auto o_r = a * b;

  std::vector<std::string> graph_strings{graph_string0,
                                         graph_string1,
                                         graph_string2};
  for (auto i = decltype(graph_strings.size()){0}; i < graph_strings.size(); ++i) {
    Graph g;
    torch::jit::parseIR(graph_strings[i], &g);

    auto outputs = debugLaunchGraph(g, {a, b});
    ASSERT_EQ(outputs.size(), 2);

    float max_diff = (o_r - outputs[0]).abs().max().item<double>();
    ASSERT_EQ(max_diff, 0);

    const auto o2_r = at::cat({a, o_r}, i);
    float max_diff2 = (o2_r - outputs[1]).abs().max().item<double>();
    ASSERT_EQ(max_diff2, 0);
  };
}

void testRegisterFusionCachesKernel() {
  // Constructs two functionally equivalent graphs
  const auto graph0_string = R"IR(
    graph(%0 : Float(2, 3, 4),
          %1 : Float(2, 3, 4)):
      %c0 : Float(2, 3, 4) = aten::mul(%0, %1)
      %d0 : Float(2, 3, 4) = aten::mul(%c0, %0)
      return (%d0))IR";
  auto g0 = std::make_shared<Graph>();
  torch::jit::parseIR(graph0_string, g0.get());

  const auto graph1_string = R"IR(
    graph(%0 : Float(2, 3, 4),
          %1 : Float(2, 3, 4)):
      %c1 : Float(2, 3, 4) = aten::mul(%0, %1)
      %d1 : Float(2, 3, 4) = aten::mul(%c1, %0)
      return (%d1))IR";
  auto g1 = std::make_shared<Graph>();
  torch::jit::parseIR(graph1_string, g1.get());

  auto getFusionGroup = [](const std::shared_ptr<Graph>& graph) {
    const auto& nodes = graph->nodes();
    auto maybe_fusion_group =
        std::find_if(nodes.begin(), nodes.end(), [](const Node* node) {
          return node->kind() == prim::FusionGroup;
        });
    TORCH_CHECK(
        maybe_fusion_group != nodes.end(),
        "testRegisterFusionCachesKernel: could not create FusionGroup");
    return *maybe_fusion_group;
  };

  // Creates two alpha-equivalent fusion groups
  torch::jit::overrideCanFuseOnCPU(true);
  FuseGraph(g0);
  FuseGraph(g1);
  torch::jit::overrideCanFuseOnCPU(false);
  auto fg0 = getFusionGroup(g0);
  auto fg1 = getFusionGroup(g1);

  // Registers both with the fusion compiler.
  auto expected_key = registerFusion(fg0);
  auto second_key = registerFusion(fg1);

  // Because the graphs are alpha-equivalent, they should return the same key
  // and therefore share a KernelSpec to share kernels for specializations
  ASSERT_EQ(second_key, expected_key);
}
} // namespace jit
} // namespace torch
