#pragma once

#include "test/cpp/jit/test_base.h"

#include <torch/csrc/jit/passes/canonicalize.h>
#include "ATen/core/interned_strings.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/argument_spec.h"
#include "torch/csrc/jit/attributes.h"
#include "torch/csrc/jit/autodiff.h"
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/custom_operator.h"
#include "torch/csrc/jit/dynamic_dag.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/import.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/passes/alias_analysis.h"
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
#include "torch/csrc/jit/symbolic_script.h"
#include "torch/csrc/jit/symbolic_variable.h"
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/utils/hash.h"
#include "torch/csrc/utils/memory.h"

#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/variable.h"

#include <torch/csrc/jit/testing/file_check.h>
#include "ATen/core/ivalue.h"
#include "torch/csrc/jit/graph_executor.h"
#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/jit/script/module.h"

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
namespace {

using Var = SymbolicVariable;

void testFusion() {
  auto testSimple = [&] {
    Graph graph;
    Var i0 = Var::asNewInput(graph);
    Var i1 = Var::asNewInput(graph);
    auto o0 = i0 * i1;
    o0.addAsOutput();
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
    Graph graph;

    Var i0 = Var::asNewInput(graph);
    Var i1 = Var::asNewInput(graph);
    Var i2 = Var::asNewInput(graph);
    Var i3 = Var::asNewInput(graph);
    Var i4 = Var::asNewInput(graph);

    auto p22 = i4.sigmoid();
    auto p20 = i3.sigmoid();
    auto p18 = i2.tanh();
    auto p16 = i1.sigmoid();
    auto p14 = p20 * i0;
    auto p11 = p22 * p18;
    auto o1 = p14 + p11;
    auto p5 = o1.tanh();
    auto o0 = p16 * p5;
    o0.addAsOutput();
    o1.addAsOutput();

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

  auto createFusedConcat =
      [](Graph& graph, at::ArrayRef<Value*> inputs, int64_t dim) -> Value* {
    return graph
        .insertNode(graph.create(prim::FusedConcat, inputs)->i_(attr::dim, dim))
        ->output();
  };

  auto testConcat = [&](int dim) {
    Graph graph;
    Var i0 = Var::asNewInput(graph);
    Var i1 = Var::asNewInput(graph);
    auto o0 = i0 * i1;
    o0.addAsOutput();
    Var(createFusedConcat(graph, {i0, o0}, dim)).addAsOutput();

    auto a = at::rand({3, 4, 5}, at::kCUDA);
    auto b = at::rand({4, 3, 5}, at::kCUDA).transpose(0, 1);

    auto o_r = a * b;
    auto o2_r = at::cat({a, o_r}, dim);
    auto outputs = debugLaunchGraph(graph, {a, b});
    ASSERT_EQ(outputs.size(), 2);

    float max_diff = (o_r - outputs[0]).abs().max().item<double>();
    ASSERT_EQ(max_diff, 0);
    float max_diff2 = (o2_r - outputs[1]).abs().max().item<double>();
    ASSERT_EQ(max_diff2, 0);
  };
  testConcat(0);
  testConcat(1);
  testConcat(2);
}

void testRegisterFusionCachesKernel(std::ostream& out = std::cout) {
  // Build up a fake graph with a FusionGroup
  auto createGraphWithNames = [](std::string cname, std::string dname) {
    auto graph = std::make_shared<Graph>();
    at::ScalarType s = at::ScalarType::Float;
    auto type = CompleteTensorType::create(s, at::kCPU, {2, 3, 4}, {12, 4, 1});
    auto a = SymbolicVariable::asNewInput(*graph, type);
    auto b = SymbolicVariable::asNewInput(*graph, type);
    auto c = a * b;
    auto d = c * a;
    c.value()->setDebugName(cname);
    d.value()->setDebugName(dname);
    graph->registerOutput(d.value());
    torch::jit::overrideCanFuseOnCPU(true);
    FuseGraph(graph);
    torch::jit::overrideCanFuseOnCPU(false);
    return graph;
  };

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

  // Create two alpha-equivalent fusion groups
  auto graph1 = createGraphWithNames("c1", "d1");
  auto fg1 = getFusionGroup(graph1);

  auto graph2 = createGraphWithNames("c2", "d2");
  auto fg2 = getFusionGroup(graph2);

  // Register both with the fusion compiler.
  auto expected_key = registerFusion(fg1);
  auto second_key = registerFusion(fg2);

  // Because the graphs are alpha-equivalent, they should return the same key
  // and therefore share a KernelSpec to share kernels for specializations
  ASSERT_EQ(second_key, expected_key);
}
} // namespace
} // namespace jit
} // namespace torch
