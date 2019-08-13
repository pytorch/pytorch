#pragma once

#include <ATen/ATen.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <ATen/Parallel.h>
#include <ATen/ThreadLocalDebugInfo.h>

#include "test/cpp/jit/test_base.h"
#include "test/cpp/jit/test_utils.h"

#include <torch/csrc/jit/passes/canonicalize.h>
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
#include "torch/csrc/jit/pass_manager.h"
#include "torch/csrc/jit/passes/alias_analysis.h"
#include "torch/csrc/jit/passes/bailout_graph.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/graph_fuser.h"
#include "torch/csrc/jit/passes/guard_elimination.h"
#include "torch/csrc/jit/passes/insert_guards.h"
#include "torch/csrc/jit/passes/liveness.h"
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
#include "torch/csrc/jit/profiling_record.h"
#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/jit/script/module.h"
#include "torch/jit.h"

#include "onnx/onnx_pb.h"

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
c10::OperatorOptions aliasAnalysisFromSchema();
namespace test {

using Var = SymbolicVariable;

using namespace torch::autograd;

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& list) {
  size_t i = 0;
  out << "{";
  for (auto&& e : list) {
    if (i++ > 0)
      out << ", ";
    out << e;
  }
  out << "}";
  return out;
}

void testInternedStrings() {
  ASSERT_EQ(prim::Param, Symbol::prim("Param"));
  ASSERT_EQ(prim::Return, Symbol::prim("Return"));
  ASSERT_EQ(prim::Return.toUnqualString(), std::string("Return"));
  ASSERT_EQ(prim::Return.toQualString(), std::string("prim::Return"));
  Symbol newsym = Symbol::aten("__NEW_SYMBOL");
  size_t symstart = newsym;
  ASSERT_EQ(newsym.toQualString(), std::string("aten::__NEW_SYMBOL"));
  // TODO: This test is a bit too close to the implementation details.
  ASSERT_EQ(Symbol::aten("What"), symstart + 1);
  ASSERT_EQ(Symbol::aten("What2"), symstart + 2);
  ASSERT_EQ(Symbol::aten("What"), symstart + 1);
  ASSERT_EQ(Symbol::aten("What2"), symstart + 2);
  ASSERT_EQ(Symbol(symstart + 2).toUnqualString(), std::string("What2"));
}

void testFromQualString() {
  ASSERT_EQ(Symbol::fromQualString("prim::Param"), Symbol::prim("Param"));
  ASSERT_EQ(Symbol::fromQualString("aten::mm"), Symbol::aten("mm"));
  ASSERT_EQ(Symbol::fromQualString("onnx::LSTM"), Symbol::onnx("LSTM"));
  ASSERT_EQ(Symbol::fromQualString("attr::value"), Symbol::attr("value"));
  ASSERT_EQ(Symbol::fromQualString("scope::"), Symbol::scope(""));
  ASSERT_EQ(Symbol::fromQualString("::").toUnqualString(), std::string(""));
  ASSERT_EQ(
      Symbol::fromQualString("::").ns().toQualString(),
      std::string("namespaces::"));
  ASSERT_EQ(
      Symbol::fromQualString("new_ns::param").toUnqualString(),
      std::string("param"));
  ASSERT_EQ(
      Symbol::fromQualString("new_ns::param").ns().toUnqualString(),
      std::string("new_ns"));
  ASSERT_EQ(
      Symbol::fromQualString("new_ns::param").ns(),
      Symbol::fromQualString("namespaces::new_ns"));

  auto bad_inputs = {"scope", ":", ""};
  for (auto input : bad_inputs) {
    try {
      Symbol::fromQualString(input);
      ASSERT_TRUE(0);
    } catch (const std::exception& c) {
    }
  }
}

void testTHNNConv() {
  std::vector<int64_t> input_size = {4, 3, 15, 17}; // B x C x H x W
  std::vector<int64_t> kernel_size = {3, 5};
  std::vector<int64_t> stride = {1, 2};
  std::vector<int64_t> padding = {2, 1};
  constexpr int out_channels = 5;

  // make inputs
  at::Tensor input = torch::randn(input_size);
  at::Tensor weight = torch::randn(
      {out_channels, input_size[1], kernel_size[0], kernel_size[1]});
  at::Tensor bias = torch::randn({out_channels});

  // run forward eagerly
  at::Tensor output, finput, fgradinput;
  std::tie(output, finput, fgradinput) = at::thnn_conv2d_forward(
      input, weight, kernel_size, bias, stride, padding);

  // make grad_outputs
  at::Tensor grad_output = torch::randn_like(output);
  at::Tensor grad_finput = torch::zeros_like(finput);
  at::Tensor grad_fgradinput = torch::zeros_like(fgradinput);

  // run backward eagerly
  at::Tensor grad_input, grad_weight, grad_bias;
  std::tie(grad_input, grad_weight, grad_bias) = at::thnn_conv2d_backward(
      grad_output,
      input,
      weight,
      kernel_size,
      stride,
      padding,
      finput,
      fgradinput,
      {true, true, true});

  // make JIT graph
  auto graph = std::make_shared<Graph>();
  auto ksz_val = graph->insertConstant(c10::impl::toList(kernel_size));
  auto kst_val = graph->insertConstant(c10::impl::toList(stride));
  auto pad_val = graph->insertConstant(c10::impl::toList(padding));

  auto inputg = graph->addInput("self");
  auto weightg = graph->addInput("weight");
  auto biasg = graph->addInput("bias");

  Value* conv = graph->insert(
      aten::thnn_conv2d_forward,
      {inputg, weightg, ksz_val, biasg, kst_val, pad_val});
  auto outputs = conv->node()->outputs();
  for (auto output : outputs) {
    graph->registerOutput(output);
  }
  LowerAllTuples(graph);
  graph->lint();

  // differentiate JIT graph
  EliminateDeadCode(graph); // Tracing of some ops depends on the DCE trick
  ConstantPropagation(graph);
  auto grad_spec = differentiate(graph);
  LowerGradOf(*grad_spec.df);

  // prepare JIT inputs / gradients
  tensor_list tensors_in;
  tensors_in.push_back(input);
  tensors_in.push_back(weight);
  tensors_in.push_back(bias);

  tensor_list tensor_grads_in;
  tensor_grads_in.push_back(grad_output);
  tensor_grads_in.push_back(grad_finput);
  tensor_grads_in.push_back(grad_fgradinput);

  // Get outputs from the interpreter
  tensor_list tensors_out, tensor_grads_out;
  std::tie(tensors_out, tensor_grads_out) =
      runGradient(grad_spec, tensors_in, tensor_grads_in);

  // prepare expected structs
  tensor_list expected_tensors_out, expected_tensor_grads_out;
  expected_tensors_out.push_back(output);
  expected_tensors_out.push_back(finput);
  expected_tensors_out.push_back(fgradinput);
  expected_tensor_grads_out.push_back(grad_input);
  expected_tensor_grads_out.push_back(grad_weight);
  expected_tensor_grads_out.push_back(grad_bias);

  // Compare results
  assertAllClose(tensors_out, expected_tensors_out);
  assertAllClose(tensor_grads_out, expected_tensor_grads_out);
}

void testATenNativeBatchNorm() {
  // aten::native_batch_norm(Tensor input, Tensor weight, Tensor bias, Tensor
  // running_mean, Tensor running_var, bool training, float momentum, float eps)
  // -> (Tensor, Tensor, Tensor)
  std::vector<int64_t> input_size = {4, 3, 15, 17}; // B x C x H x W
  bool training = true;
  float momentum = 0.9;
  float eps = 1e-5;

  // make inputs
  at::Tensor input = torch::randn(input_size);
  at::Tensor weight = torch::randn({input_size[1]});
  at::Tensor bias = torch::randn({input_size[1]});
  at::Tensor running_mean = torch::randn({input_size[1]});
  at::Tensor running_var = torch::randn({input_size[1]});

  // running_mean and running_var are changed in-place, so clone and send them
  at::Tensor running_mean_eager = running_mean.clone();
  at::Tensor running_var_eager = running_var.clone();
  at::Tensor running_mean_jit = running_mean.clone();
  at::Tensor running_var_jit = running_var.clone();

  // run forward eagerly
  at::Tensor output, savemean, saveinvstd;
  std::tie(output, savemean, saveinvstd) = at::native_batch_norm(
      input,
      weight,
      bias,
      running_mean_eager,
      running_var_eager,
      training,
      momentum,
      eps);

  // make grad_outputs
  at::Tensor grad_output = torch::randn_like(output);
  at::Tensor grad_savemean = torch::zeros_like(savemean);
  at::Tensor grad_saveinvstd = torch::zeros_like(saveinvstd);

  // run backward eagerly
  at::Tensor grad_input, grad_weight, grad_bias;
  // aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor
  // weight, Tensor running_mean, Tensor running_var, Tensor save_mean, Tensor
  // save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor,
  // Tensor, Tensor)
  std::tie(grad_input, grad_weight, grad_bias) = at::native_batch_norm_backward(
      grad_output,
      input,
      weight,
      running_mean_eager,
      running_var_eager,
      savemean,
      saveinvstd,
      training,
      eps,
      {true, true, true});

  // make JIT graph
  auto graph = std::make_shared<Graph>();
  auto training_val = graph->insertConstant(IValue(training));
  auto momentum_val = graph->insertConstant(IValue(momentum));
  auto eps_val = graph->insertConstant(IValue(eps));

  auto inputg = graph->addInput("self");
  auto weightg = graph->addInput("weight");
  auto biasg = graph->addInput("bias");
  auto running_meang = graph->addInput("running_mean");
  auto running_varg = graph->addInput("running_var");

  Value* bn = graph->insert(
      aten::native_batch_norm,
      {inputg,
       weightg,
       biasg,
       running_meang,
       running_varg,
       training_val,
       momentum_val,
       eps_val});
  auto outputs = bn->node()->outputs();
  for (auto output : outputs) {
    graph->registerOutput(output);
  }
  LowerAllTuples(graph);
  graph->lint();

  // differentiate JIT graph
  EliminateDeadCode(graph); // Tracing of some ops depends on the DCE trick
  ConstantPropagation(graph);
  auto grad_spec = differentiate(graph);
  LowerGradOf(*grad_spec.df);

  // prepare JIT inputs / gradients
  tensor_list tensors_in;
  tensors_in.push_back(input);
  tensors_in.push_back(weight);
  tensors_in.push_back(bias);
  tensors_in.push_back(running_mean_jit);
  tensors_in.push_back(running_var_jit);

  tensor_list tensor_grads_in;
  tensor_grads_in.push_back(grad_output);
  tensor_grads_in.push_back(grad_savemean);
  tensor_grads_in.push_back(grad_saveinvstd);

  // Get outputs from the interpreter
  tensor_list tensors_out, tensor_grads_out;
  std::tie(tensors_out, tensor_grads_out) =
      runGradient(grad_spec, tensors_in, tensor_grads_in);

  // prepare expected structs
  tensor_list expected_tensors_out, expected_tensor_grads_out;
  expected_tensors_out.push_back(output);
  expected_tensors_out.push_back(savemean);
  expected_tensors_out.push_back(saveinvstd);
  expected_tensors_out.push_back(running_mean_eager);
  expected_tensors_out.push_back(running_var_eager);
  expected_tensor_grads_out.push_back(grad_input);
  expected_tensor_grads_out.push_back(grad_weight);
  expected_tensor_grads_out.push_back(grad_bias);

  tensors_out.push_back(running_mean_jit);
  tensors_out.push_back(running_var_jit);

  // Compare results
  assertAllClose(tensors_out, expected_tensors_out);
  assertAllClose(tensor_grads_out, expected_tensor_grads_out);
}

void testCustomFusion() {
  auto graph = std::make_shared<Graph>();
  at::ScalarType s = at::ScalarType::Float;
  auto type = TensorType::create(s, at::kCPU, {2, 3, 4}, {12, 4, 1});
  auto a = SymbolicVariable::asNewInput(*graph, type);
  auto b = SymbolicVariable::asNewInput(*graph, type);
  auto c = a * b;
  auto d = c * a;
  graph->registerOutput(d.value());

  torch::jit::overrideCanFuseOnCPU(true);
  CustomFuseGraph(
      graph,
      [](Node* n) { return n->kind() != prim::Param; },
      Symbol::fromQualString("prim::FusionGroup"));
  torch::jit::overrideCanFuseOnCPU(false);

  const auto& nodes = graph->nodes();
  auto fusion_group =
      std::find_if(nodes.begin(), nodes.end(), [](const Node* node) {
        return node->kind() == Symbol::fromQualString("prim::FusionGroup");
      });
  AT_ASSERT(fusion_group != nodes.end());

  auto subgraph = fusion_group->g(attr::Subgraph);
  auto hits = 0;
  // two multiplications
  for (const auto& n : subgraph->nodes()) {
    (void)n;
    hits++;
  }
  AT_ASSERT(hits == 2);
}

void testCustomFusionNestedBlocks() {
  auto g = std::make_shared<Graph>();
  at::ScalarType s = at::ScalarType::Float;
  auto type = TensorType::create(s, at::kCPU, {2, 3, 4}, {12, 4, 1});

  // test CustomFusion in nested blocks;
  auto a = SymbolicVariable::asNewInput(*g, type);
  auto b = SymbolicVariable::asNewInput(*g, type);
  auto c = SymbolicVariable::asNewInput(*g, type);

  auto r =
      g->appendNode(g->create(prim::If, {c.value()}));
  auto then_block = r->addBlock();
  auto else_block = r->addBlock();
  {
    WithInsertPoint guard(then_block);
    auto d = c * a;
    auto t = d * b;
    then_block->registerOutput(t.value());
  }
  {
    WithInsertPoint guard(else_block);
    auto d = c + a;
    auto t = d + b;
    else_block->registerOutput(t.value());
  }
  g->registerOutput((Var(r->output()) + c).value());

  CustomFuseGraph(
      g,
      [](Node* n) { return n->kind() == aten::mul; },
      Symbol::fromQualString("prim::FusionGroup"));

  // Could be done in more efficient ways, but this is only a test.
  std::function<bool(const Block*, Symbol)> dfs = [&](const Block* b, Symbol s) {
      for (auto node : b->nodes()) {
          if (node->kind() == s)
              return true;
          for (auto nested_b : node->blocks())
              if (dfs(nested_b, s))
                  return true;
      }
      return false;
  };

  AT_ASSERT(dfs(g->block(), Symbol::fromQualString("prim::FusionGroup")));
}

static const auto cf_examples = R"JIT(
  def if_test(a, b):
      # FIXME: use 0 instead of a.
      # c = 0
      c = a
      if bool(a < b):
        c = b
      else:
        c = a
      return c
  def if_one(a, b):
    c = b
    if bool(a < b):
      c = a
    return c
  def while_test(a, i):
    while bool(i < 3):
      a *= a
      i += 1
    return a
)JIT";
void testControlFlow() {
  auto cu = compile(cf_examples);

  auto run = [&](const std::string& name, std::vector<IValue> stack) {
    auto graph = cu->get_function(name).graph();
    Code code(graph);
    InterpreterState interp(code);
    interp.run(stack);
    return stack;
  };

  auto L = [](int64_t l) {
    return IValue(autograd::make_variable(scalar_to_tensor(at::Scalar(l))));
  };
  auto V = [](IValue t) { return std::move(t).toTensor().item<int64_t>(); };
  auto run_binary = [&](const std::string& name, int64_t a, int64_t b) {
    return V(run(name, {L(a), L(b)})[0]);
  };
  ASSERT_EQ(2, run_binary("if_test", 1, 2));
  ASSERT_EQ(3, run_binary("if_test", 3, 2));
  ASSERT_EQ(2, run_binary("if_one", 2, 3));
  ASSERT_EQ(2, run_binary("if_one", 3, 2));
  ASSERT_EQ(256, run_binary("while_test", 2, 0));
}

void testProto() {
  ::ONNX_NAMESPACE::ModelProto proto;
  proto.set_producer_name("foo");
}

void testEvalModeForLoadedModule() {
  if (isSandcastle())
    return; // The module file to load is not generated in Sandcastle
  std::string module_path = "dropout_model.pt";
  torch::jit::script::Module module = torch::jit::load(module_path);
  AT_ASSERT(module.get_module("dropout").is_training());
  module.eval();
  AT_ASSERT(!module.get_module("dropout").is_training());
  module.train();
  AT_ASSERT(module.get_module("dropout").is_training());
}

// test a few features that are not directly used in schemas yet
void testSchemaParser() {
  // nested arrays
  auto s = parseSchema("at::what(int[][4] foo) -> ()");
  ASSERT_TRUE(s.arguments().at(0).N() == 4);
  ASSERT_TRUE(IntType::get()->isSubtypeOf(s.arguments()
                                              .at(0)
                                              .type()
                                              ->expect<ListType>()
                                              ->getElementType()
                                              ->expect<ListType>()
                                              ->getElementType()));
  auto s2 = parseSchema("at::what(int[][] foo) -> ()");
  ASSERT_TRUE(IntType::get()->isSubtypeOf(s2.arguments()
                                              .at(0)
                                              .type()
                                              ->expect<ListType>()
                                              ->getElementType()
                                              ->expect<ListType>()
                                              ->getElementType()));

  // named returns
  parseSchema("at::what(Tensor! i_will_be_written_to) -> ()");
  auto s3 =
      parseSchema("at::what() -> (Tensor the_return, Tensor the_return2)");
  ASSERT_TRUE(s3.returns().at(0).name() == "the_return");
  ASSERT_TRUE(s3.returns().at(1).name() == "the_return2");

  // futures
  auto s4 = parseSchema("at::what(Future(int) foo) -> ()");
  ASSERT_TRUE(IntType::get()->isSubtypeOf(
      s4.arguments().at(0).type()->expect<FutureType>()->getElementType()));

  // test tensor with annotated alias sets
  parseSchema("at::what(Tensor(a) foo) -> (Tensor(a))");

  {
    const auto s = parseSchema(
        "at::what(Tensor(b|c)[](a!) list, Tensor(c) element)"
        " -> (Tensor(b|c)[](a!))");

    // The list itself is annotated with `a`
    const auto& aliasInfo = *s.arguments().at(0).alias_info();
    ASSERT_TRUE(
        aliasInfo.beforeSets() ==
        std::unordered_set<Symbol>{Symbol::fromQualString("alias::a")});
    ASSERT_TRUE(aliasInfo.isWrite());

    // Check the contained types
    ASSERT_TRUE(!aliasInfo.containedTypes().empty());
    const auto& containedAliasInfo = aliasInfo.containedTypes()[0];
    const auto expected = std::unordered_set<Symbol>{
        Symbol::fromQualString("alias::b"),
        Symbol::fromQualString("alias::c"),
    };
    ASSERT_TRUE(containedAliasInfo.beforeSets() == expected);
    ASSERT_TRUE(containedAliasInfo.afterSets() == expected);
    ASSERT_FALSE(containedAliasInfo.isWrite());
  }
  {
    const auto s = parseSchema(
        "at::what(Tensor(b -> b|c)[](a!) list, Tensor(c) element)"
        " -> (Tensor(b|c)[](a!))");

    // The list itself is annotated with `a`
    const auto& aliasInfo = *s.arguments().at(0).alias_info();
    ASSERT_EQ(
        aliasInfo.beforeSets(),
        std::unordered_set<Symbol>{Symbol::fromQualString("alias::a")});
    ASSERT_EQ(
        aliasInfo.afterSets(),
        std::unordered_set<Symbol>{Symbol::fromQualString("alias::a")});
    ASSERT_TRUE(aliasInfo.isWrite());
    ASSERT_EQ(aliasInfo.containedTypes().size(), 1);

    // Check the contained types
    ASSERT_TRUE(!aliasInfo.containedTypes().empty());
    const auto& containedAliasInfo = aliasInfo.containedTypes()[0];
    const auto expectedBefore = std::unordered_set<Symbol>{
        Symbol::fromQualString("alias::b"),
    };
    const auto expectedAfter = std::unordered_set<Symbol>{
        Symbol::fromQualString("alias::b"), Symbol::fromQualString("alias::c")};
    ASSERT_TRUE(containedAliasInfo.beforeSets() == expectedBefore);
    ASSERT_TRUE(containedAliasInfo.afterSets() == expectedAfter);
    ASSERT_FALSE(containedAliasInfo.isWrite());
  }
}

void testTopologicalIndex() {
  {
    Graph graph;
    auto node1 = graph.create(prim::AutogradZero);
    auto node2 = graph.create(prim::AutogradZero);
    auto node3 = graph.create(prim::AutogradZero);
    auto node4 = graph.create(prim::AutogradZero);

    graph.appendNode(node4);
    graph.prependNode(node1);
    node2->insertAfter(node1);
    node3->insertBefore(node4);

    // nodes should be in numerical order
    ASSERT_TRUE(node1->isBefore(node2));
    ASSERT_TRUE(node1->isBefore(node3));
    ASSERT_TRUE(node1->isBefore(node4));
    ASSERT_TRUE(node2->isAfter(node1));
    ASSERT_TRUE(node2->isBefore(node3));
    ASSERT_TRUE(node2->isBefore(node4));
    ASSERT_FALSE(node3->isBefore(node1));
    ASSERT_FALSE(node3->isBefore(node2));
    ASSERT_FALSE(node3->isAfter(node4));

    // Built up a block structure
    //  node3
    //   /\        ...
    //  A  B     block1
    //      \      ...
    //      C    block2
    auto block1 = node3->addBlock();
    auto A = graph.create(prim::AutogradZero);
    block1->appendNode(A);
    auto B = graph.create(prim::AutogradZero);
    block1->appendNode(B);
    auto block2 = B->addBlock();
    auto C = graph.create(prim::AutogradZero);
    block2->appendNode(C);

    // Check isAfter on different block levels
    ASSERT_TRUE(node1->isBefore(A));
    ASSERT_TRUE(A->isBefore(B));
    ASSERT_TRUE(A->isBefore(C));

    // make sure things don't blow up on deletions
    node2->destroy();
    auto node2p = graph.create(prim::AutogradZero);
    node2p->insertAfter(node1);
    ASSERT_TRUE(node1->isBefore(node2p));
    ASSERT_TRUE(node2p->isBefore(node3));
  }
  {
    // Induce reindexing to test that path
    Graph graph;
    std::map<size_t, Node*> nodes;

    auto anchor = graph.create(prim::AutogradZero);
    graph.appendNode(anchor);
    // Inserting to the same place a lot will trigger reindexing
    for (auto i = 0; i < 100; ++i) {
      auto n = graph.create(prim::AutogradZero);
      n->insertAfter(anchor);
      nodes[i] = n;
    }

    // Nodes should be in reverse order
    for (auto i = 0; i < 100; ++i) {
      for (auto j = i + 1; j < 100; ++j) {
        ASSERT_TRUE(nodes[i]->isAfter(nodes[j]));
      }
    }
  }
}

at::Tensor invokeTestRecordFunction(at::Tensor& t) {
  RECORD_FUNCTION("test", std::vector<c10::IValue>({t}));

  auto t2 = t.pow(2);
  return t2;
}

static const auto invokeTestRecordFunction_JIT = R"JIT(
  def forward(t):
    t2 = t.pow(2)
    return t2
)JIT";

at::Tensor invokeTestRecordFunctionJIT(at::Tensor& t) {
  RECORD_FUNCTION("test", std::vector<c10::IValue>({t}));

  auto cu = compile(invokeTestRecordFunction_JIT);
  return cu->get_function("forward")({t}).toTensor();
}

using TracedTestInputs =
    std::vector<std::tuple<std::string, std::vector<std::vector<int64_t>>>>;

void checkTracedInputs(const TracedTestInputs& inputs) {
  bool found_test = false;
  bool found_pow = false;
  bool found_mul = false;
  for (const auto& input : inputs) {
    const auto& fn = std::get<0>(input);
    const auto& sizes = std::get<1>(input);
    if (fn == "test") {
      found_test = true;
      TORCH_CHECK(sizes.size() == 1);
      TORCH_CHECK(sizes[0] == std::vector<int64_t>({1, 2, 3}));
    } else if (fn == "test::pow") {
      found_pow = true;
      TORCH_CHECK(sizes.size() == 2);
      TORCH_CHECK(sizes[0] == std::vector<int64_t>({1, 2, 3}));
      TORCH_CHECK(sizes[1].empty());
    } else if (fn.find("::mul") != std::string::npos) {
      found_mul = true;
      TORCH_CHECK(sizes.size() > 1);
      TORCH_CHECK(sizes[0] == std::vector<int64_t>({1, 2, 3}));
    }
  }
  TORCH_CHECK(found_test);
  TORCH_CHECK(found_pow);
  TORCH_CHECK(found_mul);
}

std::string getFullName(const autograd::profiler::RecordFunction* fn_ptr) {
  std::string full_name = "";
  while (fn_ptr != nullptr) {
    if (!full_name.empty()) {
      full_name = std::string(fn_ptr->name().str()) + "::" + full_name;
    } else {
      full_name = fn_ptr->name().str();
    }
    fn_ptr = fn_ptr->parent();
  }
  return full_name;
}

void testRecordFunction() {
  // [(fn, [[sizes], [sizes], ...]), ...]
  TracedTestInputs traced_inputs;
  autograd::profiler::pushCallback(
      [&traced_inputs](const autograd::profiler::RecordFunction& fn) {
        auto inputs = fn.inputs();
        std::vector<std::vector<int64_t>> sizes;
        for (const auto& input : inputs) {
          if (input.isTensor()) {
            sizes.push_back(input.toTensor().sizes().vec());
          } else if (input.isScalar()) {
            sizes.push_back(std::vector<int64_t>());
          }
        }
        traced_inputs.push_back(
            std::make_tuple(std::string(getFullName(&fn)), sizes));
      },
      [](const autograd::profiler::RecordFunction&) {},
      /* needs_inputs */ true);

  auto t = torch::randn({1, 2, 3}, at::kCPU);
  t.set_requires_grad(true);
  auto t2 = invokeTestRecordFunction(t);
  t2.backward();
  auto eager_inputs = traced_inputs;
  traced_inputs.clear();

  t = torch::randn({1, 2, 3}, at::kCPU);
  t.set_requires_grad(true);
  t2 = invokeTestRecordFunctionJIT(t);
  t2.backward();
  auto jit_inputs = traced_inputs;
  traced_inputs.clear();

  autograd::profiler::popCallback();

  checkTracedInputs(eager_inputs);
  checkTracedInputs(jit_inputs);

  // test sampled callbacks
  int sampled_cb_ctr = 0;
  autograd::profiler::pushCallback(
      [&sampled_cb_ctr](const autograd::profiler::RecordFunction& fn) {
        if (std::string(fn.name().str()) == "test") {
          ++sampled_cb_ctr;
        }
      },
      [](const autograd::profiler::RecordFunction&) {},
      /* needs_inputs */ false,
      /* sampled */ true);

  int non_sampled_cb_ctr = 0;
  autograd::profiler::pushCallback(
      [&non_sampled_cb_ctr](const autograd::profiler::RecordFunction& fn) {
        if (std::string(fn.name().str()) == "test") {
          ++non_sampled_cb_ctr;
        }
      },
      [](const autograd::profiler::RecordFunction&) {},
      /* needs_inputs */ false,
      /* sampled */ false);

  auto run_test_function = []() {
    auto t = torch::randn({1, 2, 3}, at::kCPU);
    for (auto k = 0; k < 1000; k++) {
      invokeTestRecordFunction(t);
    }
  };

  autograd::profiler::setSamplingProbability(0.5);
  run_test_function();

  TORCH_CHECK(non_sampled_cb_ctr == 1000);
  TORCH_CHECK(sampled_cb_ctr > 0 && sampled_cb_ctr < 1000);

  sampled_cb_ctr = 0;
  autograd::profiler::setSamplingProbability(0.0);
  run_test_function();

  TORCH_CHECK(non_sampled_cb_ctr == 2000);
  TORCH_CHECK(sampled_cb_ctr == 0);

  sampled_cb_ctr = 0;
  autograd::profiler::setSamplingProbability(1.0);
  run_test_function();

  TORCH_CHECK(non_sampled_cb_ctr == 3000);
  TORCH_CHECK(sampled_cb_ctr == 1000);

  autograd::profiler::popCallback();
  autograd::profiler::popCallback();
}

class TestThreadLocalDebugInfo
  : public at::ThreadLocalDebugInfoBase {
 public:
  int getModelId() const {
    return model_id_;
  }

  void setModelId(int model_id) {
    model_id_ = model_id;
  }

  virtual ~TestThreadLocalDebugInfo() {}

 private:
  int model_id_ = 0;
};

void testThreadLocalDebugInfo() {
  auto checkDebugInfo = [](){
    auto debug_info = at::getThreadLocalDebugInfo();
    TORCH_CHECK(debug_info != nullptr);
    auto* test_debug_info = dynamic_cast<TestThreadLocalDebugInfo*>(
        debug_info.get());
    TORCH_CHECK(test_debug_info != nullptr);
    TORCH_CHECK(test_debug_info->getModelId() == 42);
  };

  TORCH_CHECK(at::getThreadLocalDebugInfo() == nullptr);
  auto debug_info = std::make_shared<TestThreadLocalDebugInfo>();
  debug_info->setModelId(42);
  at::setThreadLocalDebugInfo(debug_info);

  checkDebugInfo();

  // check that thread local debug info is propagated through fork calls
  std::atomic<bool> done {false};
  at::launch([checkDebugInfo, &done](){
    checkDebugInfo();
    done = true;
  });
  while (!done) {}
  checkDebugInfo();

  // check that thread local debug info is propagated through backward pass
  autograd::profiler::pushCallback(
      [&checkDebugInfo](const autograd::profiler::RecordFunction& fn) {
        checkDebugInfo();
      },
      [](const autograd::profiler::RecordFunction&) {});
  {
    auto t = torch::randn({1, 2, 3}, at::kCPU);
    t.set_requires_grad(true);
    auto t2 = t.pow(2);
    t2.backward();
  }
  autograd::profiler::popCallback();

  checkDebugInfo();
  at::setThreadLocalDebugInfo(nullptr);
  TORCH_CHECK(at::getThreadLocalDebugInfo() == nullptr);
}

void testAutogradProfiler() {
  constexpr int batch_size = 4;
  constexpr int input_size = 256;
  constexpr int seq_len = 32;

  int hidden_size = 2 * input_size;
  auto input = torch::randn({seq_len, batch_size, input_size}, at::kCPU);
  auto hx = torch::randn({batch_size, hidden_size}, at::kCPU);
  auto cx = torch::randn({batch_size, hidden_size}, at::kCPU);
  auto w_ih = t_def(torch::randn({4 * hidden_size, input_size}, at::kCPU));
  auto w_hh = t_def(torch::randn({4 * hidden_size, hidden_size}, at::kCPU));

  std::stringstream ss;
  {
    autograd::profiler::RecordProfile guard(ss);
    for (size_t i = 0; i < 100; ++i) {
      std::tie(hx, cx) = lstm(input[0], hx, cx, w_ih, w_hh);
    }
  }

  std::string result = ss.str();
  size_t count = 0;
  for (size_t pos = 0; (pos = result.find("tanh", pos)) != std::string::npos;
       count++, pos++) {
  }
  TORCH_CHECK(count == 200);
}

void testNoneSchemaMatch() {
  RegisterOperators reg({
      Operator(
          "prim::test_none() -> int?",
          [](const Node* node) {
            return [](Stack& stack) {
              push(stack, IValue());
              return 0;
            };
          },
          aliasAnalysisFromSchema()),
      Operator(
          "prim::is_none(int? a) -> bool",
          [](const Node* node) {
            return [](Stack& stack) {
              IValue a = pop(stack);
              if (a.isNone()) {
                push(stack, true);
              } else {
                push(stack, false);
              }
              return 0;
            };
          },
          aliasAnalysisFromSchema()),
  });

  // Constant propagation will run test_none and produce a None,
  // testing that its type is set appropriately and schema matching  doesn't
  // fail when running is_none

  auto r = std::make_shared<Graph>();
  auto& g = *r;
  auto opt_int = g.insert(Symbol::fromQualString("prim::test_none"), {});
  auto out_bool = g.insert(Symbol::fromQualString("prim::is_none"), {opt_int});
  g.registerOutput(out_bool);
  ConstantPropagation(r);

  auto nodes = r->block()->nodes();
  // checking that constant propagation ran wo/failure
  AT_ASSERT(std::distance(nodes.begin(), nodes.end()) == 1);
}

void testModuleDefine() {
  script::Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    def add_it(self, x, b : int = 4):
      return self.foo + x + b
  )");
  auto result = m.run_method("add_it", torch::ones({}));
  AT_ASSERT(result.toTensor().item<float>() == 6);
}

void testModuleConversion() {
  script::Module m("test");
  {
    // test cuda to cpu for params and buffers
    m.register_parameter("foo", torch::ones({}, at::kCUDA), false);
    m.register_buffer("bar", torch::ones({}, at::kCUDA));

    m.to(at::kCUDA);
    m.to(at::kCPU);
    AT_ASSERT(m.get_parameter("foo").device().is_cpu());
    AT_ASSERT(m.get_buffer("bar").device().is_cpu());
  }
  {
    // test cpu to cuda for params and buffers
    m.register_parameter("foo", torch::ones({}), false);
    m.register_buffer("bar", torch::ones({}));

    m.to(at::kCUDA);
    AT_ASSERT(m.get_parameter("foo").device().is_cuda());
    AT_ASSERT(m.get_buffer("bar").device().is_cuda());
  }
}

static int testPassValue = 0;
void fakePass(std::shared_ptr<Graph>& g) {
  testPassValue++;
  return;
}

RegisterPass p(fakePass);

void testPassManagement() {
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  script::parseIR(
      R"IR(
graph(%a):
  return (%a))IR",
      &*graph);

  std::vector<IValue> stack = {IValue(torch::randn({22}, at::kCPU))};
  auto run = [&](std::shared_ptr<Graph>& graph, std::vector<IValue> stack) {
    GraphExecutor executor(graph);
    executor.run(stack);
    return stack;
  };
  run(graph, stack);
  AT_ASSERT(testPassValue);
}

static void checkShape(
    Node* n,
    std::vector<int64_t> expected,
    bool prev = true) {
  auto profile = (prev) ? n->inputs().at(0)->node() : n;
  auto tp = profile->output()->type();
  auto ptp = tp->expect<TensorType>();
  ASSERT_EQ(ptp->sizes().concrete_sizes().value(), expected);
}

void testInsertAndEliminateRedundantGuards() {
  static const auto basic_example = R"JIT(
  def basic(x, y):
    a = x + y
    b = x * y
    c = x + 1
    d = a - c
    e = b - c
    return d + e
  )JIT";

  auto cu = compile(basic_example);
  auto& fun = cu->get_function("basic");
  auto pr = ProfilingRecord::instrumentGraph(fun.graph());
  auto x = at::randn({2, 3}, at::kCPU);
  auto y = at::randn({2, 3}, at::kCPU);
  auto v = [](at::Tensor t) { return autograd::make_variable(t, false); };
  auto stack = createStack({v(x), v(y)});
  // introduce some profiling information
  Code cd(pr->profiled_graph_);
  InterpreterState is{cd};
  is.run(stack);
  auto copy = pr->profiled_graph_->copy();
  InsertGuards(copy);
  auto nodes = copy->block()->nodes();
  auto guard = std::find_if(nodes.begin(), nodes.end(), [](Node* n) {
    return n->kind() == prim::Guard;
  });
  ASSERT_NE(guard, nodes.end());
  ASSERT_EQ(guard->input()->type()->cast<TensorType>(), nullptr);
  checkShape(*guard, {2, 3}, false);
  auto is_guard = [](Node* n) { return n->kind() == prim::Guard; };
  int num_guards = std::count_if(nodes.begin(), nodes.end(), is_guard);
  ASSERT_EQ(num_guards, 11);
  // now eliminate as many guards as possible
  // we should be left with two guards on x and y's defs
  EliminateRedundantGuards(copy);
  num_guards = std::count_if(nodes.begin(), nodes.end(), is_guard);
  ASSERT_EQ(num_guards, 2);
}

void testInsertBailOuts() {
  static const auto basic_example = R"JIT(
  def basic_loop(x, y):

      a = x + 1
      b = y + 2
      c = x + y + 3

      for i in range(10):
          a = a + b
          # invariant
          d = b * c
          #
          a = a - d

      e = a + 4
      return e
  )JIT";

  auto cu = compile(basic_example);
  auto& fun = cu->get_function("basic_loop");
  auto pr = ProfilingRecord::instrumentGraph(fun.graph());
  auto x = at::randn({2, 3}, at::kCPU);
  auto y = at::randn({2, 3}, at::kCPU);
  auto v = [](at::Tensor t) { return autograd::make_variable(t, false); };
  auto stack = createStack({v(x), v(y)});
  // introduce some profiling information
  Code cd(pr->profiled_graph_);
  InterpreterState is{cd};
  is.run(stack);
  auto copy = pr->profiled_graph_->copy();
  InsertGuards(copy);
  EliminateRedundantGuards(copy);
  auto nodes = copy->block()->nodes();
  auto is_guard = [](Node* n) { return n->kind() == prim::Guard; };
  auto num_guards = std::count_if(nodes.begin(), nodes.end(), is_guard);
  ASSERT_EQ(num_guards, 3);
  InsertBailOuts(copy);
  auto is_bailout = [](Node* n) { return n->kind() == prim::BailOut; };
  auto num_bailouts = std::count_if(nodes.begin(), nodes.end(), is_bailout);
  ASSERT_EQ(num_guards, num_bailouts);
  std::vector<Node*> bailouts(num_bailouts);
  std::copy_if(nodes.begin(), nodes.end(), bailouts.begin(), is_bailout);

  for (auto blo : bailouts) {
    ASSERT_EQ(blo->inputs().at(0)->node()->kind(), prim::BailoutTemplate);
  }
}

void testProfiler() {
  constexpr int batch_size = 4;
  constexpr int input_size = 256;

  int hidden_size = 2 * input_size;

  auto v = [](at::Tensor t) { return autograd::make_variable(t, false); };

  auto input = at::randn({batch_size, input_size}, at::kCPU);
  auto hx = at::randn({batch_size, hidden_size}, at::kCPU);
  auto cx = at::randn({batch_size, hidden_size}, at::kCPU);
  auto w_ih = t_def(at::randn({4 * hidden_size, input_size}, at::kCPU));
  auto w_hh = t_def(at::randn({4 * hidden_size, hidden_size}, at::kCPU));

  auto g = build_lstm();
  auto stack = createStack({v(input), v(hx), v(cx), v(w_ih), v(w_hh)});

  auto& opt_graph = *g.get();
  ArgumentSpecCreator arg_spec_creator(opt_graph);
  ArgumentSpec spec =
      arg_spec_creator.create(autograd::GradMode::is_enabled(), stack);
  arg_spec_creator.specializeTypes(opt_graph, spec);
  auto pr = ProfilingRecord::instrumentGraph(g);
  Code cd(pr->profiled_graph_);
  InterpreterState is{cd};
  is.run(stack);

  auto begin = pr->profiled_graph_->block()->nodes().begin();
  auto end = pr->profiled_graph_->block()->nodes().end();
  auto mm =
      std::find_if(begin, end, [](Node* n) { return n->kind() == aten::mm; });
  ASSERT_NE(mm, end);
  std::vector<int64_t> mm_expected{4, 256};
  std::vector<int64_t> eltwise{4, 512};
  checkShape(*mm, mm_expected);
  auto sigmoid_n = std::find_if(
      begin, end, [](Node* n) { return n->kind() == aten::sigmoid; });
  ASSERT_NE(sigmoid_n, end);
  checkShape(*sigmoid_n, eltwise);
  auto tanh_n =
      std::find_if(begin, end, [](Node* n) { return n->kind() == aten::tanh; });
  checkShape(*tanh_n, eltwise);
}

void testInsertConstant() {
  Graph g;
  ASSERT_THROWS_WITH(
      insertConstant(
          g, IValue(), TensorType::get(), c10::nullopt, c10::nullopt),
      "Expected OptionalType");
}

} // namespace test
} // namespace jit
} // namespace torch
