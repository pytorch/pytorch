#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>

#include "test/cpp/jit/test_base.h"
#include "test/cpp/jit/test_utils.h"

#include <torch/csrc/jit/ir/type_hashing.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/codegen/fuser/interface.h"
#include "torch/csrc/jit/frontend/code_template.h"
#include "torch/csrc/jit/frontend/tracer.h"
#include "torch/csrc/jit/ir/alias_analysis.h"
#include "torch/csrc/jit/ir/attributes.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/scope.h"
#include "torch/csrc/jit/passes/bailout_graph.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/graph_fuser.h"
#include "torch/csrc/jit/passes/guard_elimination.h"
#include "torch/csrc/jit/passes/inline_autodiff_subgraphs.h"
#include "torch/csrc/jit/passes/insert_guards.h"
#include "torch/csrc/jit/passes/liveness.h"
#include "torch/csrc/jit/passes/lower_grad_of.h"
#include "torch/csrc/jit/passes/lower_tuples.h"
#include "torch/csrc/jit/passes/pass_manager.h"
#include "torch/csrc/jit/passes/requires_grad_analysis.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/passes/utils/subgraph_utils.h"
#include "torch/csrc/jit/runtime/argument_spec.h"
#include "torch/csrc/jit/runtime/autodiff.h"
#include "torch/csrc/jit/runtime/custom_operator.h"
#include "torch/csrc/jit/runtime/interpreter.h"
#include "torch/csrc/jit/runtime/symbolic_script.h"
#include "torch/csrc/jit/serialization/import.h"

#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/variable.h"

#include <torch/csrc/jit/testing/file_check.h>
#include <torch/script.h>
#include "torch/csrc/jit/api/module.h"
#include "torch/csrc/jit/frontend/ir_emitter.h"
#include "torch/csrc/jit/runtime/profiling_record.h"
#include "torch/jit.h"

#include "onnx/onnx_pb.h"

#include <c10/util/Exception.h>
#include <c10/util/ThreadLocalDebugInfo.h>

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

using namespace torch::autograd::profiler;

namespace torch {
namespace jit {
inline c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

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
  at::Tensor grad_output =
      torch::randn_like(output, at::MemoryFormat::Preserve);
  at::Tensor grad_finput =
      torch::zeros_like(finput, at::MemoryFormat::Preserve);
  at::Tensor grad_fgradinput =
      torch::zeros_like(fgradinput, at::MemoryFormat::Preserve);

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
  auto ksz_val = graph->insertConstant(kernel_size);
  auto kst_val = graph->insertConstant(stride);
  auto pad_val = graph->insertConstant(padding);

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
  at::Tensor grad_output =
      torch::randn_like(output, at::MemoryFormat::Preserve);
  at::Tensor grad_savemean =
      torch::zeros_like(savemean, at::MemoryFormat::Preserve);
  at::Tensor grad_saveinvstd =
      torch::zeros_like(saveinvstd, at::MemoryFormat::Preserve);

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
  auto graph_string = R"IR(
    graph(%0 : Float(2, 3, 4),
          %1 : Float(2, 3, 4)):
      %2 : Tensor = aten::mul(%0, %1)
      %3 : Tensor = aten::mul(%2, %0)
      return (%3))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());

  torch::jit::overrideCanFuseOnCPU(true);
  CustomFuseGraph(
      g,
      [](Node* n) { return n->kind() != prim::Param; },
      Symbol::fromQualString("prim::FusionGroup"));
  torch::jit::overrideCanFuseOnCPU(false);

  const auto& nodes = g->nodes();
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
  auto graph_string = R"IR(
  graph(%0 : Float(2, 3, 4),
        %1 : Float(2, 3, 4),
        %2 : Float(2, 3, 4)):
    %3 : int = prim::Constant[value=1]()
    %4 : Tensor = prim::If(%2)
      block0():
        %5 : Tensor = aten::mul(%0, %2)
        %6 : Tensor = aten::mul(%5, %1)
        -> (%6)
      block1():
        %7 : Tensor = aten::add(%0, %2, %3)
        %8 : Tensor = aten::add(%7, %1, %3)
        -> (%8)
    %9 : Tensor = aten::add(%4, %2, %3)
    return (%4))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());

  CustomFuseGraph(
      g,
      [](Node* n) { return n->kind() == aten::mul; },
      Symbol::fromQualString("prim::FusionGroup"));

  // Could be done in more efficient ways, but this is only a test.
  std::function<bool(const Block*, Symbol)> dfs = [&](const Block* b,
                                                      Symbol s) {
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
    Code code(graph, "");
    InterpreterState interp(code);
    interp.run(stack);
    return stack;
  };

  auto L = [](int64_t l) { return IValue(scalar_to_tensor(at::Scalar(l))); };
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
  torch::jit::Module module = torch::jit::load(module_path);
  AT_ASSERT(module.attr("dropout").toModule().is_training());
  module.eval();
  AT_ASSERT(!module.attr("dropout").toModule().is_training());
  module.train();
  AT_ASSERT(module.attr("dropout").toModule().is_training());
}

void testSerializationInterop() {
  if (isSandcastle()) {
    // The module file to load is not generated in Sandcastle
    return;
  }

  // This should be generated by `test/cpp/jit/tests_setup.py`
  std::ifstream input_stream("ivalue.pt");
  std::vector<char> input;
  input.insert(
      input.begin(),
      std::istream_iterator<char>(input_stream),
      std::istream_iterator<char>());
  IValue ivalue = pickle_load(input);

  auto elements = ivalue.toTuple()->elements();
  auto ones = torch::ones({2, 2});
  ASSERT_TRUE(ones.equal(elements.at(0).toTensor()));

  auto twos = torch::ones({3, 5}) * 2;
  ASSERT_TRUE(twos.equal(elements.at(1).toTensor()));
}

void testTorchSaveError() {
  if (isSandcastle()) {
    // The file to load is not generated in Sandcastle
    return;
  }

  // This should be generated by `test/cpp/jit/tests_setup.py`
  bool passed = true;
  try {
    torch::jit::load("eager_value.pt");
    passed = false;
  } catch (const std::exception& c) {
  }
  // Ensure torch::jit::load did not run
  ASSERT_TRUE(passed);
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
  def foo(self, t):
    t2 = t.pow(2)
    return t2

  def forward(self, t):
    return self.foo(t)
)JIT";

at::Tensor invokeTestRecordFunctionJIT(at::Tensor& t) {
  RECORD_FUNCTION("test", std::vector<c10::IValue>({t}));

  auto module = std::make_shared<script::Module>(
      "RecordFunctionTestModule", std::make_shared<script::CompilationUnit>());
  module->define(invokeTestRecordFunction_JIT);
  return module->forward({t}).toTensor();
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
    } else if (fn == "pow") {
      found_pow = true;
      TORCH_CHECK(sizes.size() == 2);
      TORCH_CHECK(sizes[0] == std::vector<int64_t>({1, 2, 3}));
      TORCH_CHECK(sizes[1].empty());
    } else if (fn == "mul") {
      found_mul = true;
      TORCH_CHECK(sizes.size() > 1);
      TORCH_CHECK(sizes[0] == std::vector<int64_t>({1, 2, 3}));
    }
  }
  TORCH_CHECK(found_test);
  TORCH_CHECK(found_pow);
  TORCH_CHECK(found_mul);
}

using namespace torch::autograd;

void checkScopeCallbacks() {
  bool found_function_scope = false;
  bool found_method_scope = false;
  bool found_user_scope = false;
  at::addGlobalCallback(at::RecordFunctionCallback(
      [&](const at::RecordFunction& fn) {
        if (fn.scope() == at::RecordScope::FUNCTION &&
            std::string(fn.name().str()) == "test_function") {
          found_function_scope = true;
        }
        if (fn.scope() == at::RecordScope::TORCHSCRIPT_FUNCTION &&
            std::string(fn.name().str()) == "test_method") {
          found_method_scope = true;
        }
        if (fn.scope() == at::RecordScope::USER_SCOPE &&
            std::string(fn.name().str()) == "test_user_scope") {
          found_user_scope = true;
        }
      },
      [](const at::RecordFunction&) {}));

  bool bad_scope = false;
  auto pushScopedCallback = [&](at::RecordScope scope, size_t& cnt) {
    at::addGlobalCallback(
        at::RecordFunctionCallback(
            [&bad_scope, &cnt, scope](const at::RecordFunction& fn) {
              if (fn.scope() == scope) {
                ++cnt;
              } else {
                bad_scope = true;
              }
              return true;
            },
            [](const at::RecordFunction&) {})
            .scopes({scope}));
  };

  size_t fun_cnt = 0;
  pushScopedCallback(at::RecordScope::FUNCTION, fun_cnt);
  size_t ts_fun_cnt = 0;
  pushScopedCallback(at::RecordScope::TORCHSCRIPT_FUNCTION, ts_fun_cnt);
  size_t user_scope_cnt = 0;
  pushScopedCallback(at::RecordScope::USER_SCOPE, user_scope_cnt);

  TORCH_CHECK(at::hasCallbacks());

  {
    RECORD_TORCHSCRIPT_FUNCTION("test_method", {});
    { RECORD_FUNCTION("test_function", {}); }
    { RECORD_USER_SCOPE("test_user_scope"); }
  }

  TORCH_CHECK(!bad_scope);
  TORCH_CHECK(fun_cnt == 1);
  TORCH_CHECK(ts_fun_cnt == 1);
  TORCH_CHECK(user_scope_cnt == 1);

  TORCH_CHECK(found_function_scope);
  TORCH_CHECK(found_method_scope);
  TORCH_CHECK(found_user_scope);
}

void testRecordFunction() {
  // enable observers
  c10::impl::IncludeDispatchKeyGuard observer_guard(c10::DispatchKey::Profiler);
  // disabling the inlining of method calls
  GraphOptimizerEnabledGuard opt_guard(false);

  // [(fn, [[sizes], [sizes], ...]), ...]
  TracedTestInputs traced_inputs;
  std::unordered_set<std::string> ts_names;
  addGlobalCallback(
      RecordFunctionCallback(
          [&](const RecordFunction& fn) {
            if (fn.scope() == RecordScope::FUNCTION) {
              auto inputs = fn.inputs();
              std::vector<std::vector<int64_t>> sizes;
              for (const auto& input : inputs) {
                if (input.isTensor()) {
                  sizes.push_back(input.toTensor().sizes().vec());
                } else if (input.isScalar()) {
                  sizes.push_back(std::vector<int64_t>());
                }
              }
              traced_inputs.push_back(std::make_tuple(fn.name().str(), sizes));
            } else if (fn.scope() == RecordScope::TORCHSCRIPT_FUNCTION) {
              ts_names.insert(fn.name().str());
            }
          },
          [](const RecordFunction&) {})
          .needsInputs(true));

  TracedTestInputs eager_inputs, jit_inputs;
  {
    auto t = torch::randn({1, 2, 3}, at::kCPU);
    t.set_requires_grad(true);
    auto t2 = invokeTestRecordFunction(t);
    t2.backward(torch::ones_like(t2, at::MemoryFormat::Preserve));
    eager_inputs = traced_inputs;
    traced_inputs.clear();

    TORCH_CHECK(ts_names.empty());

    t = torch::randn({1, 2, 3}, at::kCPU);
    t.set_requires_grad(true);
    t2 = invokeTestRecordFunctionJIT(t);
    t2.backward(torch::ones_like(t2, at::MemoryFormat::Preserve));
    jit_inputs = traced_inputs;
    traced_inputs.clear();
  }

  TORCH_CHECK(ts_names.size() == 2);
  TORCH_CHECK(ts_names.find("forward") != ts_names.end());
  TORCH_CHECK(ts_names.find("foo") != ts_names.end());

  checkTracedInputs(eager_inputs);
  checkTracedInputs(jit_inputs);
  at::clearCallbacks();

  // test sampled callbacks
  int sampled_cb_ctr = 0;
  auto setup_sampled_callback = [&sampled_cb_ctr](double sampling_prob) {
    return addGlobalCallback(RecordFunctionCallback(
                                 [&sampled_cb_ctr](const RecordFunction& fn) {
                                   if (std::string(fn.name().str()) == "test") {
                                     ++sampled_cb_ctr;
                                   }
                                   return true;
                                 },
                                 [](const RecordFunction&) {})
                                 .samplingProb(sampling_prob));
  };

  int non_sampled_cb_ctr = 0;
  addGlobalCallback(RecordFunctionCallback(
      [&non_sampled_cb_ctr](const RecordFunction& fn) {
        if (std::string(fn.name().str()) == "test") {
          ++non_sampled_cb_ctr;
        }
        return true;
      },
      [](const RecordFunction&) {}));

  auto handle = setup_sampled_callback(0.5);

  auto run_test_function = []() {
    auto t = torch::randn({1, 2, 3}, at::kCPU);
    for (auto k = 0; k < 1000; k++) {
      invokeTestRecordFunction(t);
    }
  };

  run_test_function();
  TORCH_CHECK(non_sampled_cb_ctr == 1000);
  TORCH_CHECK(sampled_cb_ctr > 0 && sampled_cb_ctr < 1000);

  sampled_cb_ctr = 0;
  removeCallback(handle);
  handle = setup_sampled_callback(0.0);
  run_test_function();

  TORCH_CHECK(non_sampled_cb_ctr == 2000);
  TORCH_CHECK(sampled_cb_ctr == 0);

  sampled_cb_ctr = 0;
  removeCallback(handle);
  handle = setup_sampled_callback(1.0);
  run_test_function();

  TORCH_CHECK(non_sampled_cb_ctr == 3000);
  TORCH_CHECK(sampled_cb_ctr == 1000);
  clearCallbacks();

  // test the scope of the callbacks
  checkScopeCallbacks();
  clearCallbacks();

  // check record function guard
  std::vector<std::string> fn_names;
  std::mutex mtx;
  addGlobalCallback(RecordFunctionCallback(
      [&fn_names, &mtx](const RecordFunction& fn) {
        std::lock_guard<std::mutex> lock(mtx);
        fn_names.push_back(fn.name().str());
        return true;
      },
      [](const RecordFunction&) {}));
  {
    RecordFunctionGuard g1(false);
    {
      RECORD_USER_SCOPE("A");
      {
        RecordFunctionGuard g2(true);
        RECORD_USER_SCOPE("B");
        {
          DisableRecordFunctionGuard g3;
          RECORD_USER_SCOPE("C");
        }
      }
      { RECORD_USER_SCOPE("D"); }
    }
  }
  TORCH_CHECK(fn_names.size() == 1);
  TORCH_CHECK(fn_names[0] == "B");
  clearCallbacks();

  // test add/remove
  std::vector<size_t> ids;
  auto add_remove_test_add_cb = [&ids](size_t id) {
    return addGlobalCallback(RecordFunctionCallback(
        [&ids, id](const RecordFunction& fn) { ids.push_back(id); },
        [](const RecordFunction&) {}));
  };

  auto h1 = add_remove_test_add_cb(1);
  auto h2 = add_remove_test_add_cb(2);
  auto h3 = add_remove_test_add_cb(3);

  { RECORD_USER_SCOPE("test"); }

  TORCH_CHECK(ids.size() == 3);
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 1) != ids.end());
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 3) != ids.end());

  ids.clear();
  removeCallback(h1);

  { RECORD_USER_SCOPE("test"); }

  TORCH_CHECK(ids.size() == 2);
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 3) != ids.end());

  ids.clear();
  removeCallback(h3);

  { RECORD_USER_SCOPE("test"); }

  TORCH_CHECK(ids.size() == 1);
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());

  clearCallbacks();

  // thread local / global callbacks

  ids.clear();
  addGlobalCallback(RecordFunctionCallback(
      [&ids](const RecordFunction& fn) { ids.push_back(1); },
      [](const RecordFunction&) {}));

  { RECORD_USER_SCOPE("test"); }

  TORCH_CHECK(ids.size() == 1);
  TORCH_CHECK(ids[0] == 1);
  ids.clear();

  auto th = std::thread([&ids]() {
    c10::impl::IncludeDispatchKeyGuard observer_guard(
        c10::DispatchKey::Profiler);
    addThreadLocalCallback(RecordFunctionCallback(
        [&ids](const RecordFunction& fn) { ids.push_back(2); },
        [](const RecordFunction&) {}));

    { RECORD_USER_SCOPE("test_thread"); }
  });
  th.join();
  TORCH_CHECK(ids.size() == 2);
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 1) != ids.end());
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());
  ids.clear();

  { RECORD_USER_SCOPE("test"); }

  TORCH_CHECK(ids.size() == 1);
  TORCH_CHECK(ids[0] == 1);
  ids.clear();

  clearCallbacks();

  // test should_run

  bool ran = false;
  bool should_run = false;
  addGlobalCallback(
      RecordFunctionCallback(
          [&ran](const RecordFunction& fn) { ran = true; },
          [](const RecordFunction&) {})
          .setShouldRun([&should_run](const RecordFunctionCallback&) {
            return should_run;
          }));

  { RECORD_USER_SCOPE("test"); }

  TORCH_CHECK(!ran);

  should_run = true;

  { RECORD_USER_SCOPE("test"); }

  TORCH_CHECK(ran);

  clearCallbacks();

  // test propagation of TLS callbacks
  std::thread t([]() {
    RecordFunctionGuard enable_rec_fn;
    std::string recorded_op;
    auto handle = addThreadLocalCallback(RecordFunctionCallback(
        [&recorded_op](const RecordFunction& fn) {
          recorded_op = fn.name().str();
        },
        [](const RecordFunction&) {}));
    ThreadLocalState state;
    std::thread t_child([state]() {
      ThreadLocalStateGuard g_tls(state);
      RECORD_USER_SCOPE("test_in_thread");
    });
    t_child.join();
    TORCH_CHECK(recorded_op == "test_in_thread");
    removeCallback(handle);
  });
  t.join();
}

class TestThreadLocalDebugInfo : public c10::DebugInfoBase {
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

void checkDebugInfo(c10::DebugInfoKind kind, int model_id) {
  auto debug_info = c10::ThreadLocalDebugInfo::get(kind);
  TORCH_CHECK(debug_info != nullptr);
  auto* test_debug_info =
      dynamic_cast<TestThreadLocalDebugInfo*>(debug_info.get());
  TORCH_CHECK(test_debug_info != nullptr);
  TORCH_CHECK(test_debug_info->getModelId() == model_id);
}

void testThreadLocalDebugInfo() {
  // enable observers
  c10::impl::IncludeDispatchKeyGuard observer_guard(c10::DispatchKey::Profiler);

  TORCH_CHECK(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::TEST_INFO) == nullptr);
  auto debug_info = std::make_shared<TestThreadLocalDebugInfo>();
  debug_info->setModelId(42);
  {
    c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO, debug_info);
    checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
  }

  // check that thread local debug info is propagated through fork calls
  TORCH_CHECK(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::TEST_INFO) == nullptr);
  std::atomic<bool> done{false};
  {
    c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO, debug_info);
    at::launch([&done]() {
      checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
      done = true;
    });
  }
  while (!done) {
  }

  // check that thread local debug info is propagated through backward pass
  TORCH_CHECK(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::TEST_INFO) == nullptr);
  done = false;
  auto handle = addGlobalCallback(RecordFunctionCallback(
      [&done](const RecordFunction&) {
        checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
        done = true;
        return true;
      },
      [](const RecordFunction&) {}));
  {
    c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO, debug_info);
    auto t = torch::randn({1, 2, 3}, at::kCPU);
    t.set_requires_grad(true);
    auto t2 = t.pow(2);
    t2.backward(torch::ones_like(t2, at::MemoryFormat::Preserve));
  }
  removeCallback(handle);
  TORCH_CHECK(done);

  // check nested debug info
  TORCH_CHECK(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::TEST_INFO) == nullptr);
  {
    c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO, debug_info);
    {
      checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
      {
        auto debug_info = std::make_shared<TestThreadLocalDebugInfo>();
        debug_info->setModelId(314);
        c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO_2, debug_info);
        {
          checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
          checkDebugInfo(c10::DebugInfoKind::TEST_INFO_2, 314);
          done = false;
          at::launch([&done]() {
            checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
            checkDebugInfo(c10::DebugInfoKind::TEST_INFO_2, 314);
            done = true;
          });
          while (!done) {
          }
        }
      }
    }
  }
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
    RecordProfile guard(ss);
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
          [](Stack& stack) {
            push(stack, IValue());
            return 0;
          },
          aliasAnalysisFromSchema()),
      Operator(
          "prim::is_none(int? a) -> bool",
          [](Stack& stack) {
            IValue a = pop(stack);
            if (a.isNone()) {
              push(stack, true);
            } else {
              push(stack, false);
            }
            return 0;
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
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    def add_it(self, x, b : int = 4):
      return self.foo + x + b
  )");
  auto result = m.run_method("add_it", torch::ones({}));
  AT_ASSERT(result.toTensor().item<float>() == 6);
}

void testModuleConversion() {
  Module m("test");
  {
    // test cuda to cpu for params and buffers
    m.register_parameter("foo", torch::ones({}, at::kCUDA), false);
    m.register_buffer("bar", torch::ones({}, at::kCUDA));

    m.to(at::kCUDA);
    m.to(at::kCPU);
    AT_ASSERT(m.attr("foo").toTensor().device().is_cpu());
    AT_ASSERT(m.attr("bar").toTensor().device().is_cpu());
  }
  {
    // test cpu to cuda for params and buffers
    m.register_parameter("foo", torch::ones({}), false);
    m.register_buffer("bar", torch::ones({}));

    m.to(at::kCUDA);
    AT_ASSERT(m.attr("foo").toTensor().device().is_cuda());
    AT_ASSERT(m.attr("bar").toTensor().device().is_cuda());
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
  parseIR(
      R"IR(
graph(%a):
  return (%a))IR",
      &*graph);

  std::vector<IValue> stack = {IValue(torch::randn({22}, at::kCPU))};
  auto run = [&](std::shared_ptr<Graph>& graph, std::vector<IValue> stack) {
    GraphExecutor executor(graph, "");
    executor.run(stack);
    return stack;
  };
  run(graph, stack);
  // we will not run fusion in simple mode
  if (!getExecutorMode()) {
    AT_ASSERT(testPassValue);
  }
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
  auto stack = createStack({x, y});
  // introduce some profiling information
  Code cd(pr->profiled_graph_, "");
  InterpreterState is{cd};
  is.run(stack);
  auto copy = pr->profiled_graph_->copy();
  InsertGuards(copy);
  auto nodes = copy->block()->nodes();
  auto guard = std::find_if(nodes.begin(), nodes.end(), [](Node* n) {
    return n->kind() == prim::Guard;
  });
  ASSERT_NE(guard, nodes.end());
  ASSERT_EQ(
      guard->input()->type()->expect<TensorType>()->sizes().size(),
      c10::nullopt);
  checkShape(*guard, {2, 3}, false);
  auto is_guard = [](Node* n) { return n->kind() == prim::Guard; };
  int num_guards = std::count_if(nodes.begin(), nodes.end(), is_guard);
  ASSERT_EQ(num_guards, 12);
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
  auto stack = createStack({x, y});
  // introduce some profiling information
  Code cd(pr->profiled_graph_, "");
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

  auto input = at::randn({batch_size, input_size}, at::kCPU);
  auto hx = at::randn({batch_size, hidden_size}, at::kCPU);
  auto cx = at::randn({batch_size, hidden_size}, at::kCPU);
  auto w_ih = t_def(at::randn({4 * hidden_size, input_size}, at::kCPU));
  auto w_hh = t_def(at::randn({4 * hidden_size, hidden_size}, at::kCPU));

  auto g = build_lstm();
  auto stack = createStack({input, hx, cx, w_ih, w_hh});

  auto& opt_graph = *g.get();
  ArgumentSpecCreator arg_spec_creator(opt_graph);
  ArgumentSpec spec =
      arg_spec_creator.create(autograd::GradMode::is_enabled(), stack);
  arg_spec_creator.specializeTypes(opt_graph, spec);
  auto pr = ProfilingRecord::instrumentGraph(g);
  Code cd(pr->profiled_graph_, "");
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

void testCallStack() {
  const auto text = R"(
def ham(x):
    return x/7

def bar(x):
    return x*3

def baz(x):
    return ham(x)*x

def foo(x):
    return bar(x)*baz(x)*11
  )";
  auto cu = compile(text);
  const Function& foo = cu->get_function("foo");
  for (Node* n : foo.optimized_graph()->nodes()) {
    if (n->kind() == prim::Constant) {
      if (!n->hasAttribute(attr::value) ||
          n->kindOf(attr::value) != AttributeKind::i) {
        continue;
      }
      int v = n->i(attr::value);
      switch (v) {
        case 3: {
          // Const 3 comes from function 'bar', which gets inlined to 'foo'.
          // The callstack for the corresponding node should contain only the
          // function 'bar'.
          ASSERT_TRUE(n->callstack());
          auto callstack_vector = (*n->callstack())->vec();
          ASSERT_EQ(callstack_vector.size(), 1);
          ASSERT_EQ(callstack_vector[0].first, &cu->get_function("bar"));
          break;
        }
        case 7: {
          // Const 7 comes from function 'ham', which gets inlined to 'baz',
          // which is then inlined to 'foo'. The callstack for the corresponding
          // node should contain these two functions.
          ASSERT_TRUE(n->callstack());
          auto callstack_vector = (*n->callstack())->vec();
          ASSERT_EQ(callstack_vector.size(), 2);
          ASSERT_EQ(callstack_vector[0].first, &cu->get_function("baz"));
          ASSERT_EQ(callstack_vector[1].first, &cu->get_function("ham"));
          break;
        }
        case 11: {
          // Const 11 comes from function 'foo', which is not inlined anywhere
          // and thus it should not have a callstack.
          ASSERT_FALSE(n->callstack());
          break;
        }
      }
    }
  }

  // Check that inlining doesn't corrupt callstack of the callee's nodes.
  const Function& baz = cu->get_function("baz");
  for (Node* n : baz.optimized_graph()->nodes()) {
    if (n->kind() == prim::Constant) {
      if (!n->hasAttribute(attr::value) ||
          n->kindOf(attr::value) != AttributeKind::i) {
        continue;
      }
      int v = n->i(attr::value);
      ASSERT_TRUE(v == 7);
      // Const 7 comes from function 'ham', which gets inlined to 'baz'. 'baz'
      // was also inlined into 'foo', but when looking at the graph of 'baz' we
      // should only see a callstack of depth 1 (containing only 'ham').
      ASSERT_TRUE(n->callstack());
      auto callstack_vector = (*n->callstack())->vec();
      ASSERT_EQ(callstack_vector.size(), 1);
      ASSERT_EQ(callstack_vector[0].first, &cu->get_function("ham"));
    }
  }
}

void testCallStackCaching() {
  const auto text = R"(

def a(x):
    print("a1")
    print("a2")
    return x

def b(x):
    print("b1")
    print("b2")
    a(x)
    return x

def c(x):
    print("c1")
    print("c2")
    b(x)
    return x
  )";
  auto cu = compile(text);
  const Function& baz = cu->get_function("c");
  std::unordered_map<std::string, InlinedCallStack*> callstack_objects;
  for (Node* n : baz.optimized_graph()->nodes()) {
    if (n->kind() == prim::Constant) {
      if (!n->hasAttribute(attr::value) ||
          n->kindOf(attr::value) != AttributeKind::s) {
        continue;
      }
      std::string v = n->s(attr::value);
      if (n->callstack()) {
        callstack_objects[v] = n->callstack()->get();
      }
    }
  }
  // We expect to see nodes prim::Constant[value="a1"] and
  // prim::Constant[value="a2"] inlined to function 'c'. Their callstacks are
  // the same (a->b->c), so we want to make sure we're not creating different
  // callstack entries for them.
  ASSERT_TRUE(callstack_objects.count("a1") && callstack_objects.count("a2"));
  ASSERT_TRUE(callstack_objects.at("a1") == callstack_objects.at("a2"));
}

void testAutogradSymbols() {
  Symbol sym = Symbol::fromQualString("aten::test_symbol");
  Graph graph;
  auto node = graph.create(sym);
  TORCH_CHECK(canRunWithAutograd(node));

  sym = Symbol::fromQualString("prim::test_symbol");
  node = graph.create(sym);
  TORCH_CHECK(canRunWithAutograd(node));

  sym = Symbol::fromQualString("prim::FusionGroup");
  node = graph.create(sym);
  TORCH_CHECK(!canRunWithAutograd(node));

  sym = Symbol::fromQualString("custom::test_symbol");
  node = graph.create(sym);
  TORCH_CHECK(!canRunWithAutograd(node));
}

} // namespace jit
} // namespace torch
