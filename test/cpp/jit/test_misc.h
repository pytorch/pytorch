#pragma once

#include "test/cpp/jit/test_base.h"
#include "test/cpp/jit/test_utils.h"

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
#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/jit/script/module.h"
#include "torch/jit.h"

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
  auto ksz_val = graph->insertConstant(IValue(kernel_size));
  auto kst_val = graph->insertConstant(IValue(stride));
  auto pad_val = graph->insertConstant(IValue(padding));

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
  std::shared_ptr<torch::jit::script::Module> module =
      torch::jit::load(module_path);
  AT_ASSERT(module->get_module("dropout")->is_training());
  module->eval();
  AT_ASSERT(!module->get_module("dropout")->is_training());
  module->train();
  AT_ASSERT(module->get_module("dropout")->is_training());
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

void invokeTestRecordFunction(at::Tensor& t) {
  autograd::profiler::GetPackedInputsCallback inputs_cb =
    [t]() {
      Stack st;
      pack(st, t);
      return st;
    };
  autograd::profiler::RecordFunction guard("test", inputs_cb);
  t.add_(torch::ones_like(t));
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

void invokeTestRecordFunctionNested() {
  autograd::profiler::RecordFunction guard("inner");
}

void testRecordFunction() {
  std::vector<std::vector<int64_t>> input_sizes;
  autograd::profiler::pushCallback([&input_sizes](
      const autograd::profiler::RecordFunction& fn) {
    for (const auto& input : fn.inputs()) {
      if (input.isTensor()) {
        std::vector<int64_t> t = input.toTensor().sizes().vec();
        input_sizes.push_back(t);
      }
    }
  });

  auto t = torch::randn({1, 2, 3}, at::kCPU);
  invokeTestRecordFunction(t);

  autograd::profiler::popCallback();

  AT_CHECK(input_sizes.size() == 1);
  AT_CHECK(input_sizes[0] == at::IntArrayRef({1, 2, 3}));

  // test nested RecordFunctions
  std::vector<std::string> nested_names;
  autograd::profiler::pushCallback([&nested_names](
      const autograd::profiler::RecordFunction& fn) {
    nested_names.push_back(getFullName(&fn));
  });

  {
    autograd::profiler::RecordFunction guard("outer");
    invokeTestRecordFunctionNested();;
  }

  autograd::profiler::popCallback();
  AT_CHECK(nested_names.size() == 2);
  AT_CHECK(nested_names[0] == "outer");
  AT_CHECK(nested_names[1] == "outer::inner");
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
  AT_CHECK(count == 200);
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
          }),
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
          }),
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
} // namespace test
} // namespace jit
} // namespace torch
