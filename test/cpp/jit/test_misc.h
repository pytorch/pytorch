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
auto ct = CodeTemplate(R"(
int foo($args) {

    $bar
        $bar
    $a+$b
}
int commatest(int a${,stuff})
int notest(int a${,empty,})
)");
auto ct_expect = R"(
int foo(hi, 8) {

    what
    on many
    lines...
    7
        what
        on many
        lines...
        7
    3+4
}
int commatest(int a, things..., others)
int notest(int a)
)";

void testCodeTemplate() {
  {
    TemplateEnv e;
    e.s("hi", "foo");
    e.v("what", {"is", "this"});
    TemplateEnv c(e);
    c.s("hi", "foo2");
    ASSERT_EQ(e.s("hi"), "foo");
    ASSERT_EQ(c.s("hi"), "foo2");
    ASSERT_EQ(e.v("what")[0], "is");
  }

  {
    TemplateEnv e;
    e.v("args", {"hi", "8"});
    e.v("bar", {"what\non many\nlines...", "7"});
    e.s("a", "3");
    e.s("b", "4");
    e.v("stuff", {"things...", "others"});
    e.v("empty", {});
    auto s = ct.format(e);
    // std::cout << "'" << s << "'\n";
    // std::cout << "'" << ct_expect << "'\n";
    ASSERT_EQ(s, ct_expect);
  }
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

void testInterp() {
  constexpr int batch_size = 4;
  constexpr int input_size = 256;
  constexpr int seq_len = 32;

  int hidden_size = 2 * input_size;

  auto input = at::randn({seq_len, batch_size, input_size}, at::kCUDA);
  auto hx = at::randn({batch_size, hidden_size}, at::kCUDA);
  auto cx = at::randn({batch_size, hidden_size}, at::kCUDA);
  auto w_ih = t_def(at::randn({4 * hidden_size, input_size}, at::kCUDA));
  auto w_hh = t_def(at::randn({4 * hidden_size, hidden_size}, at::kCUDA));

  auto lstm_g = build_lstm();
  Code lstm_function(lstm_g);
  InterpreterState lstm_interp(lstm_function);
  auto outputs = run(lstm_interp, {input[0], hx, cx, w_ih, w_hh});
  std::tie(hx, cx) = lstm(input[0], hx, cx, w_ih, w_hh);

  // std::cout << almostEqual(outputs[0],hx) << "\n";
  ASSERT_TRUE(exactlyEqual(outputs[0], hx));
  ASSERT_TRUE(exactlyEqual(outputs[1], cx));
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

using var_meta_type = std::vector<int64_t>;
using var_meta_list = std::vector<var_meta_type>;
using test_fn_type = std::function<variable_list(const variable_list&)>;

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
    c.value()->setUniqueName(cname);
    d.value()->setUniqueName(dname);
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
    AT_CHECK(
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

const auto cf_examples = R"JIT(
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
  auto cu = std::make_shared<script::Module>();
  script::defineMethodsInModule(
      cu, cf_examples, script::nativeResolver, c10::nullopt);
  auto run = [&](const std::string& name, std::vector<IValue> stack) {
    auto graph = cu->get_method(name).graph();
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

std::unique_ptr<detail::DynamicDAG<std::string>> newDynamicDAG() {
  return std::unique_ptr<detail::DynamicDAG<std::string>>(
      new detail::DynamicDAG<std::string>());
}

void testNewVertex() {
  auto graph = newDynamicDAG();
  AT_ASSERT(graph->debugNumVertices() == 0);
  auto a = graph->newVertex("a");
  AT_ASSERT(graph->debugNumVertices() == 1);
  AT_ASSERT(a->ord == 0);
  AT_ASSERT(a->data.size() == 1);
  AT_ASSERT(a->data[0] == "a");
  AT_ASSERT(a->in_edges().size() == 0);
  AT_ASSERT(a->out_edges().size() == 0);
  auto b = graph->newVertex("b");
  auto c = graph->newVertex("c");
  AT_ASSERT(graph->debugNumVertices() == 3);
  AT_ASSERT(b->ord == 1);
  AT_ASSERT(c->ord == 2);
}

void testAddEdgeBasic() {
  // a -> b -> c
  // \---------^
  auto graph = newDynamicDAG();
  auto a = graph->newVertex("a");
  auto b = graph->newVertex("b");
  auto c = graph->newVertex("c");
  graph->addEdge(a, b);
  graph->addEdge(b, c);
  graph->addEdge(a, c);
  AT_ASSERT(a->in_edges().size() == 0);
  AT_ASSERT(a->out_edges().size() == 2);
  AT_ASSERT(a->out_edges().contains(b));
  AT_ASSERT(a->out_edges().contains(c));
  AT_ASSERT(b->in_edges().size() == 1);
  AT_ASSERT(b->out_edges().size() == 1);
  AT_ASSERT(b->in_edges().contains(a));
  AT_ASSERT(b->out_edges().contains(c));
  AT_ASSERT(c->in_edges().size() == 2);
  AT_ASSERT(c->out_edges().size() == 0);
  AT_ASSERT(c->in_edges().contains(a));
  AT_ASSERT(c->in_edges().contains(b));
}

void testAddEdgeCycleDetection() {
  // a -> b -> c
  // ^---------/
  auto graph = newDynamicDAG();
  auto a = graph->newVertex("a");
  auto b = graph->newVertex("b");
  auto c = graph->newVertex("c");
  graph->addEdge(a, b);
  graph->addEdge(b, c);
  bool erred = false;
  try {
    graph->addEdge(c, a);
  } catch (c10::Error& err) {
    erred = true;
  }
  AT_ASSERT(erred);
}

void testAddEdgeReordersBasic() {
  // a, b => b -> a
  auto graph = newDynamicDAG();
  auto a = graph->newVertex("a");
  auto b = graph->newVertex("b");
  AT_ASSERT(a->ord == 0);
  AT_ASSERT(b->ord == 1);
  graph->addEdge(b, a);
  AT_ASSERT(a->ord == 1);
  AT_ASSERT(b->ord == 0);
}

void testAddEdgeReordersComplicated() {
  // a -> b  c -> d with addEdge(d, b) ==>
  // c -> d -> a -> b
  auto graph = newDynamicDAG();
  auto a = graph->newVertex("a");
  auto b = graph->newVertex("b");
  auto c = graph->newVertex("c");
  auto d = graph->newVertex("d");
  graph->addEdge(a, b);
  graph->addEdge(c, d);
  AT_ASSERT(a->ord == 0);
  AT_ASSERT(b->ord == 1);
  AT_ASSERT(c->ord == 2);
  AT_ASSERT(d->ord == 3);
  graph->addEdge(d, a);
  AT_ASSERT(c->ord == 0);
  AT_ASSERT(d->ord == 1);
  AT_ASSERT(a->ord == 2);
  AT_ASSERT(b->ord == 3);
  AT_ASSERT(c->in_edges().size() == 0);
  AT_ASSERT(c->out_edges().size() == 1);
  AT_ASSERT(c->out_edges().contains(d));
  AT_ASSERT(d->in_edges().size() == 1);
  AT_ASSERT(d->out_edges().size() == 1);
  AT_ASSERT(d->in_edges().contains(c));
  AT_ASSERT(d->out_edges().contains(a));
  AT_ASSERT(a->in_edges().size() == 1);
  AT_ASSERT(a->out_edges().size() == 1);
  AT_ASSERT(a->in_edges().contains(d));
  AT_ASSERT(a->out_edges().contains(b));
  AT_ASSERT(b->in_edges().size() == 1);
  AT_ASSERT(b->out_edges().size() == 0);
  AT_ASSERT(b->in_edges().contains(a));
}

void testRemoveEdgeBasic() {
  // a -> b
  auto graph = newDynamicDAG();
  auto a = graph->newVertex("a");
  auto b = graph->newVertex("b");
  graph->addEdge(a, b);
  AT_ASSERT(graph->debugNumVertices() == 2);
  graph->removeEdge(a, b);
  AT_ASSERT(graph->debugNumVertices() == 2);
  AT_ASSERT(a->out_edges().size() == 0);
  AT_ASSERT(b->in_edges().size() == 0);
}

void testRemoveVertexBasic() {
  // a -> b
  auto graph = newDynamicDAG();
  auto a = graph->newVertex("a");
  auto b = graph->newVertex("b");
  auto c = graph->newVertex("c");
  graph->addEdge(a, b);
  graph->addEdge(b, c);
  AT_ASSERT(graph->debugNumVertices() == 3);
  graph->removeVertex(b);
  AT_ASSERT(graph->debugNumVertices() == 2);
  AT_ASSERT(a->out_edges().size() == 0);
  AT_ASSERT(c->in_edges().size() == 0);
}

void testContractEdgeBasic() {
  // a -> b -> c -> d
  auto graph = newDynamicDAG();
  auto a = graph->newVertex("a");
  auto b = graph->newVertex("b");
  auto c = graph->newVertex("c");
  auto d = graph->newVertex("d");
  graph->addEdge(a, b);
  graph->addEdge(b, c);
  graph->addEdge(c, d);
  graph->contractEdge(b, c);
  AT_ASSERT(graph->debugNumVertices() == 3);
  AT_ASSERT(a->out_edges().size() == 1);
  AT_ASSERT(d->in_edges().size() == 1);
  AT_ASSERT(*a->out_edges().begin() == *d->in_edges().begin());
  auto* contracted = *a->out_edges().begin();
  AT_ASSERT(contracted->data.size() == 2);
  AT_ASSERT(contracted->data[0] == "b");
  AT_ASSERT(contracted->data[1] == "c");
  AT_ASSERT(contracted->out_edges().size() == 1);
  AT_ASSERT(contracted->in_edges().size() == 1);
  AT_ASSERT(contracted->in_edges().contains(a));
  AT_ASSERT(contracted->out_edges().contains(d));
}

void testContractEdgeCycleDetection() {
  // a -> b -> c
  // `---------^
  // contractEdge(a, c) will cause a cycle
  auto graph = newDynamicDAG();
  auto a = graph->newVertex("a");
  auto b = graph->newVertex("b");
  auto c = graph->newVertex("c");
  graph->addEdge(a, b);
  graph->addEdge(b, c);
  graph->addEdge(a, c);
  AT_ASSERT(!graph->contractEdge(a, c));
}

void testDynamicDAG() {
  testNewVertex();
  testAddEdgeBasic();
  testAddEdgeCycleDetection();
  testAddEdgeReordersBasic();
  testAddEdgeReordersComplicated();
  testRemoveEdgeBasic();
  testRemoveVertexBasic();
  testContractEdgeBasic();
  testContractEdgeCycleDetection();
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
          "test::test_none() -> int?",
          [](const Node* node) {
            return [](Stack& stack) {
              push(stack, IValue());
              return 0;
            };
          }),
      Operator(
          "test::is_none(int? a) -> bool",
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
  auto opt_int = g.insert(Symbol::fromQualString("test::test_none"), {});
  auto out_bool = g.insert(Symbol::fromQualString("test::is_none"), {opt_int});
  g.registerOutput(out_bool);
  ConstantPropagation(r);

  auto nodes = r->block()->nodes();
  // checking that constant propagation ran wo/failure
  AT_ASSERT(std::distance(nodes.begin(), nodes.end()) == 1);
}
} // namespace test
} // namespace jit
} // namespace torch
