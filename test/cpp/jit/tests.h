#pragma once

#if defined(USE_GTEST)
#include <gtest/gtest.h>
#include <test/cpp/common/support.h>
#else
#include "c10/util/Exception.h"
#define ASSERT_EQ(x, y) AT_ASSERT((x) == (y))
#define ASSERT_NE(x, y) AT_ASSERT((x) != (y))
#define ASSERT_TRUE AT_ASSERT
#define ASSERT_FALSE(x) ASSERT_TRUE(!(x))
#define ASSERT_THROWS_WITH(statement, substring)                         \
  try {                                                                  \
    (void)statement;                                                     \
    ASSERT_TRUE(false);                                                  \
  } catch (const std::exception& e) {                                    \
    ASSERT_NE(std::string(e.what()).find(substring), std::string::npos); \
  }
#define ASSERT_ANY_THROW(statement)                                      \
  bool threw = false;                                                    \
  try {                                                                  \
    (void)statement;                                                     \
  } catch (const std::exception& e) {                                    \
    threw = true;                                                        \
  }                                                                      \
  ASSERT_TRUE(threw);                                                    \

#endif // defined(USE_GTEST)

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/argument_spec.h"
#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/attributes.h"
#include "torch/csrc/jit/autodiff.h"
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/custom_operator.h"
#include "torch/csrc/jit/dynamic_dag.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/lower_grad_of.h"
#include "torch/csrc/jit/passes/requires_grad_analysis.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/symbolic_variable.h"
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/utils/hash.h"
#include "torch/csrc/variable_tensor_functions.h"
#include "torch/csrc/autograd/generated/variable_factories.h"

#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/variable.h"

#include "torch/csrc/jit/graph_executor.h"
#include "torch/csrc/jit/ivalue.h"
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

Value* appendNewNode(NodeKind kind, Graph& graph, ArrayRef<Value*> inputs) {
  return graph.appendNode(graph.create(kind, inputs))->output();
}

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

  auto testOne = [&](int ti, int tj, int toi, int toj) {
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
  testOne(0, 0, 0, 0);
  testOne(0, 1, 0, 0);
  testOne(1, 2, 0, 0);
  testOne(0, 2, 0, 0);

  testOne(0, 0, 0, 1);
  testOne(0, 1, 1, 2);
  testOne(1, 2, 0, 2);

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

struct Attr : public Attributes<Attr> {};
void testAttributes() {
  auto one = attr::alpha;
  auto two = attr::device;
  auto three = attr::end;
  auto four = attr::perm;
  Attr attr;
  attr.f_(one, 3.4)->i_(two, 5)->s_(three, "what");
  ASSERT_EQ(attr.f(one), 3.4);
  ASSERT_EQ(attr.s(three), "what");
  ASSERT_EQ(attr.i(two), 5);
  attr.s_(one, "no");
  ASSERT_EQ(attr.s(one), "no");
  ASSERT_TRUE(attr.hasAttribute(three));
  ASSERT_TRUE(!attr.hasAttribute(four));
  attr.ss_(two, {"hi", "now"});
  ASSERT_EQ(attr.ss(two).at(1), "now");

  Attr attr2;
  attr2.copyAttributes(attr);
  ASSERT_EQ(attr2.s(one), "no");
  attr2.f_(one, 5);
  ASSERT_EQ(attr.s(one), "no");
  ASSERT_EQ(attr2.f(one), 5);
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

at::Tensor t_use(at::Tensor x) {
  return x;
}
at::Tensor t_def(at::Tensor x) {
  return x.t();
}

// given the difference of output vs expected tensor, check whether the
// difference is within a relative tolerance range. This is a standard way of
// matching tensor values upto certain precision
bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor> inputs) {
  double maxValue = 0.0;
  for (auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }
  return diff.abs().max().item<float>() < 2e-6 * maxValue;
}
bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
  return checkRtol(a - b, {a, b});
}

bool exactlyEqual(const at::Tensor& a, const at::Tensor& b) {
  return (a - b).abs().max().item<float>() == 0.f;
}

std::pair<at::Tensor, at::Tensor> lstm(
    at::Tensor input,
    at::Tensor hx,
    at::Tensor cx,
    at::Tensor w_ih,
    at::Tensor w_hh) {
  auto gates = input.mm(t_use(w_ih)) + hx.mm(t_use(w_hh));

  auto chunked_gates = gates.chunk(4, 1);
  auto ingate = chunked_gates[0];
  auto forgetgate = chunked_gates[1];
  auto cellgate = chunked_gates[2];
  auto outgate = chunked_gates[3];

  ingate = ingate.sigmoid();
  outgate = outgate.sigmoid();
  cellgate = cellgate.tanh();
  forgetgate = forgetgate.sigmoid();

  auto cy = (forgetgate * cx) + (ingate * cellgate);
  auto hy = outgate * cy.tanh();

  return {hy, cy};
}

std::tuple<Var, Var> build_lstm_body(
    Graph& g,
    Var input,
    Var hx,
    Var cx,
    Var w_ih,
    Var w_hh) {
  auto gates = input.mm(w_ih);
  gates = gates + hx.mm(w_hh);
  auto outputs = gates.chunk(4, 1);
  auto ingate = outputs[0];
  auto forgetgate = outputs[1];
  auto cellgate = outputs[2];
  auto outgate = outputs[3];
  ingate = ingate.sigmoid();
  outgate = outgate.sigmoid();
  cellgate = cellgate.tanh();
  forgetgate = forgetgate.sigmoid();

  auto cy = forgetgate * cx;
  cy = cy + ingate * cellgate;
  auto hy = outgate * cy.tanh();

  return std::make_tuple(hy, cy);
}

std::shared_ptr<Graph> build_lstm() {
  auto r = std::make_shared<Graph>();
  auto& g = *r;
  Value* input = g.addInput();
  Value* hx = g.addInput();
  Value* cx = g.addInput();
  Value* w_ih = g.addInput();
  Value* w_hh = g.addInput();

  Var hy;
  Var cy;
  std::tie(hy, cy) = build_lstm_body(g, input, hx, cx, w_ih, w_hh);

  hy.addAsOutput();
  cy.addAsOutput();
  g.lint();

  return r;
}

void run(InterpreterState & interp, const std::vector<at::Tensor> & inputs, std::vector<at::Tensor> & outputs) {
  std::vector<IValue> stack(inputs.begin(), inputs.end());
  interp.run(stack);
  outputs.clear();
  for (auto& ivalue : stack) {
    outputs.push_back(std::move(ivalue).toTensor());
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
  std::vector<at::Tensor> outputs;
  InterpreterState lstm_interp(lstm_function);
  run(lstm_interp, {input[0], hx, cx, w_ih, w_hh}, outputs);
  std::tie(hx, cx) = lstm(input[0], hx, cx, w_ih, w_hh);

  // std::cout << almostEqual(outputs[0],hx) << "\n";
  ASSERT_TRUE(exactlyEqual(outputs[0], hx));
  ASSERT_TRUE(exactlyEqual(outputs[1], cx));
}

using var_meta_type = std::vector<int64_t>;
using var_meta_list = std::vector<var_meta_type>;
using test_fn_type = std::function<variable_list(const variable_list&)>;

struct ADTestSpec {
  ADTestSpec(const char* name, var_meta_list input_meta, test_fn_type test_fn)
      : name(name), input_meta(input_meta), test_fn(test_fn) {}

  variable_list operator()(const variable_list& inputs) const {
    return test_fn(inputs);
  };

  std::vector<Variable> make_vars() const {
    std::vector<Variable> out;
    for (const auto& m : input_meta) {
      out.push_back(torch::randn(m, at::requires_grad(true)));
    }
    return out;
  }

  const char* name;
  var_meta_list input_meta;
  test_fn_type test_fn;
};

variable_list get_grad_outputs(const variable_list& vars) {
  return fmap(vars, [](const Variable& v) -> Variable {
    return at::randn(v.sizes(), v.options());
  });
}

std::shared_ptr<Graph> trace(
    const ADTestSpec& test,
    const variable_list& vars_in) {
  std::shared_ptr<tracer::TracingState> state;
  Stack trace_stack_in;
  std::tie(state, trace_stack_in) = tracer::enter(fmap<IValue>(vars_in));
  variable_list trace_vars_in = fmap(
      trace_stack_in, [](const IValue& v) { return Variable(v.toTensor()); });
  auto trace_vars_out = test(trace_vars_in);
  tracer::exit(fmap<IValue>(trace_vars_out));
  return state->graph;
}

variable_list grad(
    const variable_list& outputs,
    const variable_list& inputs,
    const variable_list& grad_outputs) {
  const auto get_edge = [](const Variable& v) { return v.gradient_edge(); };
  auto& engine = torch::autograd::Engine::get_default_engine();
  return engine.execute(
      fmap(outputs, get_edge),
      grad_outputs,
      true,
      false,
      fmap(inputs, get_edge));
}

void assertAllClose(const tensor_list& a, const tensor_list& b) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    ASSERT_TRUE(a[i].is_same_size(b[i]));
    ASSERT_TRUE(a[i].allclose(b[i]));
  }
}

std::pair<tensor_list, tensor_list> runGradient(
    Gradient& grad_spec,
    tensor_list& tensors_in,
    tensor_list& tensor_grads_in) {
  tensor_list tensors_out, tensor_grads_out;
  Code f_code{grad_spec.f}, df_code{grad_spec.df};
  InterpreterState f_interpreter{f_code}, df_interpreter{df_code};

  run(f_interpreter, tensors_in, tensors_out);

  tensor_list df_inputs;
  df_inputs.insert(
      df_inputs.end(), tensor_grads_in.begin(), tensor_grads_in.end());
  for (auto offset : grad_spec.df_input_captured_inputs)
    df_inputs.push_back(tensors_in[offset]);
  for (auto offset : grad_spec.df_input_captured_outputs)
    df_inputs.push_back(tensors_out[offset]);
  run(df_interpreter, df_inputs, tensor_grads_out);

  // Outputs of f needs to be sliced
  tensors_out.erase(
      tensors_out.begin() + grad_spec.f_real_outputs, tensors_out.end());
  return std::make_pair(tensors_out, tensor_grads_out);
}

void testADFormulas() {
  const auto unwrap = [](const Variable& v) { return v.data(); };

  using VL = variable_list;
  const var_meta_list binary_pointwise = {{2, 3, 4, 5}, {2, 3, 4, 5}};
  const var_meta_list unary_pointwise = {{2, 3, 4, 5}};
  const var_meta_list unary_pointwise_2d = {{2, 3}};
  const std::vector<ADTestSpec> ad_tests = {
      {"add",
       binary_pointwise,
       [](const VL& v) -> VL { return {v[0] + v[1]}; }},
      {"sub",
       binary_pointwise,
       [](const VL& v) -> VL { return {v[0] - v[1]}; }},
      {"mul",
       binary_pointwise,
       [](const VL& v) -> VL { return {v[0] * v[1]}; }},
      {"sigmoid",
       unary_pointwise,
       [](const VL& v) -> VL { return {v[0].sigmoid()}; }},
      {"tanh",
       unary_pointwise,
       [](const VL& v) -> VL { return {v[0].tanh()}; }},
      {"t", unary_pointwise_2d, [](const VL& v) -> VL { return {v[0].t()}; }},
      {"view",
        unary_pointwise_2d,
        [](const VL& v) -> VL { return {v[0].view({3, 2})}; }},
      {"expand",
        {{2, 1}},
        [](const VL& v) -> VL { return {v[0].expand({2, 3})}; }},
      {"mm",
       {{10, 12}, {12, 15}},
       [](const VL& v) -> VL { return {v[0].mm(v[1])}; }},
      // TODO: enable once we'll be able to capture lists across forward-backward
      //{"chunk",   {{10, 12, 15}}, [](const VL& v) -> VL { return
      // fmap<Variable>(v[0].chunk(4, 1)); }},
      //{"chunk",   {{10, 12, 15}}, [](const VL& v) -> VL { return
      // fmap<Variable>(v[0].chunk(3, 2)); }},
      //{"split",   {{10, 12, 15}}, [](const VL& v) -> VL { return
      // fmap<Variable>(v[0].split(4, 1)); }},
      //{"split",   {{10, 12, 15}}, [](const VL& v) -> VL { return
      // fmap<Variable>(v[0].split(3, 2)); }},
  };

  for (const auto& test : ad_tests) {
    // Get reference values form autograd
    auto vars_in = test.make_vars();
    auto vars_out = test(vars_in);
    auto var_grads_in = get_grad_outputs(vars_out);
    auto var_grads_out = grad(vars_out, vars_in, var_grads_in);

    // Trace and differentiate the op
    auto graph = trace(test, vars_in);
    EliminateDeadCode(graph); // Tracing of some ops depends on the DCE trick
    ConstantPropagation(graph);
    auto grad_spec = differentiate(graph);
    LowerGradOf(*grad_spec.df);
    // Get outputs from the interpreter
    auto tensors_in = fmap(vars_in, unwrap);
    auto tensor_grads_in = fmap(var_grads_in, unwrap);
    tensor_list tensors_out, tensor_grads_out;
    std::tie(tensors_out, tensor_grads_out) =
        runGradient(grad_spec, tensors_in, tensor_grads_in);

    // Compare results
    auto expected_tensors_out = fmap(vars_out, unwrap);
    auto expected_tensor_grads_out = fmap(var_grads_out, unwrap);
    assertAllClose(tensors_out, expected_tensors_out);
    assertAllClose(tensor_grads_out, expected_tensor_grads_out);
  }
}

std::string toString(std::shared_ptr<Graph>& graph) {
  std::ostringstream s;
  s << *graph;
  return s.str();
}

void testDifferentiate(std::ostream& out = std::cout) {
  auto graph = std::make_shared<Graph>();
  at::ScalarType s = at::ScalarType::Float;
  auto type = CompleteTensorType::create(s, -1, {2, 3, 4}, {12, 4, 1});

  // Build up a fake graph
  auto a = SymbolicVariable::asNewInput(*graph, type);
  auto b = SymbolicVariable::asNewInput(*graph, type);
  auto c = a * b * a + b;
  graph->registerOutput(c.value());

  auto grad_spec = differentiate(graph);
  std::vector<size_t> expected_captured_inputs = {0, 1};
  std::vector<size_t> expected_captured_outputs = {1};
  std::vector<size_t> expected_input_vjps = {0, 1};
  std::vector<size_t> expected_output_vjps = {0, 1};
  ASSERT_EQ(grad_spec.f_real_outputs, 1);
  ASSERT_EQ(grad_spec.df_input_captured_inputs, expected_captured_inputs);
  ASSERT_EQ(grad_spec.df_input_captured_outputs, expected_captured_outputs);
  ASSERT_EQ(grad_spec.df_input_vjps, expected_input_vjps);
  ASSERT_EQ(grad_spec.df_output_vjps, expected_output_vjps);
  out << "testDifferentiate\n";
  out << *grad_spec.f;
  out << *grad_spec.df;
  out << "\n";
}

void testDifferentiateWithRequiresGrad(std::ostream& out = std::cout) {
  // Build up a fake graph
  auto graph = std::make_shared<Graph>();
  auto a = SymbolicVariable::asNewInput(*graph);
  auto b = SymbolicVariable::asNewInput(*graph);
  auto d = b * b + b;
  auto e = (d + a) * a + b;
  graph->registerOutput(d.value());
  graph->registerOutput(e.value());

  auto a_var = autograd::make_variable(at::empty_strided(2, 2, at::CPU(at::kFloat).options()), true);
  auto b_var = autograd::make_variable(at::empty_strided(2, 2, at::CPU(at::kFloat).options()), false);
  setInputTypes(*graph, ArgumentSpec(true, {a_var, b_var}, 2));
  PropagateInputShapes(*graph);
  PropagateRequiresGrad(graph);

  auto grad_spec = differentiate(graph);
  std::vector<size_t> expected_input_vjps = {1, 2}; // for e and %4 = (d + a)
  std::vector<size_t> expected_output_vjps = {0}; // only a requires grad
  ASSERT_EQ(grad_spec.f_real_outputs, 2); // we need one temporary %4 = (d + a)
  ASSERT_EQ(grad_spec.df_input_captured_inputs, std::vector<size_t>({0}));
  ASSERT_EQ(grad_spec.df_input_captured_outputs, std::vector<size_t>({2}));
  ASSERT_EQ(grad_spec.df_input_vjps, expected_input_vjps);
  ASSERT_EQ(grad_spec.df_output_vjps, expected_output_vjps);
  out << "testDifferentiateWithRequiresGrad\n";
  out << *grad_spec.f;
  out << *grad_spec.df;
  out << "\n";
}

void testCreateAutodiffSubgraphs(std::ostream& out = std::cout) {
  auto graph = build_lstm();
  CreateAutodiffSubgraphs(*graph, /*threshold=*/2);
  out << "testCreateAutodiffSubgraphs\n";
  out << *graph << "\n";
}

autograd::Variable var(at::Type& t, at::IntList sizes, bool requires_grad) {
  return autograd::make_variable(at::rand(sizes, t.options()), requires_grad);
}
autograd::Variable undef() {
  return autograd::Variable();
}

int device(const autograd::Variable& v) {
  return v.type().is_cuda() ? v.get_device() : -1;
}

bool isEqual(at::IntList lhs, at::IntList rhs) {
  return lhs.size() == rhs.size() &&
      std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool isEqual(const CompleteArgumentInfo& ti, const autograd::Variable& v) {
  if (!ti.defined())
    return ti.defined() == v.defined();
  return ti.device() == device(v) && ti.requires_grad() == v.requires_grad() &&
      ti.type() == v.type().scalarType() && isEqual(ti.sizes(), v.sizes()) &&
      isEqual(ti.strides(), v.strides());
}

// work around the fact that variable_tensor_list doesn't duplicate all
// of std::vector's constructors.
// most constructors are never used in the implementation, just in our tests.
Stack createStack(std::vector<at::Tensor>&& list) {
  return Stack(
      std::make_move_iterator(list.begin()),
      std::make_move_iterator(list.end()));
}

void testArgumentSpec() {
  auto& CF = at::CPU(at::kFloat);
  auto& CD = at::CPU(at::kDouble);
  auto& GF = at::CUDA(at::kFloat);
  auto& GD = at::CUDA(at::kDouble);

  auto list = createStack({var(CF, {1}, true),
                           var(CD, {1, 2}, false),
                           var(GF, {}, true),
                           var(GD, {4, 5, 6}, false),
                           undef()});

  // make sure we have some non-standard strides
  list[1].toTensor().transpose_(0, 1);

  // same list but different backing values
  auto list2 = createStack({var(CF, {1}, true),
                            var(CD, {1, 2}, false),
                            var(GF, {}, true),
                            var(GD, {4, 5, 6}, false),
                            undef()});
  list2[1].toTensor().transpose_(0, 1);

  CompleteArgumentSpec a(true, list);
  CompleteArgumentSpec b(true, list);
  ASSERT_EQ(a.hashCode(), b.hashCode());

  ASSERT_EQ(a, b);
  CompleteArgumentSpec d(true, list2);
  ASSERT_EQ(d, a);
  ASSERT_EQ(d.hashCode(), a.hashCode());

  for (size_t i = 0; i < list.size(); ++i) {
    ASSERT_TRUE(isEqual(a.at(i), list[i].toTensor()));
  }
  CompleteArgumentSpec no_grad(/*with_grad=*/false, list);
  ASSERT_TRUE(no_grad != a);

  std::unordered_set<CompleteArgumentSpec> spec;
  spec.insert(std::move(a));
  ASSERT_TRUE(spec.count(b) > 0);
  ASSERT_EQ(spec.count(no_grad), 0);
  spec.insert(std::move(no_grad));
  ASSERT_EQ(spec.count(CompleteArgumentSpec(true, list)), 1);

  list2[1].toTensor().transpose_(0, 1);
  CompleteArgumentSpec c(true, list2); // same as list, except for one stride
  ASSERT_FALSE(c == a);
  ASSERT_EQ(spec.count(c), 0);

  Stack stack = {var(CF, {1, 2}, true), 3, var(CF, {1, 2}, true)};
  CompleteArgumentSpec with_const(true, stack);
  ASSERT_EQ(with_const.at(2).sizes().size(), 2);
}

void testGraphExecutor() {
  constexpr int batch_size = 4;
  constexpr int input_size = 256;

  int hidden_size = 2 * input_size;

  auto v = [](at::Tensor t) { return autograd::make_variable(t, false); };

  auto input = at::randn({batch_size, input_size}, at::kCUDA);
  auto hx = at::randn({batch_size, hidden_size}, at::kCUDA);
  auto cx = at::randn({batch_size, hidden_size}, at::kCUDA);
  auto w_ih = t_def(at::randn({4 * hidden_size, input_size}, at::kCUDA));
  auto w_hh = t_def(at::randn({4 * hidden_size, hidden_size}, at::kCUDA));

  auto g = build_lstm();
  GraphExecutor executor(g);
  auto stack = createStack({v(input), v(hx), v(cx), v(w_ih), v(w_hh)});
  executor.run(stack);
  ASSERT_EQ(stack.size(), 2);
  at::Tensor r0, r1;
  std::tie(r0, r1) = lstm(input, hx, cx, w_ih, w_hh);
  ASSERT_TRUE(almostEqual(Variable(stack[0].toTensor()).data(), r0));
  ASSERT_TRUE(almostEqual(Variable(stack[1].toTensor()).data(), r1));
}

void testBlocks(std::ostream& out = std::cout) {
  Graph g;
  auto a = Var::asNewInput(g, "a");
  auto b = Var::asNewInput(g, "b");
  auto c = a + b;
  auto r = g.appendNode(g.create(prim::If, {Var::asNewInput(g, "c").value()}));
  auto then_block = r->addBlock();
  auto else_block = r->addBlock();
  {
    WithInsertPoint guard(then_block);
    auto t = c + c;
    then_block->registerOutput(t.value());
  }
  {
    WithInsertPoint guard(else_block);
    auto d = b + c;
    auto e = d + c;
    else_block->registerOutput(e.value());
  }
  g.registerOutput((Var(r->output()) + c).value());
  g.lint();
  out << "testBlocks\n" << g << "\n";
  r->eraseBlock(0);
  out << g << "\n";
  g.lint();
  // test recursive copy of blocks works
  auto g2 = g.copy();
  out << *g2 << "\n";
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
  script::Module cu;
  script::defineMethodsInModule(
      cu, cf_examples, script::nativeResolver, nullptr);
  auto run = [&](const std::string& name, std::vector<IValue> stack) {
    auto graph = cu.get_method(name).graph();
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

void testIValue() {
  Shared<IntList> foo = IntList::create({3, 4, 5});
  ASSERT_EQ(foo.use_count(), 1);
  IValue bar{foo};
  ASSERT_EQ(foo.use_count(), 2);
  auto baz = bar;
  ASSERT_EQ(foo.use_count(), 3);
  auto foo2 = std::move(bar);
  ASSERT_EQ(foo.use_count(), 3);
  ASSERT_TRUE(foo2.isIntList());
  ASSERT_TRUE(bar.isNone());
  foo2 = IValue(4.0);
  ASSERT_TRUE(foo2.isDouble());
  ASSERT_EQ(foo2.toDouble(), 4.0);
  ASSERT_EQ(foo.use_count(), 2);
  ASSERT_TRUE(ArrayRef<int64_t>(baz.toIntList()->elements()).equals({3, 4, 5}));

  auto move_it = std::move(baz).toIntList();
  ASSERT_EQ(foo.use_count(), 2);
  ASSERT_TRUE(baz.isNone());
  IValue i(4);
  ASSERT_TRUE(i.isInt());
  ASSERT_EQ(i.toInt(), 4);
  IValue dlist(DoubleList::create({3.5}));
  ASSERT_TRUE(dlist.isDoubleList());
  ASSERT_TRUE(ArrayRef<double>(std::move(dlist).toDoubleList()->elements())
                  .equals({3.5}));
  ASSERT_TRUE(dlist.isNone());
  dlist = IValue(DoubleList::create({3.4}));
  ASSERT_TRUE(ArrayRef<double>(dlist.toDoubleList()->elements()).equals({3.4}));
  IValue the_list(Tuple::create({IValue(3.4), IValue(4), IValue(foo)}));
  ASSERT_EQ(foo.use_count(), 3);
  ASSERT_TRUE(the_list.isTuple());
  auto first = std::move(the_list).toTuple()->elements().at(1);
  ASSERT_EQ(first.toInt(), 4);
  at::Tensor tv = at::rand({3, 4});
  IValue ten(tv);
  ASSERT_EQ(tv.use_count(), 2);
  auto ten2 = ten;
  ASSERT_EQ(tv.use_count(), 3);
  ASSERT_TRUE(ten2.toTensor().equal(ten.toTensor()));
  std::move(ten2).toTensor();
  ASSERT_EQ(tv.use_count(), 2);
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
    ASSERT_EQ(op->schema().arguments()[1].type()->kind(), TypeKind::DynamicType);

    ASSERT_EQ(op->schema().returns()[0].type()->kind(), TypeKind::DynamicType);

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
    ASSERT_EQ(op->schema().arguments()[1].type()->kind(), TypeKind::DynamicType);

    ASSERT_EQ(op->schema().returns().size(), 1);
    ASSERT_EQ(op->schema().returns()[0].type()->kind(), TypeKind::DynamicType);

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
        "but the provided schema specified type Dynamic "
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
        "but the provided schema specified type Dynamic "
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

// test a few features that are not directly used in schemas yet
void testSchemaParser() {
  // nested arrays
  auto s = parseSchema("at::what(int[][4] foo) -> ()");
  ASSERT_TRUE(s.arguments().at(0).N() == 4);
  ASSERT_TRUE(IntType::get()->isSubtypeOf(s.arguments().at(0)
                                              .type()->expect<ListType>()
                                              ->getElementType()
                                              ->expect<ListType>()
                                              ->getElementType()));
  auto s2 = parseSchema("at::what(int[][] foo) -> ()");
  ASSERT_TRUE(IntType::get()->isSubtypeOf(s2.arguments().at(0)
                                            .type()->expect<ListType>()
                                            ->getElementType()
                                            ->expect<ListType>()
                                            ->getElementType()));

  // named returns
  parseSchema("at::what(Tensor! i_will_be_written_to) -> ()");
  auto s3 = parseSchema("at::what() -> (Tensor the_return, Tensor the_return2)");
  ASSERT_TRUE(s3.returns().at(0).name() == "the_return");
  ASSERT_TRUE(s3.returns().at(1).name() == "the_return2");

  // futures
  auto s4 = parseSchema("at::what(Future(int) foo) -> ()");
  ASSERT_TRUE(IntType::get()->isSubtypeOf(s4.arguments().at(0)
                                          .type()->expect<FutureType>()
                                          ->getElementType()));

  // test tensor with annotated alias sets
  parseSchema("at::what(Tensor(t) foo) -> (Tensor(t))");

}

void testTopologicalIndex() {
  {
    Graph graph;
    auto node1 = graph.create(prim::Undefined);
    auto node2 = graph.create(prim::Undefined);
    auto node3 = graph.create(prim::Undefined);
    auto node4 = graph.create(prim::Undefined);

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
    auto A = graph.create(prim::Undefined);
    block1->appendNode(A);
    auto B = graph.create(prim::Undefined);
    block1->appendNode(B);
    auto block2 = B->addBlock();
    auto C = graph.create(prim::Undefined);
    block2->appendNode(C);

    // Check isAfter on different block levels
    ASSERT_TRUE(node1->isBefore(A));
    ASSERT_TRUE(A->isBefore(B));
    ASSERT_TRUE(A->isBefore(C));

    // make sure things don't blow up on deletions
    node2->destroy();
    auto node2p = graph.create(prim::Undefined);
    node2p->insertAfter(node1);
    ASSERT_TRUE(node1->isBefore(node2p));
    ASSERT_TRUE(node2p->isBefore(node3));
  }
  {
    // Induce reindexing to test that path
    Graph graph;
    std::map<size_t, Node*> nodes;

    auto anchor = graph.create(prim::Undefined);
    graph.appendNode(anchor);
    // Inserting to the same place a lot will trigger reindexing
    for (auto i = 0; i < 100; ++i) {
      auto n = graph.create(prim::Undefined);
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
  return std::unique_ptr<detail::DynamicDAG<std::string>>(new detail::DynamicDAG<std::string>());
}

void testNewVertex() {
  auto graph = newDynamicDAG();
  JIT_ASSERT(graph->debugNumVertices() == 0);
  auto a = graph->newVertex("a");
  JIT_ASSERT(graph->debugNumVertices() == 1);
  JIT_ASSERT(a->ord == 0);
  JIT_ASSERT(a->data.size() == 1);
  JIT_ASSERT(a->data[0] == "a");
  JIT_ASSERT(a->in_edges().size() == 0);
  JIT_ASSERT(a->out_edges().size() == 0);
  auto b = graph->newVertex("b");
  auto c = graph->newVertex("c");
  JIT_ASSERT(graph->debugNumVertices() == 3);
  JIT_ASSERT(b->ord == 1);
  JIT_ASSERT(c->ord == 2);
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
  JIT_ASSERT(a->in_edges().size() == 0);
  JIT_ASSERT(a->out_edges().size() == 2);
  JIT_ASSERT(a->out_edges().contains(b));
  JIT_ASSERT(a->out_edges().contains(c));
  JIT_ASSERT(b->in_edges().size() == 1);
  JIT_ASSERT(b->out_edges().size() == 1);
  JIT_ASSERT(b->in_edges().contains(a));
  JIT_ASSERT(b->out_edges().contains(c));
  JIT_ASSERT(c->in_edges().size() == 2);
  JIT_ASSERT(c->out_edges().size() == 0);
  JIT_ASSERT(c->in_edges().contains(a));
  JIT_ASSERT(c->in_edges().contains(b));
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
  JIT_ASSERT(erred);
}

void testAddEdgeReordersBasic() {
  // a, b => b -> a
  auto graph = newDynamicDAG();
  auto a = graph->newVertex("a");
  auto b = graph->newVertex("b");
  JIT_ASSERT(a->ord == 0);
  JIT_ASSERT(b->ord == 1);
  graph->addEdge(b, a);
  JIT_ASSERT(a->ord == 1);
  JIT_ASSERT(b->ord == 0);
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
  JIT_ASSERT(a->ord == 0);
  JIT_ASSERT(b->ord == 1);
  JIT_ASSERT(c->ord == 2);
  JIT_ASSERT(d->ord == 3);
  graph->addEdge(d, a);
  JIT_ASSERT(c->ord == 0);
  JIT_ASSERT(d->ord == 1);
  JIT_ASSERT(a->ord == 2);
  JIT_ASSERT(b->ord == 3);
  JIT_ASSERT(c->in_edges().size() == 0);
  JIT_ASSERT(c->out_edges().size() == 1);
  JIT_ASSERT(c->out_edges().contains(d));
  JIT_ASSERT(d->in_edges().size() == 1);
  JIT_ASSERT(d->out_edges().size() == 1);
  JIT_ASSERT(d->in_edges().contains(c));
  JIT_ASSERT(d->out_edges().contains(a));
  JIT_ASSERT(a->in_edges().size() == 1);
  JIT_ASSERT(a->out_edges().size() == 1);
  JIT_ASSERT(a->in_edges().contains(d));
  JIT_ASSERT(a->out_edges().contains(b));
  JIT_ASSERT(b->in_edges().size() == 1);
  JIT_ASSERT(b->out_edges().size() == 0);
  JIT_ASSERT(b->in_edges().contains(a));
}

void testRemoveEdgeBasic() {
  // a -> b
  auto graph = newDynamicDAG();
  auto a = graph->newVertex("a");
  auto b = graph->newVertex("b");
  graph->addEdge(a, b);
  JIT_ASSERT(graph->debugNumVertices() == 2);
  graph->removeEdge(a, b);
  JIT_ASSERT(graph->debugNumVertices() == 2);
  JIT_ASSERT(a->out_edges().size() == 0);
  JIT_ASSERT(b->in_edges().size() == 0);
}

void testRemoveVertexBasic() {
  // a -> b
  auto graph = newDynamicDAG();
  auto a = graph->newVertex("a");
  auto b = graph->newVertex("b");
  auto c = graph->newVertex("c");
  graph->addEdge(a, b);
  graph->addEdge(b, c);
  JIT_ASSERT(graph->debugNumVertices() == 3);
  graph->removeVertex(b);
  JIT_ASSERT(graph->debugNumVertices() == 2);
  JIT_ASSERT(a->out_edges().size() == 0);
  JIT_ASSERT(c->in_edges().size() == 0);
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
  JIT_ASSERT(graph->debugNumVertices() == 3);
  JIT_ASSERT(a->out_edges().size() == 1);
  JIT_ASSERT(d->in_edges().size() == 1);
  JIT_ASSERT(*a->out_edges().begin() == *d->in_edges().begin());
  auto* contracted = *a->out_edges().begin();
  JIT_ASSERT(contracted->data.size() == 2);
  JIT_ASSERT(contracted->data[0] == "b");
  JIT_ASSERT(contracted->data[1] == "c");
  JIT_ASSERT(contracted->out_edges().size() == 1);
  JIT_ASSERT(contracted->in_edges().size() == 1);
  JIT_ASSERT(contracted->in_edges().contains(a));
  JIT_ASSERT(contracted->out_edges().contains(d));
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
  JIT_ASSERT(!graph->contractEdge(a, c));
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

// Fixture to set up a graph and make assertions clearer
struct TopoMoveTestFixture {
  TopoMoveTestFixture() {
    createGraph();
  }

  // Nodes are named after their output.
  // e.g. "a" is an alias for "the node that outputs the value `a`"
  void createGraph() {
    createNode("a", {});
    createNode("b", {"a"});
    createNode("c", {});
    createNode("d", {"a", "b"});
    createNode("e", {"c", "b"});
    createNode("f", {"e"});
    createNode("g", {"e"});
    createNode("h", {"g"});
    createNode("i", {"g"});
    createNode("j", {"i"});
    createNode("k", {"i"});
    createNode("l", {"a"});
    createNode("m", {}, {"l"}); // block depends on l
    createNode("n", {"m"});
    createNode("o", {"n"});
    createNode("p", {});
    createNode("q", {});
    createNode("r", {"q"});
    createNode("s", {"q"});

    graph.lint();
  }

  void createNode(
      const std::string& name,
      const std::vector<std::string>& inputNames,
      const std::vector<std::string>& blockInputNames = {}) {
    std::vector<Value*> inputs;
    for (const auto name : inputNames) {
      inputs.push_back(nodes.at(name)->output());
    }
    auto node = graph.appendNode(graph.create(prim::Undefined, inputs));
    node->output()->setUniqueName(name);
    nodes[name] = node;

    if (blockInputNames.size() != 0) {
      node->addBlock();
      std::vector<Value*> blockDeps;
      for (const auto name : blockInputNames) {
        blockDeps.push_back(nodes.at(name)->output());
      }

      auto block = node->blocks().at(0);
      block->appendNode(graph.create(prim::Undefined, blockDeps));
    }
  }

  bool moveBeforeTopologicallyValid(
      const std::string& toInsert,
      const std::string& insertPoint) {
    std::function<bool(Node*, Node*)> func = [](Node* toInsert,
                                                Node* insertPoint) {
      return toInsert->moveBeforeTopologicallyValid(insertPoint);
    };
    return moveWithChecks(toInsert, insertPoint, func);
  }

  bool moveAfterTopologicallyValid(
      const std::string& toInsert,
      const std::string& insertPoint) {
    std::function<bool(Node*, Node*)> func = [](Node* toInsert,
                                                Node* insertPoint) {
      return toInsert->moveAfterTopologicallyValid(insertPoint);
    };
    return moveWithChecks(toInsert, insertPoint, func);
  }

  bool moveWithChecks(
      const std::string& toInsert,
      const std::string& insertPoint,
      std::function<bool(Node*, Node*)> func) {
    auto n = nodes.at(toInsert);
    auto insert = nodes.at(insertPoint);
    bool isAfter = n->isAfter(insert);

    std::vector<Node*> originalOrdering;
    Node* original = isAfter ? n->next() : n->prev();

    auto curNode = original;
    while (curNode != n->owningBlock()->return_node()) {
      originalOrdering.push_back(curNode);
      if (isAfter) {
        curNode = curNode->next();
      } else {
        curNode = curNode->prev();
      }
    }

    const auto couldMove = func(n, insert);
    // Check the graph is okay
    graph.lint();

    // If this is the picture of nodes
    // <some nodes> ... toInsert ... <some more nodes> ... insertPoint
    // ^----------^ check that these nodes haven't moved
    curNode = original;
    size_t idx = 0;
    while (curNode != n->owningBlock()->return_node()) {
      JIT_ASSERT(originalOrdering[idx] == curNode);
      if (isAfter) {
        curNode = curNode->next();
      } else {
        curNode = curNode->prev();
      }
      idx++;
    }

    return couldMove;
  }

  void checkPostCondition(
      const std::string& toInsert,
      const std::string& insertPoint,
      bool after) {
    if (after) {
      JIT_ASSERT(nodes.at(toInsert)->prev() == nodes.at(insertPoint));
    } else {
      JIT_ASSERT(nodes.at(toInsert)->next() == nodes.at(insertPoint));
    }
  }

  Graph graph;
  std::unordered_map<std::string, Node*> nodes;
};

void testTopologicalMove() {
  {
    // Check that we are removing `this`'s deps properly when we need to split
    // `this` and deps (see code for what the hell that means)
    TopoMoveTestFixture fixture;
    JIT_ASSERT(fixture.moveBeforeTopologicallyValid("q", "s"));
    fixture.checkPostCondition("q", "s", false);
  }
  // Move after
  {
    // Simple move backward
    TopoMoveTestFixture fixture;
    JIT_ASSERT(fixture.moveAfterTopologicallyValid("c", "a"));
    fixture.checkPostCondition("c", "a", true);
  }
  {
    // simple invalid move backward
    TopoMoveTestFixture fixture;
    JIT_ASSERT(!fixture.moveAfterTopologicallyValid("d", "a"));
  }
  {
    // doesn't actually move anything
    TopoMoveTestFixture fixture;
    JIT_ASSERT(fixture.moveAfterTopologicallyValid("f", "e"));
    fixture.checkPostCondition("f", "e", true);
  }
  {
    // move backward with multiple dependencies
    TopoMoveTestFixture fixture;
    JIT_ASSERT(fixture.moveAfterTopologicallyValid("e", "c"));
    fixture.checkPostCondition("e", "c", true);
  }
  {
    // Move backward with non-zero working set
    TopoMoveTestFixture fixture;
    JIT_ASSERT(fixture.moveAfterTopologicallyValid("k", "f"));
    fixture.checkPostCondition("k", "f", true);
  }
  {
    // Simple move forward
    TopoMoveTestFixture fixture;
    JIT_ASSERT(fixture.moveAfterTopologicallyValid("c", "d"));
    fixture.checkPostCondition("c", "d", true);
  }
  {
    // Move forward with non-zero working set
    TopoMoveTestFixture fixture;
    JIT_ASSERT(fixture.moveAfterTopologicallyValid("f", "l"));
    fixture.checkPostCondition("f", "l", true);
  }

  // Move before
  {
    // Simple move forward
    TopoMoveTestFixture fixture;
    JIT_ASSERT(fixture.moveBeforeTopologicallyValid("b", "d"));
    fixture.checkPostCondition("b", "d", false);
  }
  {
    // Simple move backward
    TopoMoveTestFixture fixture;
    JIT_ASSERT(fixture.moveBeforeTopologicallyValid("c", "a"));
    fixture.checkPostCondition("c", "a", false);
  }
  {
    // doesn't actually move anything
    TopoMoveTestFixture fixture;
    JIT_ASSERT(fixture.moveBeforeTopologicallyValid("a", "b"));
    fixture.checkPostCondition("a", "b", false);
  }
  {
    // move forward with deps
    TopoMoveTestFixture fixture;
    JIT_ASSERT(fixture.moveBeforeTopologicallyValid("f", "m"));
    fixture.checkPostCondition("f", "m", false);
  }
  {
    // move backward with deps
    TopoMoveTestFixture fixture;
    JIT_ASSERT(fixture.moveBeforeTopologicallyValid("l", "f"));
    fixture.checkPostCondition("l", "f", false);
  }

  // check that dependencies in blocks are recognized
  {
    TopoMoveTestFixture fixture;
    JIT_ASSERT(!fixture.moveAfterTopologicallyValid("l", "m"));
    JIT_ASSERT(!fixture.moveBeforeTopologicallyValid("m", "l"));
    JIT_ASSERT(!fixture.moveAfterTopologicallyValid("n", "l"));
    JIT_ASSERT(!fixture.moveBeforeTopologicallyValid("l", "n"));
  }

  // Test that moveAfter(n) and moveBefore(n->next()) are not necessarily
  // equivalent. Here, the dependency ordering is n -> o -> p.  So we can't
  // move `n` after `o`, but we can move `n` before `p` (which pushes `o` after
  // `p`)
  {
    TopoMoveTestFixture fixture;
    JIT_ASSERT(!fixture.moveAfterTopologicallyValid("n", "o"));
    JIT_ASSERT(fixture.moveBeforeTopologicallyValid("o", "p"));
    fixture.checkPostCondition("o", "p", false);
  }
}
} // namespace
} // namespace jit
} // namespace torch
