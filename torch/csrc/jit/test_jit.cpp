#ifdef USE_CATCH

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using Catch::StartsWith;

#else

#define REQUIRE JIT_ASSERT

#endif

#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/fusion_compiler.h"
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/attributes.h"
#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/symbolic_variable.h"
#include "torch/csrc/jit/autodiff.h"
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/hash.h"
#include "torch/csrc/jit/argument_spec.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/lower_grad_of.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/custom_operator.h"
#include "torch/csrc/variable_tensor_functions.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/jit/passes/shape_analysis.h"

#include "torch/csrc/jit/graph_executor.h"
#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/jit/script/module.h"
#include "torch/csrc/jit/ivalue.h"

#include "onnx/onnx_pb.h"


#include <ATen/ATen.h>

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

namespace torch { namespace jit {

using Var = SymbolicVariable;

using namespace torch::autograd;

template<typename T>
static std::ostream & operator<<(std::ostream & out, const std::vector<T> & list) {
  size_t i = 0;
  out << "{";
  for(auto && e : list) {
    if(i++ > 0)
      out << ", ";
    out << e;
  }
  out << "}";
  return out;
}
static auto ct = CodeTemplate(R"(
int foo($args) {

    $bar
        $bar
    $a+$b
}
int commatest(int a${,stuff})
int notest(int a${,empty,})
)");
static auto ct_expect = R"(
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

static void codeTemplateTest() {
  {
    TemplateEnv e;
    e.s("hi","foo");
    e.v("what",{"is","this"});
    TemplateEnv c(e);
    c.s("hi","foo2");
    REQUIRE(e.s("hi") == "foo");
    REQUIRE(c.s("hi") == "foo2");
    REQUIRE(e.v("what")[0] == "is");
  }

  {
    TemplateEnv e;
    e.v("args",{"hi","8"});
    e.v("bar",{"what\non many\nlines...","7"});
    e.s("a","3");
    e.s("b","4");
    e.v("stuff",{"things...","others"});
    e.v("empty",{});
    auto s = ct.format(e);
    //std::cout << "'" << s << "'\n";
    //std::cout << "'" << ct_expect << "'\n";
    REQUIRE(s == ct_expect);
  }
}

Value * appendNewNode(NodeKind kind, Graph& graph, ArrayRef<Value*> inputs) {
  return graph.appendNode(graph.create(kind,inputs))->output();
}


static void fusionTests() {
  FusionCompiler comp;

  auto testSimple = [&] {
    Graph graph;
    Var i0 = Var::asNewInput(graph);
    Var i1 = Var::asNewInput(graph);
    auto o0 = i0 * i1;
    o0.addAsOutput();
    auto a = at::rand({3,4}, at::kCUDA);
    auto b = at::rand({4,3}, at::kCUDA).transpose(0,1);
    auto o = at::zeros({3,4}, at::kCUDA);
    comp.debugLaunchGraph(graph, 0, {a,b}, {o});
    auto o2 = a*b;
    float max_diff = (o2 - o).abs().max().toCDouble();
    //std::cout << "max diff: " << max_diff << "\n";
    REQUIRE(max_diff == 0);
  };
  testSimple();

  auto testOne = [&](int ti, int tj, int toi, int toj) {

    Graph graph;

    Var i0 = Var::asNewInput(graph);
    Var i1 = Var::asNewInput(graph);
    Var i2 = Var::asNewInput(graph);
    Var i3 = Var::asNewInput(graph);
    Var i4 = Var::asNewInput(graph);

    auto p22 =  i4.sigmoid();
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
    std::vector<at::Tensor> outputs;
    // We want to generate input/output tensors with dimension 128x128x32, but
    // with different internal strides.  To do this, we generate a tensor
    // with the "wrong" dimensions, and then use transpose to get an appropriately
    // sized view.
    for(size_t i = 0; i < graph.inputs().size(); i++) {
      std::vector<int64_t> dims = {128, 128, 32};
      std::swap(dims[ti],dims[tj]);
      inputs.push_back(at::rand(dims, at::kCUDA).transpose(ti, tj));
    }
    for(size_t i = 0; i < graph.outputs().size(); i++) {
      std::vector<int64_t> dims = {128, 128, 32};
      std::swap(dims[toi],dims[toj]);
      outputs.push_back(at::zeros(dims, at::kCUDA).transpose(toi,toj));
    }

    auto t22 = inputs[4].sigmoid();
    auto t20 = inputs[3].sigmoid();
    auto t18 = inputs[2].tanh();
    auto t16 = inputs[1].sigmoid();
    auto t14 = t20*inputs[0];
    auto t11 = t22*t18;
    auto out1 = t14+t11;
    auto t5 = out1.tanh();
    auto out0 = t16*t5;


    //auto out0 = inputs[0]*inputs[1];
    comp.debugLaunchGraph(graph, 0, inputs, outputs);
    REQUIRE(out0.is_same_size(outputs.front()));
    float max_diff = (outputs.front() - out0).abs().max().toCDouble();
    REQUIRE(max_diff < 1e-6);

  };
  testOne(0,0,0,0);
  testOne(0,1,0,0);
  testOne(1,2,0,0);
  testOne(0,2,0,0);

  testOne(0,0,0,1);
  testOne(0,1,1,2);
  testOne(1,2,0,2);


  auto createFusedConcat = [](Graph & graph, at::ArrayRef<Value*> inputs, int64_t dim) -> Value* {
    return graph.insertNode(graph.create(prim::FusedConcat, inputs)->i_(attr::dim, dim))->output();
  };

  auto testConcat = [&](int dim) {
    Graph graph;
    Var i0 = Var::asNewInput(graph);
    Var i1 = Var::asNewInput(graph);
    auto o0 = i0 * i1;
    o0.addAsOutput();
    Var(createFusedConcat(graph, {i0, o0}, dim)).addAsOutput();

    auto a = at::rand({3,4,5}, at::kCUDA);
    auto b = at::rand({4,3,5}, at::kCUDA).transpose(0,1);
    auto o = at::zeros({3,4,5}, at::kCUDA);

    auto o_r = a*b;
    auto o2_r = at::cat({a, o_r}, dim);
    auto o2 = at::zeros(o2_r.sizes(), at::kCUDA);
    comp.debugLaunchGraph(graph, 0, {a,b}, {o, o2});

    float max_diff = (o_r - o).abs().max().toCDouble();
    REQUIRE(max_diff == 0);
    float max_diff2 = (o2_r - o2).abs().max().toCDouble();
    REQUIRE(max_diff2 == 0);
  };
  testConcat(0);
  testConcat(1);
  testConcat(2);
}

struct Attr : public Attributes<Attr> {
};
void attributesTest() {
  auto one = attr::alpha;
  auto two = attr::device;
  auto three = attr::end;
  auto four = attr::perm;
  Attr attr;
  attr.f_(one,3.4)->i_(two,5)->s_(three,"what");
  REQUIRE(attr.f(one) == 3.4);
  REQUIRE(attr.s(three) == "what");
  REQUIRE(attr.i(two) == 5);
  attr.s_(one,"no");
  REQUIRE(attr.s(one) == "no");
  REQUIRE(attr.hasAttribute(three));
  REQUIRE(!attr.hasAttribute(four));
  attr.ss_(two, {"hi", "now"});
  REQUIRE(attr.ss(two).at(1) == "now");

  Attr attr2;
  attr2.copyAttributes(attr);
  REQUIRE(attr2.s(one) == "no");
  attr2.f_(one,5);
  REQUIRE(attr.s(one) == "no");
  REQUIRE(attr2.f(one) == 5);
}

void internedStringsTests () {

  REQUIRE(prim::Param == Symbol::prim("Param"));
  REQUIRE(prim::Return == Symbol::prim("Return"));
  REQUIRE(prim::Return.toUnqualString() == std::string("Return"));
  REQUIRE(prim::Return.toQualString() == std::string("prim::Return"));
  Symbol newsym = Symbol::aten("__NEW_SYMBOL");
  size_t symstart = newsym;
  REQUIRE(newsym.toQualString() == std::string("aten::__NEW_SYMBOL"));
  // TODO: This test is a bit too close to the implementation details.
  REQUIRE(Symbol::aten("What") == symstart+1);
  REQUIRE(Symbol::aten("What2") == symstart+2);
  REQUIRE(Symbol::aten("What") == symstart+1);
  REQUIRE(Symbol::aten("What2") == symstart+2);
  REQUIRE(Symbol(symstart+2).toUnqualString() == std::string("What2"));
}

void fromQualStringTests() {
  REQUIRE(Symbol::fromQualString("prim::Param") == Symbol::prim("Param"));
  REQUIRE(Symbol::fromQualString("aten::mm") == Symbol::aten("mm"));
  REQUIRE(Symbol::fromQualString("onnx::LSTM") == Symbol::onnx("LSTM"));
  REQUIRE(Symbol::fromQualString("attr::value") == Symbol::attr("value"));
  REQUIRE(Symbol::fromQualString("scope::") == Symbol::scope(""));
  REQUIRE(Symbol::fromQualString("::").toUnqualString() == std::string(""));
  REQUIRE(Symbol::fromQualString("::").ns().toQualString() == std::string("namespaces::"));
  REQUIRE(Symbol::fromQualString("new_ns::param").toUnqualString() == std::string("param"));
  REQUIRE(Symbol::fromQualString("new_ns::param").ns().toUnqualString() == std::string("new_ns"));
  REQUIRE(Symbol::fromQualString("new_ns::param").ns() == Symbol::fromQualString("namespaces::new_ns"));

  auto bad_inputs = {"scope", ":", ""};
  for (auto input : bad_inputs) {
    try {
      Symbol::fromQualString(input);
      REQUIRE(0);
    } catch (std::runtime_error c) {
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
    maxValue = fmax(tensor.abs().max().toCFloat(), maxValue);
  }
  return diff.abs().max().toCFloat() < 2e-6 * maxValue;
}
bool almostEqual(const at::Tensor & a, const at::Tensor & b) {
  return checkRtol(a - b,{a, b});
}

bool exactlyEqual(const at::Tensor & a, const at::Tensor & b) {
  return (a - b).abs().max().toCFloat() == 0.f;
}

std::pair<at::Tensor, at::Tensor>
lstm(at::Tensor input,
      at::Tensor hx,
      at::Tensor cx,
      at::Tensor w_ih,
      at::Tensor w_hh) {
  auto gates = input.mm(t_use(w_ih)) + hx.mm(t_use(w_hh));

  auto chunked_gates = gates.chunk(4, 1);
  auto ingate     = chunked_gates[0];
  auto forgetgate = chunked_gates[1];
  auto cellgate = chunked_gates[2];
  auto outgate    = chunked_gates[3];

  ingate = ingate.sigmoid();
  outgate = outgate.sigmoid();
  cellgate = cellgate.tanh();
  forgetgate = forgetgate.sigmoid();

  auto cy = (forgetgate * cx) + (ingate * cellgate);
  auto hy = outgate * cy.tanh();

  return {hy, cy};
}

std::tuple<Var, Var> build_lstm_body(
  Graph & g,
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

    auto cy = forgetgate*cx;
    cy =  cy + ingate*cellgate;
    auto hy = outgate*cy.tanh();

    return std::make_tuple(hy,cy);
}

std::shared_ptr<Graph> build_lstm() {
  auto r = std::make_shared<Graph>();
  auto & g = *r;
  Value * input = g.addInput();
  Value * hx = g.addInput();
  Value * cx = g.addInput();
  Value * w_ih = g.addInput();
  Value * w_hh = g.addInput();

  Var hy;
  Var cy;
  std::tie(hy,cy) = build_lstm_body(g, input, hx, cx, w_ih, w_hh);

  hy.addAsOutput();
  cy.addAsOutput();
  g.lint();

  return r;
}

std::shared_ptr<Graph> build_lstm_stages() {
  auto r = std::make_shared<Graph>();
  auto & g = *r;
  Var input = g.addInput();
  Var hx = g.addInput();
  Var cx = g.addInput();
  Var w_ih = g.addInput();
  Var w_hh = g.addInput();

  Var hy;
  Var cy;
  std::tie(hy,cy) = build_lstm_body(g, input, hx, cx, w_ih, w_hh);

  // use some stuff from the previous stage as well
  // as a new input
  g.advanceStage();
  hx = hy;
  cy.addAsOutput();
  cx = g.addInput();

  std::tie(hy,cy) = build_lstm_body(g, input, hx, cx, w_ih, w_hh);

  hy.addAsOutput();
  cy.addAsOutput();
  g.lint();

  return r;
}

void runOneStage(InterpreterState & interp, const std::vector<at::Tensor> & inputs, std::vector<at::Tensor> & outputs) {
  std::vector<IValue> stack(inputs.begin(), inputs.end());
  interp.runOneStage(stack);
  outputs.clear();
  for(auto & ivalue : stack) {
    outputs.push_back(std::move(ivalue).toTensor());
  }
}

void interpTest() {
    constexpr int batch_size = 4;
    constexpr int input_size = 256;
    constexpr int seq_len = 32;

    int hidden_size = 2*input_size;

    auto input = at::randn({seq_len, batch_size, input_size}, at::kCUDA);
    auto hx    = at::randn({batch_size, hidden_size}, at::kCUDA);
    auto cx    = at::randn({batch_size, hidden_size}, at::kCUDA);
    auto w_ih  = t_def(at::randn({4 * hidden_size, input_size}, at::kCUDA));
    auto w_hh  = t_def(at::randn({4 * hidden_size, hidden_size}, at::kCUDA));

    auto lstm_g = build_lstm();
    Code lstm_function(lstm_g);
    std::vector<at::Tensor> outputs;
    InterpreterState lstm_interp(lstm_function);
    runOneStage(lstm_interp, {input[0], hx, cx, w_ih, w_hh}, outputs);
    std::tie(hx, cx) = lstm(input[0], hx, cx, w_ih, w_hh);

    //std::cout << almostEqual(outputs[0],hx) << "\n";
    REQUIRE(exactlyEqual(outputs[0],hx));
    REQUIRE(exactlyEqual(outputs[1],cx));
}

void interpStageTest() {
    constexpr int batch_size = 4;
    constexpr int input_size = 256;
    constexpr int seq_len = 32;

    int hidden_size = 2*input_size;
    auto input = at::randn({seq_len, batch_size, input_size}, at::kCUDA);
    auto hx    = at::randn({batch_size, hidden_size}, at::kCUDA);
    auto cx    = at::randn({batch_size, hidden_size}, at::kCUDA);
    auto cx1 = at::randn({batch_size, hidden_size}, at::kCUDA);
    auto w_ih  = t_def(at::randn({4 * hidden_size, input_size}, at::kCUDA));
    auto w_hh  = t_def(at::randn({4 * hidden_size, hidden_size}, at::kCUDA));


    auto lstm_g = build_lstm_stages();
    Code lstm_function(lstm_g);
    std::vector<at::Tensor> outputs;
    InterpreterState lstm_interp(lstm_function);
    runOneStage(lstm_interp, {input[0], hx, cx, w_ih, w_hh}, outputs);
    auto cy0 = outputs[0];
    runOneStage(lstm_interp, {cx1}, outputs);
    at::Tensor ihx = outputs[0];
    at::Tensor icx = outputs[1];


    std::tie(hx, cx) = lstm(input[0], hx, cx, w_ih, w_hh);
    std::tie(hx, cx) = lstm(input[0], hx, cx1, w_ih, w_hh);

    //std::cout << almostEqual(outputs[0],hx) << "\n";
    REQUIRE(exactlyEqual(outputs[0],hx));
    REQUIRE(exactlyEqual(outputs[1],cx));
}

using var_meta_type = std::vector<int64_t>;
using var_meta_list = std::vector<var_meta_type>;
using test_fn_type = std::function<variable_list(const variable_list&)>;

struct ADTestSpec {
  ADTestSpec(const char *name, var_meta_list input_meta, test_fn_type test_fn)
    : name(name)
    , input_meta(input_meta)
    , test_fn(test_fn) {}

  variable_list operator()(const variable_list& inputs) const {
    return test_fn(inputs);
  };

  std::vector<Variable> make_vars() const {
    std::vector<Variable> out;
    for (const auto & m : input_meta) {
      out.emplace_back(autograd::make_variable(at::CPU(at::kFloat).tensor(m).normal_(), /*requires_grad=*/true));
    }
    return out;
  }

  const char *name;
  var_meta_list input_meta;
  test_fn_type test_fn;
};

variable_list get_grad_outputs(const variable_list& vars) {
  return fmap(vars, [](const Variable& v) -> Variable {
                      return v.type().tensor(v.sizes()).normal_();
                    });
}

std::shared_ptr<Graph> trace(const ADTestSpec& test, const variable_list& vars_in) {
  std::shared_ptr<tracer::TracingState> state;
  variable_list trace_vars_in;
  std::tie(state, trace_vars_in) = tracer::enter(vars_in);
  auto trace_vars_out = test(trace_vars_in);
  tracer::exit(trace_vars_out);
  return state->graph;
}

variable_list grad(const variable_list& outputs, const variable_list& inputs, const variable_list& grad_outputs) {
  static const auto get_edge = [](const Variable& v) { return v.gradient_edge(); };
  auto & engine = torch::autograd::Engine::get_default_engine();
  return engine.execute(fmap(outputs, get_edge), grad_outputs, true, false, fmap(inputs, get_edge));
}

void assertAllClose(const tensor_list& a, const tensor_list& b) {
  REQUIRE(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    REQUIRE(a[i].is_same_size(b[i]));
    REQUIRE(a[i].allclose(b[i]));
  }
}

std::pair<tensor_list, tensor_list> runGradient(Gradient& grad_spec,
                                                tensor_list& tensors_in,
                                                tensor_list& tensor_grads_in) {
  tensor_list tensors_out, tensor_grads_out;
  Code f_code{grad_spec.f},
      df_code{grad_spec.df};
  InterpreterState f_interpreter { f_code }, df_interpreter { df_code };

  runOneStage(f_interpreter, tensors_in, tensors_out);

  tensor_list df_inputs;
  df_inputs.insert(df_inputs.end(), tensor_grads_in.begin(), tensor_grads_in.end());
  for(auto offset : grad_spec.df_input_captured_inputs)
    df_inputs.push_back(tensors_in[offset]);
  for(auto offset : grad_spec.df_input_captured_outputs)
    df_inputs.push_back(tensors_out[offset]);
  runOneStage(df_interpreter, df_inputs, tensor_grads_out);

  // Outputs of f needs to be sliced
  tensors_out.erase(tensors_out.begin() + grad_spec.f_real_outputs, tensors_out.end());
  return std::make_pair(tensors_out, tensor_grads_out);
}

void testADFormulas() {
  static const auto unwrap = [](const Variable& v) { return v.data(); };

  using VL = variable_list;
  static const var_meta_list binary_pointwise = {{2, 3, 4, 5}, {2, 3, 4, 5}};
  static const var_meta_list unary_pointwise  = {{2, 3, 4, 5}};
  static const var_meta_list unary_pointwise_2d  = {{2, 3}};
  static const std::vector<ADTestSpec> ad_tests = {
    {"add",     binary_pointwise, [](const VL& v) -> VL { return {v[0] + v[1]}; }},
    {"sub",     binary_pointwise, [](const VL& v) -> VL { return {v[0] - v[1]}; }},
    {"mul",     binary_pointwise, [](const VL& v) -> VL { return {v[0] * v[1]}; }},
    {"sigmoid", unary_pointwise,  [](const VL& v) -> VL { return {v[0].sigmoid()}; }},
    {"tanh",    unary_pointwise,  [](const VL& v) -> VL { return {v[0].tanh()}; }},
    {"t",       unary_pointwise_2d,  [](const VL& v) -> VL { return {v[0].t()}; }},
    {"mm",      {{10, 12}, {12, 15}}, [](const VL& v) -> VL { return {v[0].mm(v[1])}; }},
    {"chunk",   {{10, 12, 15}}, [](const VL& v) -> VL { return fmap<Variable>(v[0].chunk(4, 1)); }},
    {"chunk",   {{10, 12, 15}}, [](const VL& v) -> VL { return fmap<Variable>(v[0].chunk(3, 2)); }},
    {"split",   {{10, 12, 15}}, [](const VL& v) -> VL { return fmap<Variable>(v[0].split(4, 1)); }},
    {"split",   {{10, 12, 15}}, [](const VL& v) -> VL { return fmap<Variable>(v[0].split(3, 2)); }},
  };

  for (const auto & test : ad_tests) {
    // Get reference values form autograd
    auto vars_in        = test.make_vars();
    auto vars_out       = test(vars_in);
    auto var_grads_in   = get_grad_outputs(vars_out);
    auto var_grads_out  = grad(vars_out, vars_in, var_grads_in);

    // Trace and differentiate the op
    auto graph = trace(test, vars_in);
    EliminateDeadCode(graph); // Tracing of some ops depends on the DCE trick
    auto grad_spec = differentiate(graph, std::vector<bool>(vars_in.size(), true));
    LowerGradOf(*grad_spec.df);
    // Get outputs from the interpreter
    auto tensors_in                = fmap(vars_in, unwrap);
    auto tensor_grads_in           = fmap(var_grads_in, unwrap);
    tensor_list tensors_out, tensor_grads_out;
    std::tie(tensors_out, tensor_grads_out) = runGradient(grad_spec, tensors_in, tensor_grads_in);

    // Compare results
    auto expected_tensors_out      = fmap(vars_out, unwrap);
    auto expected_tensor_grads_out = fmap(var_grads_out, unwrap);
    assertAllClose(tensors_out,      expected_tensors_out);
    assertAllClose(tensor_grads_out, expected_tensor_grads_out);
  }
}

std::string toString(std::shared_ptr<Graph>& graph) {
  std::ostringstream s;
  s << *graph;
  return s.str();
}

void testDifferentiate(std::ostream & out) {
  auto graph = std::make_shared<Graph>();
  at::ScalarType s = at::ScalarType::Float;
  auto type = TensorType::create(s, -1, {2, 3, 4}, {12, 4, 1});

  // Build up a fake graph
  auto a = SymbolicVariable::asNewInput(*graph, type);
  auto b = SymbolicVariable::asNewInput(*graph, type);
  auto c = a * b * a + b;
  graph->registerOutput(c.value());

  auto grad_spec = differentiate(graph, {true, true});
  std::vector<size_t> expected_captured_inputs = {0, 1};
  std::vector<size_t> expected_captured_outputs = {1};
  std::vector<size_t> expected_input_vjps = {0, 1};
  std::vector<size_t> expected_output_vjps = {0, 1};
  REQUIRE(grad_spec.f_real_outputs == 1);
  REQUIRE(grad_spec.df_input_captured_inputs == expected_captured_inputs);
  REQUIRE(grad_spec.df_input_captured_outputs == expected_captured_outputs);
  REQUIRE(grad_spec.df_input_vjps == expected_input_vjps);
  REQUIRE(grad_spec.df_output_vjps == expected_output_vjps);
  out << "testDifferentiate\n";
  out << *grad_spec.f;
  out << *grad_spec.df;
  out << "\n";
}

void testDifferentiateWithRequiresGrad(std::ostream & out) {
  auto graph = std::make_shared<Graph>();
  at::ScalarType s = at::ScalarType::Float;
  auto type = TensorType::create(s, -1, {2, 3, 4}, {12, 4, 1});

  // Build up a fake graph
  auto a = SymbolicVariable::asNewInput(*graph, type);
  auto b = SymbolicVariable::asNewInput(*graph, type);
  auto d = b * b + b;
  auto e = (d + a) * a + b;
  graph->registerOutput(d.value());
  graph->registerOutput(e.value());

  auto grad_spec = differentiate(graph, {true, false});
  std::vector<size_t> expected_input_vjps = {1, 2};  // for e and %4 = (d + a)
  std::vector<size_t> expected_output_vjps = {0};    // only a requires grad
  REQUIRE(grad_spec.f_real_outputs == 2);              // we need one temporary %4 = (d + a)
  REQUIRE(grad_spec.df_input_captured_inputs == std::vector<size_t>({0}));
  REQUIRE(grad_spec.df_input_captured_outputs == std::vector<size_t>({2}));
  REQUIRE(grad_spec.df_input_vjps == expected_input_vjps);
  REQUIRE(grad_spec.df_output_vjps == expected_output_vjps);
  out << "testDifferentiateWithRequiresGrad\n";
  out << *grad_spec.f;
  out << *grad_spec.df;
  out << "\n";
}

void testCreateAutodiffSubgraphs(std::ostream & out) {
  auto graph = build_lstm();
  CreateAutodiffSubgraphs(*graph, /*threshold=*/2);
  out << "testCreateAutodiffSubgraphs\n";
  out << *graph << "\n";
}

autograd::Variable var(at::Type & t, at::IntList sizes, bool requires_grad) {
  return autograd::make_variable(at::rand(sizes, t), requires_grad);
}
autograd::Variable undef() {
  return autograd::Variable();
}

int device(const autograd::Variable & v) {
  return v.type().is_cuda() ? v.get_device() : -1;
}

bool isEqual(at::IntList lhs, at::IntList rhs) {
  return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool isEqual(const ArgumentInfo & ti, const autograd::Variable & v) {
  REQUIRE(ti.isTensor());
  if(!ti.defined())
    return ti.defined() == v.defined();
  return
    ti.device() == device(v) &&
    ti.requires_grad() == v.requires_grad() &&
    ti.type() == v.type().scalarType() &&
    isEqual(ti.sizes(), v.sizes()) &&
    isEqual(ti.strides(), v.strides());
}

// work around the fact that variable_tensor_list doesn't duplicate all
// of std::vector's constructors.
// most constructors are never used in the implementation, just in our tests.
Stack createStack(std::vector<at::Tensor> && list) {
  return Stack(std::make_move_iterator(list.begin()), std::make_move_iterator(list.end()));
}

void argumentSpecTest() {
  auto & CF = at::CPU(at::kFloat);
  auto & CD = at::CPU(at::kDouble);
  auto & GF = at::CUDA(at::kFloat);
  auto & GD = at::CUDA(at::kDouble);

  auto list = createStack({ var(CF, {1}, true), var(CD, {1, 2}, false) , var(GF, {}, true), var(GD, {4,5,6}, false), undef()});

  // make sure we have some non-standard strides
  list[1].toTensor().transpose_(0, 1);

  // same list but different backing values
  auto list2 = createStack({ var(CF, {1}, true), var(CD, {1, 2}, false) , var(GF, {}, true), var(GD, {4,5,6}, false), undef()});
  list2[1].toTensor().transpose_(0, 1);


  ArgumentSpec a(true, list);
  ArgumentSpec b(true, list);
  REQUIRE(a.hashCode() == b.hashCode());

  REQUIRE(a == b);
  ArgumentSpec d(true, list2);
  REQUIRE(d == a);
  REQUIRE(d.hashCode() == a.hashCode());

  for(size_t i = 0; i < list.size(); ++i) {
    REQUIRE(isEqual(a.at(i), list[i].toTensor()));
  }
  ArgumentSpec no_grad(/*with_grad=*/false, list);
  REQUIRE(no_grad != a);

  std::unordered_set<ArgumentSpec> spec;
  spec.insert(std::move(a));
  REQUIRE(spec.count(b) > 0);
  REQUIRE(spec.count(no_grad) == 0);
  spec.insert(std::move(no_grad));
  REQUIRE(spec.count(ArgumentSpec(true,list)) == 1);

  list2[1].toTensor().transpose_(0,1);
  ArgumentSpec c(true, list2); // same as list, except for one stride
  REQUIRE(!(c == a));
  REQUIRE(spec.count(c) == 0);

  Stack stack = { var(CF, {1,2}, true), 3, var(CF, {1,2}, true) };
  ArgumentSpec with_const(true, stack);
  REQUIRE(with_const.at(2).sizes().size() == 2);
}

void shapeAnalysisTest() {

  constexpr int batch_size = 4;
  constexpr int input_size = 256;

  int hidden_size = 2*input_size;

  auto v = [](at::Tensor t) { return autograd::make_variable(t, false); };

  auto input = at::randn({batch_size, input_size}, at::kCUDA);
  auto hx    = at::randn({batch_size, hidden_size}, at::kCUDA);
  auto cx    = at::randn({batch_size, hidden_size}, at::kCUDA);
  auto w_ih  = t_def(at::randn({4 * hidden_size, input_size}, at::kCUDA));
  auto w_hh  = t_def(at::randn({4 * hidden_size, hidden_size}, at::kCUDA));

  auto g = build_lstm();
  ArgumentSpec spec(false, createStack({v(input), v(hx), v(cx), v(w_ih), v(w_hh) }));
  PropagateInputShapes(*g, spec);
  at::Tensor r0, r1;
  std::tie(r0, r1) = lstm(input, hx, cx, w_ih, w_hh);
  auto o0 = g->outputs()[0]->type()->expect<TensorType>();
  auto o1 = g->outputs()[1]->type()->expect<TensorType>();
  REQUIRE(o0->sizes() == std::vector<int64_t>(r0.sizes().begin(), r0.sizes().end()));
  REQUIRE(o1->sizes() == std::vector<int64_t>(r1.sizes().begin(), r1.sizes().end()));

}

void testGraphExecutor() {
  constexpr int batch_size = 4;
  constexpr int input_size = 256;

  int hidden_size = 2*input_size;

  auto v = [](at::Tensor t) { return autograd::make_variable(t, false); };

  auto input = at::randn({batch_size, input_size}, at::kCUDA);
  auto hx    = at::randn({batch_size, hidden_size}, at::kCUDA);
  auto cx    = at::randn({batch_size, hidden_size}, at::kCUDA);
  auto w_ih  = t_def(at::randn({4 * hidden_size, input_size}, at::kCUDA));
  auto w_hh  = t_def(at::randn({4 * hidden_size, hidden_size}, at::kCUDA));

  auto g = build_lstm();
  GraphExecutor executor(g);
  auto stack = createStack({v(input), v(hx), v(cx), v(w_ih), v(w_hh)});
  executor.run(stack);
  REQUIRE(stack.size() == 2);
  at::Tensor r0, r1;
  std::tie(r0, r1) = lstm(input, hx, cx, w_ih, w_hh);
  REQUIRE(almostEqual(Variable(stack[0].toTensor()).data(), r0));
  REQUIRE(almostEqual(Variable(stack[1].toTensor()).data(), r1));
}

void testBlocks(std::ostream & out) {
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
    auto  d = b + c;
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


const static auto cf_examples = R"JIT(
  def if_test(a, b):
      # FIXME: use 0 instead of a.
      # c = 0
      c = a
      if a < b:
        c = b
      else:
        c = a
      return c
  def if_one(a, b):
    c = b
    if a < b:
      c = a
    return c
  def while_test(a, i):
    while i < 3:
      a *= a
      i += 1
    return a
)JIT";
void testControlFlow() {
  script::Module cu;
  script::defineMethodsInModule(cu, cf_examples, torch::jit::script::Resolver(), nullptr);
  auto run = [&](const std::string & name, std::vector<IValue> stack) {
    auto graph = cu.get_method(name).graph();
    Code code(graph);
    InterpreterState interp(code);
    interp.runOneStage(stack);
    return stack;
  };

  auto L = [](int64_t l) { return IValue(autograd::make_variable(at::Scalar(l).toTensor())); };
  auto V = [](IValue t) { return at::Scalar(std::move(t).toTensor()).toLong(); };
  auto run_binary = [&](const std::string & name, int64_t a, int64_t b) {
    return V(run(name, {L(a), L(b)})[0]);
  };
  REQUIRE(2 == run_binary("if_test", 1, 2));
  REQUIRE(3 == run_binary("if_test", 3, 2));
  REQUIRE(2 == run_binary("if_one", 2, 3));
  REQUIRE(2 == run_binary("if_one", 3, 2));
  REQUIRE(256 == run_binary("while_test",2,0));
}

void testIValue() {
  Shared<IntList> foo = IntList::create({3, 4, 5});
  JIT_ASSERT(foo->use_count() == 1);
  IValue bar(foo);
  JIT_ASSERT(foo->use_count() == 2);
  auto baz = bar;
  JIT_ASSERT(foo->use_count() == 3);
  auto foo2 = std::move(bar);
  JIT_ASSERT(foo->use_count() == 3);
  JIT_ASSERT(foo2.isIntList());
  JIT_ASSERT(bar.isNone());
  foo2 = IValue(4.0);
  JIT_ASSERT(foo2.isDouble());
  JIT_ASSERT(foo2.toDouble() == 4.0);
  JIT_ASSERT(foo->use_count() == 2);
  JIT_ASSERT(ArrayRef<int64_t>(baz.toIntList()->elements()).equals({3,4,5}));

  auto move_it = std::move(baz).toIntList();
  JIT_ASSERT(foo->use_count() == 2);
  JIT_ASSERT(baz.isNone());
  IValue i(4);
  JIT_ASSERT(i.isInt() && i.toInt() == 4);
  IValue dlist(DoubleList::create({3.5}));
  JIT_ASSERT(
      dlist.isDoubleList() &&
      ArrayRef<double>(std::move(dlist).toDoubleList()->elements())
          .equals({3.5}));
  JIT_ASSERT(dlist.isNone());
  dlist = IValue(DoubleList::create({3.4}));
  JIT_ASSERT(ArrayRef<double>(dlist.toDoubleList()->elements()).equals({3.4}));
  IValue the_list(Tuple::create({IValue(3.4), IValue(4), IValue(foo)}));
  JIT_ASSERT(foo->use_count() == 3);
  JIT_ASSERT(the_list.isTuple());
  auto first = std::move(the_list).toTuple()->elements().at(1);
  JIT_ASSERT(first.toInt() == 4);
  at::Tensor tv = at::rand({3,4});
  IValue ten(tv);
  JIT_ASSERT(tv.get()->use_count() == 2);
  auto ten2 = ten;
  JIT_ASSERT(tv.get()->use_count() == 3);
  JIT_ASSERT(ten2.toTensor().equal(ten.toTensor()));
  std::move(ten2).toTensor();
  JIT_ASSERT(tv.get()->use_count() == 2);
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
    REQUIRE(ops.size() == 1);

    auto& op = ops.front();
    REQUIRE(op->schema().name == "foo::bar");

    REQUIRE(op->schema().arguments.size() == 2);
    REQUIRE(op->schema().arguments[0].name == "_0");
    REQUIRE(op->schema().arguments[0].type->kind() == TypeKind::FloatType);
    REQUIRE(op->schema().arguments[1].name == "_1");
    REQUIRE(op->schema().arguments[1].type->kind() == TypeKind::DynamicType);

    REQUIRE(op->schema().returns[0].type->kind() == TypeKind::DynamicType);

    Stack stack;
    push(stack, 2.0f, autograd::make_variable(at::ones(5)));
    op->getOperation()(stack);
    at::Tensor output;
    pop(stack, output);

    REQUIRE(output.allclose(autograd::make_variable(at::full(5, 3.0f))));
  }
  {
    RegisterOperators reg({createOperator(
        "foo::bar_with_schema(float a, Tensor b) -> Tensor",
        [](double a, at::Tensor b) { return a + b; })});

    auto& ops =
        getAllOperatorsFor(Symbol::fromQualString("foo::bar_with_schema"));
    REQUIRE(ops.size() == 1);

    auto& op = ops.front();
    REQUIRE(op->schema().name == "foo::bar_with_schema");

    REQUIRE(op->schema().arguments.size() == 2);
    REQUIRE(op->schema().arguments[0].name == "a");
    REQUIRE(op->schema().arguments[0].type->kind() == TypeKind::FloatType);
    REQUIRE(op->schema().arguments[1].name == "b");
    REQUIRE(op->schema().arguments[1].type->kind() == TypeKind::DynamicType);

    REQUIRE(op->schema().returns.size() == 1);
    REQUIRE(op->schema().returns[0].type->kind() == TypeKind::DynamicType);

    Stack stack;
    push(stack, 2.0f, autograd::make_variable(at::ones(5)));
    op->getOperation()(stack);
    at::Tensor output;
    pop(stack, output);

    REQUIRE(output.allclose(autograd::make_variable(at::full(5, 3.0f))));
  }
  {
    // Check that lists work well.
    RegisterOperators reg({createOperator(
        "foo::lists(int[] ints, float[] floats, Tensor[] tensors) -> float[]",
        [](const std::vector<int64_t>& ints,
           const std::vector<double>& floats,
           std::vector<at::Tensor> tensors) { return floats; })});

    auto& ops =
        getAllOperatorsFor(Symbol::fromQualString("foo::lists"));
    REQUIRE(ops.size() == 1);

    auto& op = ops.front();
    REQUIRE(op->schema().name == "foo::lists");

    REQUIRE(op->schema().arguments.size() == 3);
    REQUIRE(op->schema().arguments[0].name == "ints");
    REQUIRE(op->schema().arguments[0].type->isSubtypeOf(ListType::ofInts()));
    REQUIRE(op->schema().arguments[1].name == "floats");
    REQUIRE(op->schema().arguments[1].type->isSubtypeOf(ListType::ofFloats()));
    REQUIRE(op->schema().arguments[2].name == "tensors");
    REQUIRE(op->schema().arguments[2].type->isSubtypeOf(ListType::ofTensors()));

    REQUIRE(op->schema().returns.size() == 1);
    REQUIRE(op->schema().returns[0].type->isSubtypeOf(ListType::ofFloats()));

    Stack stack;
    push(stack, std::vector<int64_t>{1, 2});
    push(stack, std::vector<double>{1.0, 2.0});
    push(stack, std::vector<at::Tensor>{autograd::make_variable(at::ones(5))});
    op->getOperation()(stack);
    std::vector<double> output;
    pop(stack, output);

    REQUIRE(output.size() == 2);
    REQUIRE(output[0] == 1.0);
    REQUIRE(output[1] == 2.0);
  }
  {
    RegisterOperators reg(
        "foo::lists2(Tensor[] tensors) -> Tensor[]",
        [](std::vector<at::Tensor> tensors) { return tensors; });

    auto& ops =
        getAllOperatorsFor(Symbol::fromQualString("foo::lists2"));
    REQUIRE(ops.size() == 1);

    auto& op = ops.front();
    REQUIRE(op->schema().name == "foo::lists2");

    REQUIRE(op->schema().arguments.size() == 1);
    REQUIRE(op->schema().arguments[0].name == "tensors");
    REQUIRE(op->schema().arguments[0].type->isSubtypeOf(ListType::ofTensors()));

    REQUIRE(op->schema().returns.size() == 1);
    REQUIRE(op->schema().returns[0].type->isSubtypeOf(ListType::ofTensors()));

    Stack stack;
    push(stack, std::vector<at::Tensor>{autograd::make_variable(at::ones(5))});
    op->getOperation()(stack);
    std::vector<at::Tensor> output;
    pop(stack, output);

    REQUIRE(output.size() == 1);
    REQUIRE(output[0].allclose(autograd::make_variable(at::ones(5))));
  }
  {
#ifdef USE_CATCH
    REQUIRE_THROWS_WITH(
        createOperator(
            "foo::bar_with_bad_schema(Tensor a) -> Tensor",
            [](double a, at::Tensor b) { return a + b; }),
        StartsWith("Inferred 2 argument(s) for operator implementation, "
                   "but the provided schema specified 1 argument(s)."));
    REQUIRE_THROWS_WITH(
        createOperator(
            "foo::bar_with_bad_schema(Tensor a) -> Tensor",
            [](double a) { return a; }),
        StartsWith("Inferred type for argument #0 was float, "
                   "but the provided schema specified type Dynamic "
                   "for the argument in that position"));
    REQUIRE_THROWS_WITH(
        createOperator(
            "foo::bar_with_bad_schema(float a) -> (float, float)",
            [](double a) { return a; }),
        StartsWith("Inferred 1 return value(s) for operator implementation, "
                   "but the provided schema specified 2 return value(s)."));
    REQUIRE_THROWS_WITH(
        createOperator(
            "foo::bar_with_bad_schema(float a) -> Tensor",
            [](double a) { return a; }),
        StartsWith("Inferred type for return value #0 was float, "
                   "but the provided schema specified type Dynamic "
                   "for the return value in that position"));
#endif // USE_CATCH
  }
  {
    auto op = createOperator(
        "traced::op(float a, Tensor b) -> Tensor",
        [](double a, at::Tensor b) { return a + b; });

    std::shared_ptr<tracer::TracingState> state;
    variable_list trace_vars_in;
    std::tie(state, trace_vars_in) = tracer::enter({});

    Stack stack;
    push(stack, 2.0f, autograd::make_variable(at::ones(5)));
    op.getOperation()(stack);
    at::Tensor output = autograd::make_variable(at::empty({}));
    pop(stack, output);

    tracer::exit({output});

    std::string op_name("traced::op");
    bool contains_traced_op = false;
    for (const auto& node : state->graph->nodes()) {
      if (std::string(node->kind().toQualString()) == op_name) {
        contains_traced_op = true;
        break;
      }
    }
    REQUIRE(contains_traced_op);
  }
  {
#ifdef USE_CATCH
    // vector<double> is not supported yet.
    auto op = createOperator(
        "traced::op(float[] f) -> int",
        [](const std::vector<double>& f) -> int64_t { return f.size(); });

    std::shared_ptr<tracer::TracingState> state;
    variable_list trace_vars_in;
    std::tie(state, trace_vars_in) = tracer::enter({});

    Stack stack;
    push(stack, std::vector<double>{1.0});

    REQUIRE_THROWS_WITH(
        op.getOperation()(stack),
        StartsWith("Tracing float lists currently not supported!"));
#endif
  }
}

TORCH_API std::string runJITCPPTests() {
  std::stringstream out;
  testIValue();
  testControlFlow();
  testGraphExecutor();
  testBlocks(out);
  testCreateAutodiffSubgraphs(out);
  testDifferentiate(out);
  testDifferentiateWithRequiresGrad(out);
  testADFormulas();
  interpTest();
  interpStageTest();
  codeTemplateTest();
  fusionTests();
  attributesTest();
  internedStringsTests();
  fromQualStringTests();
  argumentSpecTest();
  shapeAnalysisTest();
  testProto();
  testCustomOperators();
  return out.str();
}

#ifdef USE_CATCH

TEST_CASE( "jit test CPU", "[cpu]" ) {

  std::stringstream out;
  SECTION( "control flow" )
    testControlFlow();
  SECTION( "blocks" )
    testBlocks(out);
  SECTION( "create autodiff subgraphs" )
    testCreateAutodiffSubgraphs(out);
  SECTION( "differentiate" )
    testDifferentiate(out);
  SECTION( "differentiate with requires grad" )
    testDifferentiateWithRequiresGrad(out);
  SECTION( "AD formulas" )
    testADFormulas();
  SECTION( "code template" )
    codeTemplateTest();
  SECTION( "attributes" )
    attributesTest();
  SECTION( "interned strings" )
    internedStringsTests();
  SECTION( "custom operators" )
    testCustomOperators();
}

TEST_CASE( "jit test CUDA", "[cuda]" ) {

  SECTION( "graph executor" )
    testGraphExecutor();
  SECTION( "fusion" )
    fusionTests();
  SECTION( "interp" )
    interpTest();
  SECTION( "interp stage" )
    interpStageTest();
  SECTION( "argument spec" )
    argumentSpecTest();
  SECTION( "shape analysis" )
    shapeAnalysisTest();
}

#endif

}}
