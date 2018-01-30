#include <Python.h>
#include "torch/csrc/jit/fusion_compiler.h"
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/attributes.h"
#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/symbolic_variable.h"
#include "torch/csrc/jit/autodiff.h"
#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/hash.h"
#include "torch/csrc/jit/argument_spec.h"
#include "torch/csrc/jit/passes/shape_analysis.h"

#include "torch/csrc/assertions.h"
#include "torch/csrc/utils/auto_gil.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/python_engine.h"
#include "torch/csrc/jit/passes/shape_analysis.h"

#include <vector>
#include <iostream>

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
    JIT_ASSERT(e.s("hi") == "foo");
    JIT_ASSERT(c.s("hi") == "foo2");
    JIT_ASSERT(e.v("what")[0] == "is");
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
    JIT_ASSERT(s == ct_expect);
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
    auto a = at::CUDA(at::kFloat).rand({3,4});
    auto b = at::CUDA(at::kFloat).rand({4,3}).transpose(0,1);
    auto o = at::CUDA(at::kFloat).zeros({3,4});
    comp.debugLaunchGraph(graph, 0, {a,b}, {o});
    auto o2 = a*b;
    float max_diff = (o2 - o).abs().max().toCDouble();
    //std::cout << "max diff: " << max_diff << "\n";
    JIT_ASSERT(max_diff == 0);
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
      inputs.push_back(at::CUDA(at::kFloat).rand(dims).transpose(ti, tj));
    }
    for(size_t i = 0; i < graph.outputs().size(); i++) {
      std::vector<int64_t> dims = {128, 128, 32};
      std::swap(dims[toi],dims[toj]);
      outputs.push_back(at::CUDA(at::kFloat).zeros(dims).transpose(toi,toj));
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
    JIT_ASSERT(out0.is_same_size(outputs.front()));
    float max_diff = (outputs.front() - out0).abs().max().toCDouble();
    JIT_ASSERT(max_diff < 1e-6);

  };
  testOne(0,0,0,0);
  testOne(0,1,0,0);
  testOne(1,2,0,0);
  testOne(0,2,0,0);

  testOne(0,0,0,1);
  testOne(0,1,1,2);
  testOne(1,2,0,2);



  auto testConcat = [&](int dim) {
    Graph graph;
    Var i0 = Var::asNewInput(graph);
    Var i1 = Var::asNewInput(graph);
    auto o0 = i0 * i1;
    o0.addAsOutput();
    Var::cat({i0, o0}, dim).addAsOutput();

    auto a = at::CUDA(at::kFloat).rand({3,4,5});
    auto b = at::CUDA(at::kFloat).rand({4,3,5}).transpose(0,1);
    auto o = at::CUDA(at::kFloat).zeros({3,4,5});

    auto o_r = a*b;
    auto o2_r = at::cat({a, o_r}, dim);
    auto o2 = at::CUDA(at::kFloat).zeros(o2_r.sizes());
    comp.debugLaunchGraph(graph, 0, {a,b}, {o, o2});

    float max_diff = (o_r - o).abs().max().toCDouble();
    JIT_ASSERT(max_diff == 0);
    float max_diff2 = (o2_r - o2).abs().max().toCDouble();
    JIT_ASSERT(max_diff2 == 0);
  };
  testConcat(0);
  testConcat(1);
  testConcat(2);
}

struct Attr : public Attributes<Attr> {
};
void attributesTest() {
  auto one = kParam;
  auto two = kReturn;
  auto three = kConstant;
  auto four = kSlice;
  Attr attr;
  attr.f_(one,3.4)->i_(two,5)->s_(three,"what");
  JIT_ASSERT(attr.f(one) == 3.4);
  JIT_ASSERT(attr.s(three) == "what");
  JIT_ASSERT(attr.i(two) == 5);
  attr.s_(one,"no");
  JIT_ASSERT(attr.s(one) == "no");
  JIT_ASSERT(attr.hasAttribute(three));
  JIT_ASSERT(!attr.hasAttribute(four));
  attr.ss_(two, {"hi", "now"});
  JIT_ASSERT(attr.ss(two).at(1) == "now");

  Attr attr2;
  attr2.copyAttributes(attr);
  JIT_ASSERT(attr2.s(one) == "no");
  attr2.f_(one,5);
  JIT_ASSERT(attr.s(one) == "no");
  JIT_ASSERT(attr2.f(one) == 5);
}

void internedStringsTests () {

  JIT_ASSERT(kParam == Symbol("Param"));
  JIT_ASSERT(kReturn == Symbol("Return"));
  JIT_ASSERT(Symbol(kReturn).toString() == std::string("Return"));
  size_t symstart = Symbol("__NEW_SYMBOL");
  JIT_ASSERT(Symbol("What") == symstart+1);
  JIT_ASSERT(Symbol("What2") == symstart+2);
  JIT_ASSERT(Symbol("What") == symstart+1);
  JIT_ASSERT(Symbol("What2") == symstart+2);
  JIT_ASSERT(Symbol(symstart+2).toString() == std::string("What2"));
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
    auto gates =  input.mm(w_ih) + hx.mm(w_hh);
    auto outputs = gates.chunk(4, 1);
    auto ingate = outputs[0];
    auto forgetgate = outputs[1];
    auto cellgate = outputs[2];
    auto outgate = outputs[3];
    ingate = ingate.sigmoid();
    outgate = outgate.sigmoid();
    cellgate = cellgate.tanh();
    forgetgate = forgetgate.sigmoid();

    auto cy = forgetgate*cx + ingate*cellgate;
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


void interpTest() {
    constexpr int batch_size = 4;
    constexpr int input_size = 256;
    constexpr int seq_len = 32;

    int hidden_size = 2*input_size;

    auto input = at::CUDA(at::kFloat).randn({seq_len, batch_size, input_size});
    auto hx    = at::CUDA(at::kFloat).randn({batch_size, hidden_size});
    auto cx    = at::CUDA(at::kFloat).randn({batch_size, hidden_size});
    auto w_ih  = t_def(at::CUDA(at::kFloat).randn({4 * hidden_size, input_size}));
    auto w_hh  = t_def(at::CUDA(at::kFloat).randn({4 * hidden_size, hidden_size}));

    auto lstm_g = build_lstm();
    Code  lstm_function(lstm_g);
    std::vector<at::Tensor> outputs;
    InterpreterState lstm_interp(lstm_function);
    lstm_interp.runOneStage({input[0], hx, cx, w_ih, w_hh}, outputs);
    std::tie(hx, cx) = lstm(input[0], hx, cx, w_ih, w_hh);

    //std::cout << almostEqual(outputs[0],hx) << "\n";
    JIT_ASSERT(exactlyEqual(outputs[0],hx));
    JIT_ASSERT(exactlyEqual(outputs[1],cx));
}

void interpStageTest() {
    constexpr int batch_size = 4;
    constexpr int input_size = 256;
    constexpr int seq_len = 32;

    int hidden_size = 2*input_size;
    auto input = at::CUDA(at::kFloat).randn({seq_len, batch_size, input_size});
    auto hx    = at::CUDA(at::kFloat).randn({batch_size, hidden_size});
    auto cx    = at::CUDA(at::kFloat).randn({batch_size, hidden_size});
    auto cx1 = at::CUDA(at::kFloat).randn({batch_size, hidden_size});
    auto w_ih  = t_def(at::CUDA(at::kFloat).randn({4 * hidden_size, input_size}));
    auto w_hh  = t_def(at::CUDA(at::kFloat).randn({4 * hidden_size, hidden_size}));


    auto lstm_g = build_lstm_stages();
    Code lstm_function(lstm_g);
    std::vector<at::Tensor> outputs;
    InterpreterState lstm_interp(lstm_function);
    lstm_interp.runOneStage({input[0], hx, cx, w_ih, w_hh}, outputs);
    auto cy0 = outputs[0];
    lstm_interp.runOneStage({cx1}, outputs);
    at::Tensor ihx = outputs[0];
    at::Tensor icx = outputs[1];


    std::tie(hx, cx) = lstm(input[0], hx, cx, w_ih, w_hh);
    std::tie(hx, cx) = lstm(input[0], hx, cx1, w_ih, w_hh);

    //std::cout << almostEqual(outputs[0],hx) << "\n";
    JIT_ASSERT(exactlyEqual(outputs[0],hx));
    JIT_ASSERT(exactlyEqual(outputs[1],cx));
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
      out.emplace_back(make_variable(at::CPU(at::kFloat).tensor(m).normal_(), true));
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
  std::tie(state, trace_vars_in) = tracer::enter(fmap<tracer::TraceInput>(vars_in), 1);
  auto trace_vars_out = test(trace_vars_in);
  tracer::exit(trace_vars_out);
  return state->graph;
}

variable_list grad(const variable_list& outputs, const variable_list& inputs, const variable_list& grad_outputs) {
  static const auto get_edge = [](const Variable& v) -> edge_type {
    return std::make_pair(v.grad_fn() ? v.grad_fn() : v.grad_accumulator(), v.output_nr());
  };
  auto & engine = torch::autograd::python::PythonEngine::getDefaultEngine();
  return engine.execute(fmap(outputs, get_edge), grad_outputs, true, false, fmap(inputs, get_edge));
}

void assertAllClose(const tensor_list& a, const tensor_list& b) {
  JIT_ASSERT(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    JIT_ASSERT(a[i].is_same_size(b[i]));
    JIT_ASSERT(a[i].allclose(b[i]));
  }
}

std::pair<tensor_list, tensor_list> runGradient(Gradient& grad_spec,
                                                tensor_list& tensors_in,
                                                tensor_list& tensor_grads_in) {
  tensor_list tensors_out, tensor_grads_out;
  Code f_code { grad_spec.f }, df_code { grad_spec.df };
  InterpreterState f_interpreter { f_code }, df_interpreter { df_code };

  f_interpreter.runOneStage(tensors_in, tensors_out);

  tensor_list df_inputs;
  for (auto capture : grad_spec.df_input_captures) {
    bool is_input = capture.kind == Capture::Kind::Input;
    df_inputs.push_back(is_input ? tensors_in[capture.offset] : tensors_out[capture.offset]);
  }
  df_inputs.insert(df_inputs.end(), tensor_grads_in.begin(), tensor_grads_in.end());
  df_interpreter.runOneStage(df_inputs, tensor_grads_out);

  // Outputs of f needs to be sliced
  tensors_out.erase(tensors_out.begin() + grad_spec.f_real_outputs, tensors_out.end());
  return std::make_pair(tensors_out, tensor_grads_out);
}

void testADFormulas() {
  static const auto unwrap = [](const Variable& v) { return v.data(); };

  using VL = variable_list;
  static const var_meta_list binary_pointwise = {{2, 3, 4, 5}, {2, 3, 4, 5}};
  static const std::vector<ADTestSpec> ad_tests = {
    {"add", binary_pointwise, [](const VL& v) -> VL { return {v[0] + v[1]}; }},
    {"sub", binary_pointwise, [](const VL& v) -> VL { return {v[0] - v[1]}; }},
    {"mul", binary_pointwise, [](const VL& v) -> VL { return {v[0] * v[1]}; }},
  };

  // We have to release the GIL inside this method, because if we happen to
  // initialize the autograd engine here, the newly spawned worker threads will
  // try to initialize their PyThreadState*, and they need the GIL for this.
  AutoNoGIL _no_gil;
  for (const auto & test : ad_tests) {
    // Get reference values form autograd
    auto vars_in        = test.make_vars();
    auto vars_out       = test(vars_in);
    auto var_grads_in   = get_grad_outputs(vars_out);
    auto var_grads_out  = grad(vars_out, vars_in, var_grads_in);

    // Trace and differentiate the op
    auto graph = trace(test, vars_in);
    auto grad_spec = differentiate(graph, std::vector<bool>(vars_in.size(), true));

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

bool operator==(const Capture& a, const Capture& b) {
  return a.kind == b.kind && a.offset == b.offset;
}

void testDifferentiate(std::ostream & out) {
  auto graph = std::make_shared<Graph>();
  at::ScalarType s = at::ScalarType::Float;
  auto type = std::shared_ptr<TensorType>(new TensorType(s, -1, {2, 3, 4}, {12, 4, 1}));

  // Build up a fake graph
  auto a = SymbolicVariable::asNewInput(*graph, type);
  auto b = SymbolicVariable::asNewInput(*graph, type);
  auto c = a * b * a + b;
  graph->registerOutput(c.value());

  auto grad_spec = differentiate(graph, {true, true});
  std::vector<Capture> expected_captures = {
    {Capture::Kind::Input, 0},
    {Capture::Kind::Input, 1},
    {Capture::Kind::Output, 1},
  };
  std::vector<std::size_t> expected_input_vjps = {0, 1};
  std::vector<std::size_t> expected_output_vjps = {0, 1};
  JIT_ASSERT(grad_spec.f_real_outputs == 1);
  JIT_ASSERT(grad_spec.df_input_captures == expected_captures);
  JIT_ASSERT(grad_spec.df_input_vjps == expected_input_vjps);
  JIT_ASSERT(grad_spec.df_output_vjps == expected_output_vjps);
  out << "testDifferentiate\n";
  out << *grad_spec.f;
  out << *grad_spec.df;
  out << "\n";
}

void testDifferentiateWithRequiresGrad(std::ostream & out) {
  auto graph = std::make_shared<Graph>();
  at::ScalarType s = at::ScalarType::Float;
  auto type = std::shared_ptr<TensorType>(new TensorType(s, -1, {2, 3, 4}, {12, 4, 1}));

  // Build up a fake graph
  auto a = SymbolicVariable::asNewInput(*graph, type);
  auto b = SymbolicVariable::asNewInput(*graph, type);
  auto d = b * b + b;
  auto e = (d + a) * a + b;
  graph->registerOutput(d.value());
  graph->registerOutput(e.value());

  auto grad_spec = differentiate(graph, {true, false});
  std::vector<Capture> expected_captures = {
    {Capture::Kind::Input, 0},
    {Capture::Kind::Output, 2},
  };
  std::vector<std::size_t> expected_input_vjps = {1, 2};  // for e and %4 = (d + a)
  std::vector<std::size_t> expected_output_vjps = {0};    // only a requires grad
  JIT_ASSERT(grad_spec.f_real_outputs == 2);              // we need one temporary %4 = (d + a)
  JIT_ASSERT(grad_spec.df_input_captures == expected_captures);
  JIT_ASSERT(grad_spec.df_input_vjps == expected_input_vjps);
  JIT_ASSERT(grad_spec.df_output_vjps == expected_output_vjps);
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
  return autograd::make_variable(t.rand(sizes), requires_grad);
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

bool isEqual(const TensorInfo & ti, const autograd::Variable & v) {
  if(!ti.defined())
    return ti.defined() == v.defined();
  return
    ti.device() == device(v) &&
    ti.requires_grad() == v.requires_grad() &&
    ti.type() == v.type().scalarType() &&
    isEqual(ti.sizes(), v.sizes()) &&
    isEqual(ti.strides(), v.strides());
}

void argumentSpecTest() {
  auto & CF = at::CPU(at::kFloat);
  auto & CD = at::CPU(at::kDouble);
  auto & GF = at::CUDA(at::kFloat);
  auto & GD = at::CUDA(at::kDouble);

  autograd::variable_list list =  { var(CF, {1}, true), var(CD, {1, 2}, false) , var(GF, {}, true), var(GD, {4,5,6}, false), undef()};
  // make sure we have some non-standard strides
  list[1].transpose_(0, 1);

  // same list but different backing values
  autograd::variable_list list2 = { var(CF, {1}, true), var(CD, {1, 2}, false) , var(GF, {}, true), var(GD, {4,5,6}, false), undef()};
  list2[1].transpose_(0, 1);


  ArgumentSpec a(true, list);
  ArgumentSpec b(true, list);
  JIT_ASSERT(a.hashCode() == b.hashCode());

  JIT_ASSERT(a == b);
  ArgumentSpec d(true, list2);
  JIT_ASSERT(d == a && d.hashCode() == a.hashCode());

  for(size_t i = 0; i < list.size(); ++i) {
    JIT_ASSERT(isEqual(a.tensorInfo(i), list[i]));
  }
  ArgumentSpec no_grad(/*with_grad=*/false, list);
  JIT_ASSERT(no_grad != a);

  std::unordered_set<ArgumentSpec> spec;
  spec.insert(std::move(a));
  JIT_ASSERT(spec.count(b) > 0);
  JIT_ASSERT(spec.count(no_grad) == 0);
  spec.insert(std::move(no_grad));
  JIT_ASSERT(spec.count(ArgumentSpec(true,list)) == 1);

  list2[1].transpose_(0,1);
  ArgumentSpec c(true, list2); // same as list, except for one stride
  JIT_ASSERT(!(c == a));
  JIT_ASSERT(spec.count(c) == 0);

}

void shapeAnalysisTest() {

  constexpr int batch_size = 4;
  constexpr int input_size = 256;

  int hidden_size = 2*input_size;

  auto v = [](at::Tensor t) { return autograd::make_variable(t, false); };

  auto input = at::CUDA(at::kFloat).randn({batch_size, input_size});
  auto hx    = at::CUDA(at::kFloat).randn({batch_size, hidden_size});
  auto cx    = at::CUDA(at::kFloat).randn({batch_size, hidden_size});
  auto w_ih  = t_def(at::CUDA(at::kFloat).randn({4 * hidden_size, input_size}));
  auto w_hh  = t_def(at::CUDA(at::kFloat).randn({4 * hidden_size, hidden_size}));

  auto g = build_lstm();
  ArgumentSpec spec(false, {v(input), v(hx), v(cx), v(w_ih), v(w_hh) });
  PropagateInputShapes(*g, spec);
  at::Tensor r0, r1;
  std::tie(r0, r1) = lstm(input, hx, cx, w_ih, w_hh);
  auto o0 = g->outputs()[0]->type()->expect<TensorType>();
  auto o1 = g->outputs()[1]->type()->expect<TensorType>();
  JIT_ASSERT(o0->sizes() == std::vector<int64_t>(r0.sizes().begin(), r0.sizes().end()));
  JIT_ASSERT(o1->sizes() == std::vector<int64_t>(r1.sizes().begin(), r1.sizes().end()));

}

std::string runJITCPPTests() {
  std::stringstream out;
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
  argumentSpecTest();
  shapeAnalysisTest();
  return out.str();
}

}}
