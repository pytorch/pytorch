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

#include "torch/csrc/assertions.h"
#include "torch/csrc/utils/auto_gil.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/python_engine.h"
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
    differentiate(graph);
    Code bytecode {graph};
    InterpreterState interpreter {bytecode};

    // Get outputs from the interpreter
    auto tensors_in                = fmap(vars_in, unwrap);
    auto tensor_grads_in           = fmap(var_grads_in, unwrap);
    tensor_list tensors_out, tensor_grads_out;
    interpreter.runOneStage(tensors_in, tensors_out);
    interpreter.runOneStage(tensor_grads_in, tensor_grads_out);

    // Compare results
    auto expected_tensors_out      = fmap(vars_out, unwrap);
    auto expected_tensor_grads_out = fmap(var_grads_out, unwrap);
    assertAllClose(tensors_out,      expected_tensors_out);
    assertAllClose(tensor_grads_out, expected_tensor_grads_out);
  }
}

void testCreateAutodiffSubgraphs(std::ostream & out) {
  auto graph = build_lstm();
  CreateAutodiffSubgraphs(*graph, /*threshold=*/2);
  out << "testCreateAutodiffSubgraphs\n";
  out << *graph << "\n";
}

std::string runJITCPPTests() {
  std::stringstream out;
  testCreateAutodiffSubgraphs(out);
  testADFormulas();
  interpTest();
  interpStageTest();
  codeTemplateTest();
  fusionTests();
  attributesTest();
  internedStringsTests();
  return out.str();
}

}}
