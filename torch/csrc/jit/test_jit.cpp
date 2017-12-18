#include <Python.h>
#include <iostream>
#include "torch/csrc/jit/fusion_compiler.h"
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/assertions.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/attributes.h"
#include "torch/csrc/jit/interned_strings.h"
#include <vector>
#include "torch/csrc/jit/interpreter.h"

namespace torch { namespace jit {

// help build Graphs for tests
struct Var {
  Var() : v(nullptr) {}
  Var(Value * v) : v(v) {}
  static Var Input(Graph & g, std::string name = "") {
    return g.addInput(name);
  }
  void addAsOutput() {
    v->owningGraph()->registerOutput(v);
  }
  static Var cat(ArrayRef<Var> inputs, int32_t dim) {
    Node* n;
    auto r = create(kcat, inputs, 1, &n)[0];
    n->i_(kdim, dim);
    return r;
  }

  static std::vector<Var> create(Symbol kind, ArrayRef<Var> inputs,
                                 int num_outputs = 1,
                                 Node** created_node = nullptr,
                                 Graph * g = nullptr) {
      if(g == nullptr) {
        g = inputs.at(0).value()->owningGraph();
      }
      Node * n = g->appendNode(g->create(kind, num_outputs));
      for(auto i : inputs) {
        n->addInput(i.value());
      }
      if(created_node) {
        *created_node = n;
      }
      std::vector<Var> out;
      for(auto v : n->outputs()) {
        out.emplace_back(v);
      }
      return out;
  }
  Var operator*(Var rhs) {
    return create(kmul, {*this, rhs})[0];
  }
  Var operator+(Var rhs) {
    Node * n;
    auto r = create(kadd, {*this, rhs}, 1, &n)[0];
    n->t_(kalpha, at::Scalar(1).toTensor());
    return r;
  }

  Var mm(Var rhs) {
    return create(s("mm"), {*this, rhs})[0];
  }
  Var sigmoid() {
    return create(ksigmoid, {*this})[0];
  }
  Var tanh() {
    return create(ktanh, {*this})[0];
  }
  std::vector<Var> chunk(int32_t chunks, uint32_t dim) {
    Node * n;
    auto r = create(s("chunk"), { *this }, chunks, &n);
    n->i_(s("chunks"), chunks)
    ->i_(s("dim"), dim);
    return r;
  }
  Value * value() const {
    return v;
  }
private:
  static Symbol s(const char * s_) {
    return stringToSymbol(s_);
  }
  Value * v;
};

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
    Var i0 = Var::Input(graph);
    Var i1 = Var::Input(graph);
    auto o0 = i0 * i1;
    o0.addAsOutput();
    auto a = at::CUDA(at::kFloat).rand({3,4});
    auto b = at::CUDA(at::kFloat).rand({4,3}).transpose(0,1);
    auto o = at::CUDA(at::kFloat).zeros({3,4});
    comp.debugLaunchGraph(graph, true, {a,b}, {o});
    auto o2 = a*b;
    float max_diff = (o2 - o).abs().max().toCDouble();
    //std::cout << "max diff: " << max_diff << "\n";
    JIT_ASSERT(max_diff == 0);
  };
  testSimple();

  auto testOne = [&](int ti, int tj, int toi, int toj) {

    Graph graph;

    Var i0 = Var::Input(graph);
    Var i1 = Var::Input(graph);
    Var i2 = Var::Input(graph);
    Var i3 = Var::Input(graph);
    Var i4 = Var::Input(graph);

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
    comp.debugLaunchGraph(graph, true, inputs, outputs);
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
    Var i0 = Var::Input(graph);
    Var i1 = Var::Input(graph);
    auto o0 = i0 * i1;
    o0.addAsOutput();
    Var::cat({i0, o0}, dim).addAsOutput();

    auto a = at::CUDA(at::kFloat).rand({3,4,5});
    auto b = at::CUDA(at::kFloat).rand({4,3,5}).transpose(0,1);
    auto o = at::CUDA(at::kFloat).zeros({3,4,5});

    auto o_r = a*b;
    auto o2_r = at::cat({a, o_r}, dim);
    auto o2 = at::CUDA(at::kFloat).zeros(o2_r.sizes());
    comp.debugLaunchGraph(graph, true, {a,b}, {o, o2});

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

  JIT_ASSERT(kParam == stringToSymbol("Param"));
  JIT_ASSERT(kReturn == stringToSymbol("Return"));
  JIT_ASSERT(symbolToString(kReturn) == std::string("Return"));
  size_t symstart = stringToSymbol("__NEW_SYMBOL");
  JIT_ASSERT(stringToSymbol("What") == symstart+1);
  JIT_ASSERT(stringToSymbol("What2") == symstart+2);
  JIT_ASSERT(stringToSymbol("What") == symstart+1);
  JIT_ASSERT(stringToSymbol("What2") == symstart+2);
  JIT_ASSERT(symbolToString(symstart+2) == std::string("What2"));
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

void runJITCPPTests() {
  interpTest();
  interpStageTest();
  codeTemplateTest();
  fusionTests();
  attributesTest();
  internedStringsTests();
}

}}
