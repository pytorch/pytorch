#include <Python.h>
#include <iostream>
#ifdef WITH_CUDA
#include "torch/csrc/jit/fusion_compiler.h"
#endif
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/assert.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/attributes.h"
#include "torch/csrc/jit/interned_strings.h"

namespace torch { namespace jit {

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

#ifdef WITH_CUDA
template<typename T>
Node * appendNewNode(Graph& graph, ArrayRef<Node*> inputs) {
  return graph.appendNewNode<T>(inputs);
}

static void fusionTests() {
  FusionCompiler comp;
  cudaFree(0);

  auto testSimple = [&] {
    Graph graph;
    Node * i0 = graph.addInput();
    Node * i1 = graph.addInput();
    auto o0 = appendNewNode<Mul>(graph,{i0, i1});
    graph.registerOutput(o0);
    auto a = at::CUDA(at::kFloat).rand({3,4});
    auto b = at::CUDA(at::kFloat).rand({4,3}).transpose(0,1);
    auto o = at::CUDA(at::kFloat).zeros({3,4});
    comp.debugLaunchGraph(graph, {a,b}, {o});
    auto o2 = a*b;
    float max_diff = (o2 - o).abs().max().toDouble();
    //std::cout << "max diff: " << max_diff << "\n";
    JIT_ASSERT(max_diff == 0);
  };
  testSimple();

  auto testOne = [&](int ti, int tj, int toi, int toj) {

    Graph graph;

    Node * i0 = graph.addInput();
    Node * i1 = graph.addInput();
    Node * i2 = graph.addInput();
    Node * i3 = graph.addInput();
    Node * i4 = graph.addInput();

    auto p22 = appendNewNode<Sigmoid>(graph,{i4});
    auto p20 = appendNewNode<Sigmoid>(graph,{i3});
    auto p18 = appendNewNode<Tanh>(graph,{i2});
    auto p16 = appendNewNode<Sigmoid>(graph,{i1});
    auto p14 = appendNewNode<Mul>(graph,{p20, i0});
    auto p11 = appendNewNode<Mul>(graph,{p22, p18});
    auto o1 = appendNewNode<Add>(graph,{p14, p11});
    auto p5 = appendNewNode<Tanh>(graph,{o1});
    auto o0 = appendNewNode<Mul>(graph,{p16, p5});

    graph.registerOutput(o0);
    graph.registerOutput(o1);

    graph.lint();

    std::vector<at::Tensor> inputs;
    std::vector<at::Tensor> outputs;
    for(size_t i = 0; i < graph.inputs().size(); i++) {
      inputs.push_back(at::CUDA(at::kFloat).rand({128,128,32}).transpose(ti, tj));
    }
    for(size_t i = 0; i < graph.outputs().size(); i++) {
      outputs.push_back(at::CUDA(at::kFloat).zeros({128,128,32}).transpose(toi,toj));
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
    comp.debugLaunchGraph(graph, inputs, outputs);
    float max_diff = (outputs.front() - out0).abs().max().toDouble();
    //std::cout << "max diff: " << max_diff << "\n";
    JIT_ASSERT(max_diff < 1e-6);

  };
  testOne(0,0,0,0);
  testOne(0,1,0,0);
  testOne(1,2,0,0);
  testOne(0,2,0,0);

  testOne(0,0,0,1);
  testOne(0,1,1,2);
  testOne(1,2,0,2);

}
#else //WITH_CUDA
void fusionTests() {}
#endif

void attributesTest() {
  auto one = kParam;
  auto two = kReturn;
  auto three = kConstant;
  Attributes attr;
  attr.f_(one,3.4).i_(two,5).s_(three,"what");
  assert(attr.f(one) == 3.4);
  assert(attr.s(three) == "what");
  assert(attr.i(two) == 5);
  attr.s_(one,"no");
  assert(attr.s(one) == "no");
  assert(attr.hasAttribute(three));
  assert(!attr.hasAttribute(four));
  attr.ss_(two, {"hi", "now"});
  assert(attr.ss(two).at(1) == "now");

  Attributes attr2 = attr;
  assert(two.s(one) == "no");
  attr2.f_(one,5);
  assert(attr.s(one) == "no");
  assert(attr2.f(one) == 5);
}

void internedStringsTests () {

  assert(kParam == stringToSymbol("Param"));
  assert(kReturn == stringToSymbol("Return"));
  assert(symbolToString(kReturn) == "Return");
  assert(stringToSymbol("What") == kLastSymbol);
  assert(stringToSymbol("What2") == kLastSymbol+1);
  assert(stringToSymbol("What") == kLastSymbol);
  assert(stringToSymbol("What2") == kLastSymbol+1);
  assert(symbolToString(kLastSymbol+1) == "What2");
}


void runJITCPPTests() {
  codeTemplateTest();
  fusionTests();
  attributesTest();
  internedStringsTests();
}

}}
