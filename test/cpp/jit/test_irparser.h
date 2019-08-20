#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/irparser.h>
#include <torch/csrc/jit/testing/file_check.h>
#include "test/cpp/jit/test_base.h"

#include <sstream>
#include <string>

namespace torch {
namespace jit {

/** \brief Parse IR from \p S, print the parsed graph and verify that the output
 * string matches the original string.
 *
 * The function is sensitive to value naming and whitespace, so it should be
 * used with care. Nevertheless, it helps to keep tests more compact.
 */
static void checkRoundtrip(const std::string& s) {
  auto graph = std::make_shared<Graph>();
  script::parseIR(s, &*graph);
  std::ostringstream ss;
  ss << *graph;
  std::string parsed = ss.str();

  // Skip whitespace in the beginning of the input string.
  int i = 0;
  for (char c : s) {
    if (!isspace(c)) {
      break;
    }
    i++;
  }
  std::string original = s.substr(i, s.size());
  if (original != parsed) {
    std::cerr << "Input:" << std::endl << original << std::endl;
    std::cerr << "Parsed:" << std::endl << parsed << std::endl;
  }
  AT_ASSERT(original == parsed);
}

void testIRParser() {
  {
    auto graph = std::make_shared<Graph>();
    std::unordered_map<std::string, Value*> vmap;
    script::parseIR(
        R"IR(
graph(%0 : Tensor, %1 : Tensor):
  %2 : Tensor = foo::add(%0, %1)
  %res, %3 = foo::mul(%0, %2)
  %x, %y = foo::combine(%res, %2, %3)
  return (%x, %y, %res))IR",
        &*graph,
        vmap);

    AT_ASSERT(graph->inputs().size() == 2);
    AT_ASSERT(graph->outputs().size() == 3);
    Value* x = graph->outputs()[0];
    Value* y = graph->outputs()[1];
    Value* res = graph->outputs()[2];
    Value* t0 = graph->inputs()[0];
    Value* t1 = graph->inputs()[1];
    AT_ASSERT(vmap["x"] == x);
    AT_ASSERT(vmap["y"] == y);
    AT_ASSERT(vmap["res"] == res);
    AT_ASSERT(vmap["0"] == t0);
    AT_ASSERT(vmap["1"] == t1);
    AT_ASSERT(x->node() == y->node());
    Node* comb = x->node();
    Value* t2 = comb->inputs()[1];
    Value* t3 = comb->inputs()[2];
    AT_ASSERT(vmap["2"] == t2);
    AT_ASSERT(vmap["3"] == t3);
    AT_ASSERT(comb->kind().toQualString() == std::string("foo::combine"));
    AT_ASSERT(comb->outputs() == std::vector<Value*>({x, y}));
    AT_ASSERT(comb->inputs() == std::vector<Value*>({res, t2, t3}));
    Node* mul = res->node();
    AT_ASSERT(mul->kind().toQualString() == std::string("foo::mul"));
    AT_ASSERT(mul->inputs() == std::vector<Value*>({t0, t2}));
    AT_ASSERT(mul->outputs() == std::vector<Value*>({res, t3}));
    Node* add = t2->node();
    AT_ASSERT(add->kind().toQualString() == std::string("foo::add"));
    AT_ASSERT(add->inputs() == std::vector<Value*>({t0, t1}));
    AT_ASSERT(add->outputs() == std::vector<Value*>({t2}));
  }
  {
    checkRoundtrip(R"IR(
graph():
  %0 : Tensor = a::a()
    block0():
      %1 : Tensor = b::b()
        block0():
          %2 : Tensor = c::c()
          -> ()
      -> ()
  %3 : Tensor = d::d()
  return (%3)
)IR");
  }
  {
    checkRoundtrip(R"IR(
graph(%0 : Tensor,
      %1 : Tensor,
      %2 : Tensor):
  %3 : int = prim::Constant[value=1]()
  %4 : Tensor = aten::add(%0, %1, %3)
  %5 : Tensor = prim::If(%2)
    block0():
      %6 : int = prim::Constant[value=1]()
      %7 : Tensor = aten::add(%1, %3, %6)
      %8 : int = prim::Constant[value=1]()
      %9 : Tensor = aten::add(%7, %3, %8)
      -> (%9)
  %10 : int = prim::Constant[value=1]()
  %11 : Tensor = aten::add(%5, %3, %10)
  return (%11)
)IR");
  }
  {
    auto graph = std::make_shared<Graph>();
    script::parseIR(
        R"IR(
graph(%a):
  return (%a))IR",
        &*graph);
    AT_ASSERT(graph->inputs()[0]->type()->isSubtypeOf(TensorType::get()));
  }
  {
    // Check that parser corectly handles values reusing the same name.
    auto graph = std::make_shared<Graph>();
    script::parseIR(
        R"IR(
graph(%x):
  %x = a::a(%x)
  %x = b::b(%x)
  return (%x))IR",
        &*graph);
    Value* x0 = graph->inputs()[0];
    Value* x2 = graph->outputs()[0];
    Node* b = x2->node();
    Value* x1 = b->inputs()[0];
    Node* a = x1->node();
    AT_ASSERT(a->inputs() == std::vector<Value*>({x0}));
    AT_ASSERT(a->outputs() == std::vector<Value*>({x1}));
    AT_ASSERT(b->inputs() == std::vector<Value*>({x1}));
    AT_ASSERT(b->outputs() == std::vector<Value*>({x2}));
  }
  {
    // Check that parser handles attributes and types.
    checkRoundtrip(
        R"IR(
graph(%0 : Tensor,
      %1 : Tensor,
      %2 : Tensor):
  %3 : int, %4 : Tensor = qqq::qqq[i_asdf=2, f_asdf=3.14, s_asdf="hello", ss_asdf=["hello world", "bye bye"]](%0)
  %5 : int, %6 : Tensor = ppp::ppp[i_asdf=2, f_asdf=3.14, s_asdf="\"\"\"\"\nhe\"llo", q=[3, 2, 4]](%0)
  %7 : float = vvv::vvv[s_asdf="hello"](%0)
  %8 : string = z::z()
  return (%7)
)IR");
  }

  {
    checkRoundtrip(
        R"IR(
graph(%0 : Tensor,
      %1 : Tensor,
      %2 : Tensor):
  %3 : int? = prim::Constant()
  return (%3)
)IR");
  }

  {
    checkRoundtrip(
        R"IR(
graph(%0 : Tensor,
      %1 : Tensor,
      %2 : Tensor):
  %3 : Float(*, *, *) = prim::Constant()
  return (%3)
)IR");
  }

  {
    checkRoundtrip(
        R"IR(
graph(%0 : Tensor,
      %1 : Tensor,
      %2 : Tensor):
  %3 : Long() = prim::Constant()
  return (%3)
)IR");
  }

  {
    checkRoundtrip(
        R"IR(
graph(%0 : Tensor,
      %1 : Tensor,
      %2 : Tensor):
  %3 : Double(4, 4, 5) = prim::Constant()
  return (%3)
)IR");
  }

  {
    bool error_thrown = false;
    try {
      checkRoundtrip(
          R"IR(
graph(%0 : Tensor,
    %1 : Tensor,
    %2 : Tensor):
  %3 : Double(4!, 4, 5) = prim::Constant()
  return (%3)
)IR");
    } catch (const std::exception& error) {
      error_thrown = true;
    }
    AT_ASSERT(error_thrown);
  }

  {
    auto graph = std::make_shared<Graph>();
    const std::string& text =
        R"IR(
    graph(%a):
    # CHECK: return
      return (%a))IR";

    script::parseIR(text, &*graph);
    AT_ASSERT(graph->inputs()[0]->type()->isSubtypeOf(TensorType::get()));
    torch::jit::testing::FileCheck().run(text, *graph);
  }
}
} // namespace jit
} // namespace torch
