#include "test/cpp/jit/test_base.h"
#include "test/cpp/jit/test_utils.h"
#include "torch/csrc/jit/ir/irparser.h"

namespace torch {
namespace jit {

void testAttributes() {
  Graph g;
  auto one = attr::alpha;
  auto two = attr::device;
  auto three = attr::end;
  auto four = attr::perm;
  Node* n = g.create(Symbol::fromQualString("foo::bar"));
  Node& attr = *n;
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

  Node* n2 = g.create(Symbol::fromQualString("foo::baz"));
  Node& attr2 = *n2;
  attr2.copyAttributes(attr);
  ASSERT_EQ(attr2.s(one), "no");
  attr2.f_(one, 5);
  ASSERT_EQ(attr.s(one), "no");
  ASSERT_EQ(attr2.f(one), 5);
}

void testBlocks() {
  auto g = std::make_shared<Graph>();
  const auto graph_string = R"IR(
    graph(%a : Tensor,
          %b : Tensor,
          %c : Tensor):
      %2 : int = prim::Constant[value=1]()
      %3 : Tensor = aten::add(%a, %b, %2)
      %5 : Tensor = prim::If(%c)
        block0():
          %6 : int = prim::Constant[value=1]()
          %7 : Tensor = aten::add(%3, %3, %6)
          -> (%7)
        block1():
          %8 : int = prim::Constant[value=1]()
          %9 : Tensor = aten::add(%b, %3, %8)
          %10 : int = prim::Constant[value=1]()
          %11 : Tensor = aten::add(%9, %3, %10)
          -> (%11)
      %12 : int = prim::Constant[value=1]()
      %13 : Tensor = aten::add(%5, %3, %12)
      return (%13))IR";
  torch::jit::parseIR(graph_string, g.get());

  g->lint();
  testing::FileCheck()
      .check("add")
      ->check("prim::If")
      ->check("block0")
      ->check("aten::add")
      ->check("block1")
      ->check_count("aten::add", 3)
      ->run(*g);

  // Removes block0 of the conditional
  for (auto* node : g->block()->nodes()) {
    if (node->kind() == prim::If) {
      node->eraseBlock(0);
      break;
    }
  }

  testing::FileCheck()
      .check("add")
      ->check("prim::If")
      ->check("block0")
      ->check_not("block")
      ->run(*g);
  g->lint();
  // test recursive copy of blocks works
  auto g2 = g->copy();
  testing::FileCheck()
      .check("add")
      ->check("prim::If")
      ->check("block0")
      ->check_not("block")
      ->run(*g2);
}

void testCommonAncestor() {
  std::string input_str = R"(
graph(%x : Tensor,
      %a.1 : bool,
      %b.1 : bool,
      %c.1 : bool):
  %4 : int = prim::If(%a.1)
    block0():
      %5 : int = prim::If(%b.1)
        block0():
          %6 : int = prim::Constant[value=2]()
          -> (%6)
        block1():
          %7 : int = prim::Constant[value=3]()
          -> (%7)
      -> (%5)
    block1():
      %8 : int = prim::If(%c.1)
        block0():
          %9 : int = prim::Constant[value=4]()
          -> (%9)
        block1():
          %10 : int = prim::Constant[value=5]()
          -> (%10)
      -> (%8)
  return (%4)
)";

  torch::jit::Graph g;
  std::unordered_map<std::string, torch::jit::Value*> name_to_value;
  torch::jit::parseIR(input_str, &g, name_to_value);

  std::vector<std::string> value_names{"6", "7", "9", "10"};
  std::unordered_set<std::string> value_names_set(
      value_names.begin(), value_names.end());

  /* clang-format off */
  int ref_blocks_from_graph[4][4] = {
    /* (6, 6), (6, 7), (6, 9), (6, 10) */
    {   2,     1,      0,      0        },
    /* (7, 6), (7, 7), (7, 9), (7, 10) */
    {   1,     2,      0,      0        },
    /* (9, 6), (9, 7), (9, 9), (9, 10) */
    {   0,     0,      2,      1,       },
    /* (10, 6),(10, 7),(10, 9),(10, 10) */
    {   0,     0,      1,      2        }
  };
  /* clang-format on */

  for (size_t i = 0; i < value_names.size(); ++i) {
    Value* i_val = name_to_value[value_names[i]];
    for (size_t j = 0; j < value_names.size(); ++j) {
      Value* j_val = name_to_value[value_names[j]];
      Block* common_ancestor =
          i_val->node()->findCommonAncestorBlockWith(j_val->node());
      int blocks_from_graph_block =
          common_ancestor->param_node()->blocksFromGraphBlock();
      ASSERT_EQ(blocks_from_graph_block, ref_blocks_from_graph[i][j]);
    }
  }
}

} // namespace jit
} // namespace torch
