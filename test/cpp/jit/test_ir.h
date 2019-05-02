#pragma once

#include "test/cpp/jit/test_base.h"
#include "test/cpp/jit/test_utils.h"

namespace torch {
namespace jit {
namespace test {

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

void testBlocks(std::ostream& out = std::cout) {
  auto g = std::make_shared<Graph>();
  // auto g = *graph;
  auto a = Var::asNewInput(*g, "a");
  auto b = Var::asNewInput(*g, "b");
  auto c = a + b;
  auto r =
      g->appendNode(g->create(prim::If, {Var::asNewInput(*g, "c").value()}));
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
  g->registerOutput((Var(r->output()) + c).value());
  g->lint();
  testing::FileCheck()
      .check("add")
      ->check("prim::If")
      ->check("block0")
      ->check("aten::add")
      ->check("block1")
      ->check_count("aten::add", 3)
      ->run(*g);
  r->eraseBlock(0);
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

} // namespace test
} // namespace jit
} // namespace torch
