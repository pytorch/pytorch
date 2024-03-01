#include <gtest/gtest.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/utils/memory_dag.h>

namespace torch {
namespace jit {

TEST(MemoryDAGTest, Basic) {
  auto graph = std::make_shared<Graph>();
  const Value* aValue = graph->addInput();
  const Value* bValue = graph->addInput();
  const Value* cValue = graph->addInput();
  const Value* dValue = graph->addInput();
  const Value* eValue = graph->addInput();
  const Value* fValue = graph->addInput();
  const Value* gValue = graph->addInput();

  {
    // a <- b <- c
    //      b <- d
    // a <- e
    // f <- e
    // g is by itself
    auto t = std::make_unique<MemoryDAGBuilder>();
    auto a = t->makeFreshValue(aValue);
    auto b = t->makeFreshValue(bValue);
    auto c = t->makeFreshValue(cValue);
    auto d = t->makeFreshValue(dValue);
    auto e = t->makeFreshValue(eValue);
    auto f = t->makeFreshValue(fValue);
    auto g = t->makeFreshValue(gValue);
    t->makePointerTo(b, a);
    t->makePointerTo(c, b);
    t->makePointerTo(d, b);
    t->makePointerTo(e, a);
    t->makePointerTo(e, f);

    auto dag = std::move(*t).createMemoryDAG();

    /**
     * Test mayAlias()
     */
    // Values should alias themselves
    EXPECT_TRUE(dag->mayAlias(a, a));
    EXPECT_TRUE(dag->mayAlias(g, g));

    // Values that point to the same location should alias
    EXPECT_TRUE(dag->mayAlias(a, b));
    EXPECT_TRUE(dag->mayAlias(a, c));
    EXPECT_TRUE(dag->mayAlias(c, d));

    // e may point to a OR f
    EXPECT_TRUE(dag->mayAlias(e, a));
    EXPECT_TRUE(dag->mayAlias(e, f));
    // But a and f don't alias
    EXPECT_FALSE(dag->mayAlias(a, f));
  }
  {
    // x(y) -> x contains y

    // b(a)
    // c(a)
    auto t = std::make_unique<MemoryDAGBuilder>();
    auto a = t->makeFreshValue(aValue);
    auto b = t->makeFreshValue(bValue);
    t->addToContainedElements(a, b);

    auto c = t->makeFreshValue(cValue);
    t->addToContainedElements(a, c);

    auto dag = std::move(*t).createMemoryDAG();
    EXPECT_TRUE(dag->mayContainAlias(a, b));
    EXPECT_TRUE(dag->mayContainAlias(b, a));

    EXPECT_TRUE(dag->mayContainAlias(a, c));
    EXPECT_TRUE(dag->mayContainAlias(c, a));

    EXPECT_TRUE(dag->mayContainAlias(b, c));
    EXPECT_TRUE(dag->mayContainAlias(c, b));

    // containers contain an element in themselves
    EXPECT_TRUE(dag->mayContainAlias(b, b));
    EXPECT_TRUE(dag->mayContainAlias(c, c));
    EXPECT_TRUE(dag->mayContainAlias(a, a));
  }
  {
    // b(a)
    // c(a)
    // d(b(a))
    auto t = std::make_unique<MemoryDAGBuilder>();
    auto a = t->makeFreshValue(aValue);
    auto b = t->makeFreshValue(bValue);
    t->addToContainedElements(a, b);

    auto c = t->makeFreshValue(cValue);
    t->addToContainedElements(a, c);

    auto d = t->makeFreshValue(dValue);
    t->addToContainedElements(b, d);

    auto dag = std::move(*t).createMemoryDAG();
    EXPECT_TRUE(dag->mayContainAlias(b, d));
    EXPECT_TRUE(dag->mayContainAlias(d, b));

    EXPECT_TRUE(dag->mayContainAlias(c, d));
    EXPECT_TRUE(dag->mayContainAlias(d, c));

    EXPECT_TRUE(dag->mayContainAlias(a, d));
  }
  {
    // f(e)
    auto t = std::make_unique<MemoryDAGBuilder>();
    auto a = t->makeFreshValue(aValue);
    auto b = t->makeFreshValue(bValue);
    t->addToContainedElements(a, b);

    auto c = t->makeFreshValue(cValue);
    t->addToContainedElements(a, c);

    auto d = t->makeFreshValue(dValue);
    t->addToContainedElements(b, d);

    auto f = t->makeFreshValue(aValue);
    auto e = t->makeFreshValue(bValue);

    t->addToContainedElements(f, e);

    auto dag = std::move(*t).createMemoryDAG();
    for (auto elem : {a, b, c, d}) {
      EXPECT_FALSE(dag->mayContainAlias(f, elem));
      EXPECT_FALSE(dag->mayContainAlias(e, elem));
    }
  }
}

} // namespace jit
} // namespace torch
