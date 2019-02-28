#pragma once

#include "test/cpp/jit/test_base.h"
#include "torch/csrc/jit/passes/alias_analysis.h"
#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/utils/memory.h"

namespace torch {
namespace jit {

// Fixture to set up a graph and make assertions clearer
struct TopoMoveTestFixture {
  TopoMoveTestFixture() {
    createGraph();
    aliasDb = torch::make_unique<AliasDb>(graph);
  }

  // Nodes are named after their output.
  // e.g. "a" is an alias for "the node that outputs the value `a`"
  void createGraph() {
    graph = std::make_shared<Graph>();
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

    graph->lint();
  }

  void createNode(
      const std::string& name,
      const std::vector<std::string>& inputNames,
      const std::vector<std::string>& blockInputNames = {}) {
    std::vector<Value*> inputs;
    for (const auto name : inputNames) {
      inputs.push_back(nodes.at(name)->output());
    }
    auto node = graph->appendNode(graph->create(prim::Undefined, inputs));
    node->output()->setUniqueName(name);
    nodes[name] = node;

    if (blockInputNames.size() != 0) {
      node->addBlock();
      std::vector<Value*> blockDeps;
      for (const auto name : blockInputNames) {
        blockDeps.push_back(nodes.at(name)->output());
      }

      auto block = node->blocks().at(0);
      block->appendNode(graph->create(prim::Undefined, blockDeps));
    }
  }

  bool moveBeforeTopologicallyValid(
      const std::string& toInsert,
      const std::string& insertPoint) {
    std::function<bool(Node*, Node*)> func =
        [this](Node* toInsert, Node* insertPoint) {
          return aliasDb->moveBeforeTopologicallyValid(toInsert, insertPoint);
        };
    return moveWithChecks(toInsert, insertPoint, func);
  }

  bool moveAfterTopologicallyValid(
      const std::string& toInsert,
      const std::string& insertPoint) {
    std::function<bool(Node*, Node*)> func =
        [this](Node* toInsert, Node* insertPoint) {
          return aliasDb->moveAfterTopologicallyValid(toInsert, insertPoint);
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
    graph->lint();

    // If this is the picture of nodes
    // <some nodes> ... toInsert ... <some more nodes> ... insertPoint
    // ^----------^ check that these nodes haven't moved
    curNode = original;
    size_t idx = 0;
    while (curNode != n->owningBlock()->return_node()) {
      AT_ASSERT(originalOrdering[idx] == curNode);
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
      AT_ASSERT(nodes.at(toInsert)->prev() == nodes.at(insertPoint));
    } else {
      AT_ASSERT(nodes.at(toInsert)->next() == nodes.at(insertPoint));
    }
  }

  std::shared_ptr<Graph> graph;
  std::unique_ptr<AliasDb> aliasDb;
  std::unordered_map<std::string, Node*> nodes;
};

void testTopologicalMove() {
  {
    // Check that we are removing `this`'s deps properly when we need to split
    // `this` and deps (see code for what the hell that means)
    TopoMoveTestFixture fixture;
    AT_ASSERT(fixture.moveBeforeTopologicallyValid("q", "s"));
    fixture.checkPostCondition("q", "s", false);
  }
  // Move after
  {
    // Simple move backward
    TopoMoveTestFixture fixture;
    AT_ASSERT(fixture.moveAfterTopologicallyValid("c", "a"));
    fixture.checkPostCondition("c", "a", true);
  }
  {
    // simple invalid move backward
    TopoMoveTestFixture fixture;
    AT_ASSERT(!fixture.moveAfterTopologicallyValid("d", "a"));
  }
  {
    // doesn't actually move anything
    TopoMoveTestFixture fixture;
    AT_ASSERT(fixture.moveAfterTopologicallyValid("f", "e"));
    fixture.checkPostCondition("f", "e", true);
  }
  {
    // move backward with multiple dependencies
    TopoMoveTestFixture fixture;
    AT_ASSERT(fixture.moveAfterTopologicallyValid("e", "c"));
    fixture.checkPostCondition("e", "c", true);
  }
  {
    // Move backward with non-zero working set
    TopoMoveTestFixture fixture;
    AT_ASSERT(fixture.moveAfterTopologicallyValid("k", "f"));
    fixture.checkPostCondition("k", "f", true);
  }
  {
    // Simple move forward
    TopoMoveTestFixture fixture;
    AT_ASSERT(fixture.moveAfterTopologicallyValid("c", "d"));
    fixture.checkPostCondition("c", "d", true);
  }
  {
    // Move forward with non-zero working set
    TopoMoveTestFixture fixture;
    AT_ASSERT(fixture.moveAfterTopologicallyValid("f", "l"));
    fixture.checkPostCondition("f", "l", true);
  }

  // Move before
  {
    // Simple move forward
    TopoMoveTestFixture fixture;
    AT_ASSERT(fixture.moveBeforeTopologicallyValid("b", "d"));
    fixture.checkPostCondition("b", "d", false);
  }
  {
    // Simple move backward
    TopoMoveTestFixture fixture;
    AT_ASSERT(fixture.moveBeforeTopologicallyValid("c", "a"));
    fixture.checkPostCondition("c", "a", false);
  }
  {
    // doesn't actually move anything
    TopoMoveTestFixture fixture;
    AT_ASSERT(fixture.moveBeforeTopologicallyValid("a", "b"));
    fixture.checkPostCondition("a", "b", false);
  }
  {
    // move forward with deps
    TopoMoveTestFixture fixture;
    AT_ASSERT(fixture.moveBeforeTopologicallyValid("f", "m"));
    fixture.checkPostCondition("f", "m", false);
  }
  {
    // move backward with deps
    TopoMoveTestFixture fixture;
    AT_ASSERT(fixture.moveBeforeTopologicallyValid("l", "f"));
    fixture.checkPostCondition("l", "f", false);
  }

  // check that dependencies in blocks are recognized
  {
    TopoMoveTestFixture fixture;
    AT_ASSERT(!fixture.moveAfterTopologicallyValid("l", "m"));
    AT_ASSERT(!fixture.moveBeforeTopologicallyValid("m", "l"));
    AT_ASSERT(!fixture.moveAfterTopologicallyValid("n", "l"));
    AT_ASSERT(!fixture.moveBeforeTopologicallyValid("l", "n"));
  }

  // Test that moveAfter(n) and moveBefore(n->next()) are not necessarily
  // equivalent. Here, the dependency ordering is n -> o -> p.  So we can't
  // move `n` after `o`, but we can move `n` before `p` (which pushes `o` after
  // `p`)
  {
    TopoMoveTestFixture fixture;
    AT_ASSERT(!fixture.moveAfterTopologicallyValid("n", "o"));
    AT_ASSERT(fixture.moveBeforeTopologicallyValid("o", "p"));
    fixture.checkPostCondition("o", "p", false);
  }
}

namespace {
Node* insertIf(
    Graph& g,
    Value* condValue,
    std::function<std::vector<Value*>()> trueInst,
    std::function<std::vector<Value*>()> falseInst) {
  auto if_ = g.insertNode(g.create(prim::If, 0));
  if_->addInput(condValue); // condition value
  auto trueBlock = if_->addBlock();
  auto falseBlock = if_->addBlock();
  {
    // Mutate in true block
    WithInsertPoint g(trueBlock);
    auto outputs = trueInst();
    for (auto output : outputs) {
      trueBlock->registerOutput(output);
    }
  }
  {
    WithInsertPoint g(falseBlock);
    auto outputs = falseInst();
    for (auto output : outputs) {
      falseBlock->registerOutput(output);
    }
  }

  AT_ASSERT(trueBlock->outputs().size() == falseBlock->outputs().size());
  for (auto output : trueBlock->outputs()) {
    if_->addOutput()->setType(output->type());
  }
  return if_;
}
} // namespace

void testAliasAnalysis() {
  {
    auto graph = std::make_shared<Graph>();
    auto a = graph->addInput();
    auto b = graph->addInput();

    // addsB = b + b
    // c = a + b
    // a += b
    // d = c + c
    auto addsB = graph->insert(aten::add, {b, b});
    auto c = graph->insert(aten::add, {a, b});
    auto aMut = graph->insert(aten::add_, {a, b});
    auto d = graph->insert(aten::add, {c, c});

    graph->lint();

    AliasDb aliasDb(graph);
    // Can't move past a mutation of a used value
    AT_ASSERT(!aliasDb.moveAfterTopologicallyValid(c->node(), aMut->node()));
    AT_ASSERT(aliasDb.moveAfterTopologicallyValid(d->node(), c->node()));

    // b should alias to a (since they are both inputs)
    AT_ASSERT(
        !aliasDb.moveAfterTopologicallyValid(addsB->node(), aMut->node()));
    AT_ASSERT(aliasDb.moveAfterTopologicallyValid(addsB->node(), c->node()));

    graph->lint();
  }
  {
    auto graph = std::make_shared<Graph>();
    auto a = graph->addInput();
    auto b = graph->addInput();

    auto constant = graph->insertConstant(1);
    auto fresh = graph->insert(aten::rand, {constant});
    auto usesB = graph->insert(aten::add, {b, fresh});
    auto aliasesB = graph->insert(aten::select, {a, constant, constant});
    auto mutatesAliasOfB = graph->insert(aten::add_, {aliasesB, fresh});
    graph->insert(aten::add, {fresh, aliasesB});
    graph->lint();

    AliasDb aliasDb(graph);
    AT_ASSERT(!aliasDb.moveAfterTopologicallyValid(
        aliasesB->node(), mutatesAliasOfB->node()));
    AT_ASSERT(!aliasDb.moveAfterTopologicallyValid(
        usesB->node(), mutatesAliasOfB->node()));
  }
  {
    // Test moves across inner blocks

    // a = rand(1)
    // b = rand(1)
    // if True:
    //   a.add_(b)
    // c = a + b
    auto graph = std::make_shared<Graph>();
    auto constant = graph->insertConstant(1);
    auto a = graph->insert(aten::rand, {constant});
    auto b = graph->insert(aten::rand, {constant});

    auto if_ = insertIf(
        *graph,
        constant,
        [&]() -> std::vector<Value*> {
          auto aMut = graph->insert(aten::add_, {a, b});
          return {aMut};
        },
        [&]() -> std::vector<Value*> { return {a}; });

    auto c = graph->insert(aten::add, {a, b});

    graph->lint();

    // we should not be able to move `c` before the if statement, since it
    // may write to `a`.
    AliasDb aliasDb(graph);
    ASSERT_FALSE(aliasDb.moveBeforeTopologicallyValid(c->node(), if_));
  }
  {
    // test fork/wait

    // a = rand(1)
    // fut = fork(a)
    //    Subgraph is: return a.add_(1)
    // ... some unrelated code
    // c = wait(b)
    // d = a + a

    auto graph = std::make_shared<Graph>();
    auto constant = graph->insertConstant(1);
    auto a = graph->insert(aten::rand, {constant});

    auto forkNode = graph->insertNode(graph->create(prim::fork));
    auto forkBlock = forkNode->addBlock();
    {
      WithInsertPoint g(forkBlock);
      auto aMut = graph->insert(aten::add_, {a, constant});
      forkBlock->registerOutput(aMut);
      forkNode->output()->setType(FutureType::create(aMut->type()));
    }
    script::lambdaLiftFork(forkNode);

    auto fut = forkNode->output();
    auto wait = graph->insert(aten::wait, {fut})->node();
    auto d = graph->insert(aten::add, {a, a});

    graph->lint();

    // Should not be able to move `d` before the wait call
    AliasDb aliasDb(graph);
    ASSERT_FALSE(aliasDb.moveBeforeTopologicallyValid(d->node(), wait));
  }
  {
    // test fork/wait in an if statement

    // a = rand(1)
    // if 1:
    //   fut = fork(a)
    //     Subgraph is: return a.add_(1)
    // else:
    //   fut = fork(a)
    //     Subgraph is: return a.sub_(1)
    // c = wait(b)
    // d = a + a

    auto graph = std::make_shared<Graph>();
    auto constant = graph->insertConstant(1);
    auto a = graph->insert(aten::rand, {constant});
    auto if_ = insertIf(
        *graph,
        constant,
        [&]() -> std::vector<Value*> {
          auto forkNode = graph->insertNode(graph->create(prim::fork));
          auto forkBlock = forkNode->addBlock();
          {
            WithInsertPoint g(forkBlock);
            auto aMut = graph->insert(aten::add_, {a, constant});
            forkBlock->registerOutput(aMut);
            forkNode->output()->setType(FutureType::create(aMut->type()));
          }
          script::lambdaLiftFork(forkNode);
          return {forkNode->output()};
        },
        [&]() -> std::vector<Value*> {
          auto forkNode = graph->insertNode(graph->create(prim::fork));
          auto forkBlock = forkNode->addBlock();
          {
            WithInsertPoint g(forkBlock);
            auto aMut = graph->insert(aten::sub_, {a, constant});
            forkBlock->registerOutput(aMut);
            forkNode->output()->setType(FutureType::create(aMut->type()));
          }
          script::lambdaLiftFork(forkNode);
          return {forkNode->output()};
        });

    auto fut = if_->output();
    auto wait = graph->insert(aten::wait, {fut})->node();
    auto d = graph->insert(aten::add, {a, a});

    graph->lint();

    // Should not be able to move `d` before the wait call
    AliasDb aliasDb(graph);
    ASSERT_FALSE(aliasDb.moveBeforeTopologicallyValid(d->node(), wait));
  }
}

void testAliasTracker() {
  auto graph = std::make_shared<Graph>();
  const Value* a = graph->addInput();
  const Value* b = graph->addInput();
  const Value* c = graph->addInput();
  const Value* d = graph->addInput();
  const Value* e = graph->addInput();
  const Value* f = graph->addInput();
  const Value* g = graph->addInput();
  const Value* wc = graph->addInput();

  {
    // test contains()
    AliasTracker t;
    t.makeFreshValue(a);
    ASSERT_TRUE(t.contains(a));
    ASSERT_FALSE(t.contains(b));
  }
  {
    // a <- b <- c
    //      b <- d
    // a <- e
    // f <- e
    // g is by itself
    // wc is a wildcard value
    AliasTracker t;
    t.makeFreshValue(a);
    t.makeFreshValue(f);
    t.makeFreshValue(g);
    t.makePointerTo(b, a);
    t.makePointerTo(c, b);
    t.makePointerTo(d, b);
    t.makePointerTo(e, a);
    t.makePointerTo(e, f);
    t.setWildcard(wc);

    /**
     * Test mayAlias()
     */
    // Values should alias themselves
    ASSERT_TRUE(t.mayAlias(a, a));
    ASSERT_TRUE(t.mayAlias(g, g));

    // Values that point to the same location should alias
    ASSERT_TRUE(t.mayAlias(a, b));
    ASSERT_TRUE(t.mayAlias(a, c));
    ASSERT_TRUE(t.mayAlias(c, d));

    // e may point to a OR f
    ASSERT_TRUE(t.mayAlias(e, a));
    ASSERT_TRUE(t.mayAlias(e, f));
    // But a and f don't alias
    ASSERT_FALSE(t.mayAlias(a, f));

    // Wildcards should alias everything
    ASSERT_TRUE(t.mayAlias(wc, a));
    ASSERT_TRUE(t.mayAlias(wc, b));
    ASSERT_TRUE(t.mayAlias(wc, f));
    ASSERT_TRUE(t.mayAlias(wc, g));

    /**
     * Test mayAlias() set interface
     */
    std::multiset<const Value*> foo{c, c, d};
    std::multiset<const Value*> bar{e, f};
    std::unordered_set<const Value*> baz{f, g};
    std::set<const Value*> containsWildcard{wc};
    ASSERT_TRUE(t.mayAlias(foo, bar));
    ASSERT_TRUE(t.mayAlias(bar, baz));
    ASSERT_FALSE(t.mayAlias(foo, baz));
    // wildcard stuff aliases everything
    ASSERT_TRUE(t.mayAlias(containsWildcard, foo));
    ASSERT_TRUE(t.mayAlias(containsWildcard, bar));
    ASSERT_TRUE(t.mayAlias(containsWildcard, baz));

    /**
     * Test writer tracking
     */
    auto n1 = graph->appendNode(graph->create(prim::Undefined));
    auto n2 = graph->appendNode(graph->create(prim::Undefined));
    auto n3 = graph->appendNode(graph->create(prim::Undefined));
    t.registerWrite(a, n1);
    t.registerWrite(f, n2);
    // We should report those writes accurately
    ASSERT_TRUE(t.writesTo(n1, a));
    ASSERT_TRUE(t.writesTo(n2, f));
    ASSERT_FALSE(t.writesTo(n1, f));
    ASSERT_FALSE(t.writesTo(n2, a));
    // We should correctly report writes to aliases as well
    ASSERT_TRUE(t.writesTo(n1, c));

    // Check hasWriters()
    ASSERT_TRUE(t.hasWriters(a));
    // Aliases of written-to values should have writers
    ASSERT_TRUE(t.hasWriters(b));
    ASSERT_TRUE(t.hasWriters(d));
    ASSERT_TRUE(t.hasWriters(e));
    // Unique values not registered should be unaffected
    ASSERT_FALSE(t.hasWriters(g));

    // create a write to the wildcard set
    t.registerWrite(wc, n3);
    // Now everything may be written to
    ASSERT_TRUE(t.hasWriters(g));
    const auto& wildcardWriters = t.getWildcardWriters();
    ASSERT_EQ(wildcardWriters.size(), 1);
    ASSERT_EQ(*wildcardWriters.begin(), n3);
  }
}
} // namespace jit
} // namespace torch
