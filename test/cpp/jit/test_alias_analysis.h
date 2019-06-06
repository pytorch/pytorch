#pragma once

#include <torch/csrc/jit/irparser.h>
#include "test/cpp/jit/test_base.h"
#include "torch/csrc/jit/custom_operator.h"
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
    auto node = graph->appendNode(graph->create(prim::AutogradZero, inputs));
    node->output()->setUniqueName(name);
    nodes[name] = node;

    if (blockInputNames.size() != 0) {
      node->addBlock();
      std::vector<Value*> blockDeps;
      for (const auto name : blockInputNames) {
        blockDeps.push_back(nodes.at(name)->output());
      }

      auto block = node->blocks().at(0);
      block->appendNode(graph->create(prim::AutogradZero, blockDeps));
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

  // test none value does not have writers
  {
    {
      auto graph = std::make_shared<Graph>();
      std::unordered_map<std::string, Value*> vmap;
      script::parseIR(
          R"IR(
    graph():
      %opt : Tensor? = prim::Constant()
      %out : Tensor = prim::unchecked_unwrap_optional(%opt)
      %ret.2 : Tensor = aten::div(%out, %out, %out)
      return (%opt, %out, %ret.2)
      )IR",
          &*graph,
          vmap);

      AliasDb aliasDb(graph);
      AT_ASSERT(!aliasDb.hasWriters(vmap["opt"]->node()));
    }
  }
}

void testWriteTracking() {
  RegisterOperators reg(
      {Operator("prim::creates_alias(Tensor(a) x) -> Tensor(a)", [](Stack& s) {
        return 0;
      })});
  const auto creates_alias = Symbol::fromQualString("prim::creates_alias");
  {
    auto graph = std::make_shared<Graph>();
    auto a = graph->addInput();
    auto b = graph->addInput();

    // aten::add(%b, %b)
    // aten::add_(%a, %b)
    // foo::creates_alias(%a)
    auto pureNode = graph->insert(aten::add, {b, b})->node();
    auto writingNode = graph->insert(aten::add_, {a, b})->node();
    auto node3 = graph->insert(creates_alias, {a})->node();
    auto aAlias = node3->output();

    graph->lint();

    AliasDb aliasDb(graph);
    ASSERT_TRUE(aliasDb.mayAlias(aAlias, a));
    ASSERT_TRUE(aliasDb.mayAlias(a, b));
    ASSERT_FALSE(
        aliasDb.writesToAlias(pureNode, std::unordered_set<const Value*>{a}));
    ASSERT_FALSE(
        aliasDb.writesToAlias(pureNode, std::unordered_set<const Value*>{b}));
    ASSERT_TRUE(aliasDb.writesToAlias(
        writingNode, std::unordered_set<const Value*>{a}));
    ASSERT_TRUE(aliasDb.writesToAlias(
        writingNode, std::unordered_set<const Value*>{a, b}));
    ASSERT_TRUE(aliasDb.writesToAlias(
        writingNode, std::unordered_set<const Value*>{aAlias}));
  }
}

void testContainerAliasing() {
  {
    auto graph = std::make_shared<Graph>();
    script::parseIR(
        R"IR(
  graph():
    %x : str = prim::Constant[value="a"]()
    %y : Tensor = prim::Constant()
    %a : (Tensor) = prim::TupleConstruct(%y)
    %b : Dict(str, Tensor) = prim::DictConstruct(%x, %y)
    %c : Tensor[] = prim::ListConstruct(%y)
    return (%a, %b, %c)
    )IR",
        &*graph);

    auto node_iter = graph->block()->nodes().begin();
    auto str_node = node_iter++; // string
    Node* ten_node = *node_iter++;
    AliasDb aliasDb(graph);

    AT_ASSERT(graph->outputs().size() == 3);
    for (auto out : graph->outputs()) {
      AT_ASSERT(aliasDb.mayContainAlias(ten_node->output(), out));
    }
    AT_ASSERT(aliasDb.mayContainAlias({ten_node->output()}, graph->outputs()));
    AT_ASSERT(!aliasDb.mayContainAlias(str_node->output(), graph->outputs()));
  }

  {
    auto graph = std::make_shared<Graph>();
    script::parseIR(
        R"IR(
  graph():
    %x : str = prim::Constant[value="a"]()
    %y : int = prim::Constant[value=1]()
    %a : (int) = prim::TupleConstruct(%y)
    %b : Dict(str, int) = prim::DictConstruct(%x, %y)
    %c : int[] = prim::ListConstruct(%y)
    return (%a, %b, %c)
    )IR",
        &*graph);

    auto node_iter = graph->block()->nodes().begin();
    node_iter++; // string
    Node* int_node = *node_iter++;
    AliasDb aliasDb(graph);

    AT_ASSERT(graph->outputs().size() == 3);
    // primitive values don't need to alias container
    for (auto out : graph->outputs()) {
      AT_ASSERT(!aliasDb.mayContainAlias(int_node->output(), out));
    }
  }

  // Test input aliasing
  {
    auto graph = std::make_shared<Graph>();
    script::parseIR(
        R"IR(
  graph(%x: Tensor, %y: Tensor):
    %a : (Tensor) = prim::TupleConstruct(%x)
    return (%a)
    )IR",
        &*graph);

    auto node_iter = graph->block()->nodes().begin();
    auto tuple_node = *node_iter;
    AliasDb aliasDb(graph);

    for (auto input : graph->inputs()) {
      AT_ASSERT(aliasDb.mayContainAlias(input, tuple_node->output()));
    }
    AT_ASSERT(aliasDb.mayContainAlias(graph->inputs(), graph->outputs()));
  }

  // Test tuple that doesn't come from construct
  {
    auto graph = std::make_shared<Graph>();
    script::parseIR(
        R"IR(
graph(%x : int,
      %y : Tensor,
      %z : Tensor):
  %3 : int = prim::Constant[value=1]()
  %4 : bool = aten::eq(%x, %3)
  %a : (Tensor) = prim::If(%4)
    block0():
      %a.1 : (Tensor) = prim::TupleConstruct(%y)
      -> (%a.1)
    block1():
      %a.2 : (Tensor) = prim::TupleConstruct(%z)
      -> (%a.2)
  return (%a)
 )IR",
        &*graph);

    AliasDb aliasDb(graph);

    for (auto input : graph->inputs()) {
      if (input->type() == IntType::get()) {
        continue;
      }

      AT_ASSERT(aliasDb.mayContainAlias(input, graph->outputs().at(0)));
    }
  }

  // test nested types
  {
    auto graph = std::make_shared<Graph>();
    script::parseIR(
        R"IR(
graph():
  %4 : Device? = prim::Constant()
  %2 : int? = prim::Constant()
  %0 : float = prim::Constant[value=1]()
  %20 : bool = prim::Constant[value=0]()
  %a : Tensor = aten::tensor(%0, %2, %4, %20)
  %a_list : Tensor[] = prim::ListConstruct(%a)
  %b : Tensor = aten::tensor(%0, %2, %4, %20)
  %b_list : Tensor[] = prim::ListConstruct(%b)
  %13 : (Tensor[], Tensor[]) = prim::TupleConstruct(%a_list, %b_list)
  return (%13)
)IR",
        &*graph);
    AliasDb aliasDb(graph);
    auto g_output = graph->outputs().at(0);
    auto list_2 = g_output->node()->inputs().at(0);
    auto list_1 = g_output->node()->inputs().at(1);

    // TODO FIX assume conservatively for now
    AT_ASSERT(aliasDb.mayContainAlias(list_1, list_2));
    AT_ASSERT(aliasDb.mayContainAlias(list_2, list_1));

    AT_ASSERT(aliasDb.mayContainAlias(list_1, g_output));
    AT_ASSERT(aliasDb.mayContainAlias(list_2, g_output));
  }

  // simple example
  {
    auto graph = std::make_shared<Graph>();
    script::parseIR(
        R"IR(
graph():
  %0 : Tensor = prim::Constant()
  %1 : Tensor = prim::Constant()
  %13 : (Tensor) = prim::TupleConstruct(%0)
  return (%13)
)IR",
        &*graph);
    AliasDb aliasDb(graph);

    auto node_iter = graph->block()->nodes().begin();
    auto first_ten = *node_iter++;
    auto second_ten = *node_iter++;
    auto tup_node = *node_iter;

    AT_ASSERT(aliasDb.mayContainAlias(first_ten->output(), tup_node->output()));
    AT_ASSERT(
        !aliasDb.mayContainAlias(second_ten->output(), tup_node->output()));

    std::vector<Value*> first_st = {first_ten->output()};
    std::vector<Value*> second_st = {second_ten->output()};
    std::vector<Value*> tup_st = {tup_node->output()};
    AT_ASSERT(aliasDb.mayContainAlias(first_st, tup_st));
    AT_ASSERT(!aliasDb.mayContainAlias(first_st, second_st));
    AT_ASSERT(!aliasDb.mayContainAlias(second_st, tup_st));
  }
  {
    // Test list container aliasing
    auto graph = std::make_shared<Graph>();
    std::unordered_map<std::string, Value*> vmap;
    script::parseIR(
        R"IR(
graph():
  %10 : bool? = prim::Constant()
  %8 : Device? = prim::Constant()
  %4 : int? = prim::Constant()
  %0 : int = prim::Constant[value=2]()
  %1 : int = prim::Constant[value=3]()
  %2 : int[] = prim::ListConstruct(%0, %1)
  %x : Tensor = aten::rand(%2, %4, %4, %8, %10)
  %12 : int[] = prim::ListConstruct(%0, %1)
  %y : Tensor = aten::rand(%12, %4, %4, %8, %10)
  %22 : int[] = prim::ListConstruct(%0, %1)
  %z : Tensor = aten::rand(%22, %4, %4, %8, %10)
  %32 : int[] = prim::ListConstruct(%0, %1)
  %fresh : Tensor = aten::rand(%32, %4, %4, %8, %10)
  %foo : Tensor[] = prim::ListConstruct(%x, %y)
  %43 : Tensor[] = aten::append(%foo, %z)
  return ()
)IR",
        graph.get(),
        vmap);
    AliasDb aliasDb(graph);
    auto x = vmap["x"];
    auto y = vmap["y"];
    auto z = vmap["z"];
    // Tensors x, y, and z went into a list, so they all may alias each other.
    ASSERT_TRUE(aliasDb.mayAlias(x, y));
    ASSERT_TRUE(aliasDb.mayAlias(y, z));
    ASSERT_TRUE(aliasDb.mayAlias(x, z));

    // But we know `fresh` didn't go into a list, so x, y, and z should not
    // alias it.
    auto fresh = vmap["fresh"];
    ASSERT_FALSE(aliasDb.mayAlias(x, fresh));
    ASSERT_FALSE(aliasDb.mayAlias(y, fresh));
    ASSERT_FALSE(aliasDb.mayAlias(z, fresh));
  }
  {
    // test "conservative" analysis writes to the inside of a container.
    auto ops = torch::RegisterOperators(
        "custom::conservative", [](std::vector<at::Tensor> in) { return in; });

    auto graph = std::make_shared<Graph>();
    std::unordered_map<std::string, Value*> vmap;
    script::parseIR(
        R"IR(
graph():
  %10 : bool? = prim::Constant()
  %8 : Device? = prim::Constant()
  %4 : int? = prim::Constant()
  %0 : int = prim::Constant[value=2]()
  %1 : int = prim::Constant[value=3]()
  %2 : int[] = prim::ListConstruct(%0, %1)
  %11 : Tensor = aten::rand(%2, %4, %4, %8, %10)
  %12 : Tensor[] = prim::ListConstruct(%11)
  %out : Tensor[] = custom::conservative(%12)
  %ret.2 : Tensor = aten::div(%11, %11)
  return ()
)IR",
        graph.get(),
        vmap);
    AliasDb aliasDb(graph);
    auto conservativeOp = vmap["out"]->node();
    auto tensor = vmap["11"];
    ASSERT_TRUE(aliasDb.writesToAlias(conservativeOp, ValueSet{tensor}));
  }
}

void testWildcards() {
  RegisterOperators reg(
      {Operator(
           "prim::returns_wildcard(Tensor a) -> Tensor(*)",
           [](Stack& stack) { return 0; }),
       Operator("prim::writes(Tensor(z!) a) -> Tensor(a)", [](Stack& stack) {
         return 0;
       })});
  const auto returns_wildcard =
      Symbol::fromQualString("prim::returns_wildcard");
  const auto writes = Symbol::fromQualString("prim::writes");

  auto graph = std::make_shared<Graph>();
  const auto a = graph->addInput();

  const auto constant = graph->insertConstant(1);
  const auto fresh = graph->insert(aten::rand, {constant});
  const auto fresh2 = graph->insert(aten::rand, {constant});
  const auto wildcard = graph->insert(returns_wildcard, {fresh});

  {
    graph->lint();
    AliasDb aliasDb(graph);

    ASSERT_FALSE(aliasDb.mayAlias(a, fresh));
    ASSERT_FALSE(aliasDb.mayAlias(wildcard, fresh));
    ASSERT_TRUE(aliasDb.mayAlias(wildcard, a));
    ASSERT_FALSE(aliasDb.mayAlias(ValueSet{wildcard}, ValueSet{}));
    ASSERT_FALSE(aliasDb.hasWriters(wildcard->node()));
  }

  graph->insert(writes, {fresh2})->node();
  {
    graph->lint();
    AliasDb aliasDb(graph);
    ASSERT_FALSE(aliasDb.hasWriters(wildcard->node()));
  }

  const auto wildcardWrite = graph->insert(writes, {wildcard})->node();
  {
    graph->lint();
    AliasDb aliasDb(graph);
    // Test writes to wildcards
    ASSERT_FALSE(aliasDb.writesToAlias(
        wildcardWrite, std::unordered_set<const Value*>{fresh}));
    ASSERT_FALSE(aliasDb.writesToAlias(
        wildcardWrite, std::unordered_set<const Value*>{fresh2}));
    ASSERT_TRUE(aliasDb.writesToAlias(
        wildcardWrite, std::unordered_set<const Value*>{a}));
    ASSERT_TRUE(aliasDb.hasWriters(wildcard->node()));
  }
}

void testMemoryDAG() {
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
    MemoryDAG t;
    auto a = t.makeFreshValue(aValue);
    auto b = t.makeFreshValue(bValue);
    auto c = t.makeFreshValue(cValue);
    auto d = t.makeFreshValue(dValue);
    auto e = t.makeFreshValue(eValue);
    auto f = t.makeFreshValue(fValue);
    auto g = t.makeFreshValue(gValue);
    t.makePointerTo(b, a);
    t.makePointerTo(c, b);
    t.makePointerTo(d, b);
    t.makePointerTo(e, a);
    t.makePointerTo(e, f);

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
  }

  {
    // x(y) -> x contains y

    // b(a)
    // c(a)
    MemoryDAG t;
    auto a = t.makeFreshValue(aValue);
    auto b = t.makeFreshValue(bValue);
    t.addToContainedElements(a, b);

    auto c = t.makeFreshValue(cValue);
    t.addToContainedElements(a, c);

    AT_ASSERT(t.mayContainAlias(a, b));
    AT_ASSERT(t.mayContainAlias(b, a));

    AT_ASSERT(t.mayContainAlias(a, c))
    AT_ASSERT(t.mayContainAlias(c, a));

    AT_ASSERT(t.mayContainAlias(b, c));
    AT_ASSERT(t.mayContainAlias(c, b));

    // containers contain an element in themselves
    AT_ASSERT(t.mayContainAlias(b, b));
    AT_ASSERT(t.mayContainAlias(c, c));
    AT_ASSERT(t.mayContainAlias(a, a));

    auto d = t.makeFreshValue(dValue);

    // b(a)
    // c(a)
    // d(b(a))
    t.addToContainedElements(b, d);
    AT_ASSERT(t.mayContainAlias(b, d));
    AT_ASSERT(t.mayContainAlias(d, b));

    AT_ASSERT(t.mayContainAlias(c, d));
    AT_ASSERT(t.mayContainAlias(d, c));

    AT_ASSERT(t.mayContainAlias(a, d));

    // f(e)
    auto f = t.makeFreshValue(aValue);
    auto e = t.makeFreshValue(bValue);

    t.addToContainedElements(f, e);

    for (auto elem : {a, b, c, d}) {
      AT_ASSERT(!t.mayContainAlias(f, elem));
      AT_ASSERT(!t.mayContainAlias(e, elem));
    }
  }
}

void testAliasRegistration() {
  {
    auto registry = torch::RegisterOperators()
      .op("foo::rand", torch::RegisterOperators::options()
        .catchAllKernel([](at::Tensor) -> at::Tensor { return at::rand({2, 2}); })
        .aliasAnalysis(AliasAnalysisKind::DEFAULT));
    const auto rand_op = Symbol::fromQualString("foo::rand");
    auto graph = std::make_shared<Graph>();
    auto a = graph->addInput();
    auto b = graph->insert(rand_op, {a});
    AliasDb aliasDb(graph);
    // Conservatively we assume there is a reference
    ASSERT_TRUE(aliasDb.mayAlias(a, b));
  }
  {
    auto registry = torch::RegisterOperators()
      .op("foo::pure", torch::RegisterOperators::options()
        .catchAllKernel([](at::Tensor t) -> at::Tensor { return t * 2; })
        .aliasAnalysis(AliasAnalysisKind::PURE));
    const auto rand_op = Symbol::fromQualString("foo::pure");
    auto graph = std::make_shared<Graph>();
    auto a = graph->addInput();
    auto b = graph->insert(rand_op, {a});
    AliasDb aliasDb(graph);
    // PURE means there is no reference
    ASSERT_FALSE(aliasDb.mayAlias(a, b));
  }
}

} // namespace jit
} // namespace torch
