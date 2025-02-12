#include <gtest/gtest.h>

#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/type_hashing.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

#include <ATen/TensorOperators.h>

namespace torch {
namespace jit {

inline c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

// Fixture to set up a graph and make assertions clearer
class TopologicalMoveTest : public ::testing::Test {
 protected:
  TopologicalMoveTest() {
    createGraph();
    aliasDb = std::make_unique<AliasDb>(graph);
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
    for (const auto& name_ : inputNames) {
      // NOLINTNEXTLINE(performance-inefficient-vector-operation)
      inputs.push_back(nodes.at(name_)->output());
    }
    auto node = graph->appendNode(graph->create(prim::AutogradZero, inputs));
    node->output()->setDebugName(name);
    nodes[name] = node;

    if (blockInputNames.size() != 0) {
      node->addBlock();
      std::vector<Value*> blockDeps;
      for (const auto& name_ : blockInputNames) {
        // NOLINTNEXTLINE(performance-inefficient-vector-operation)
        blockDeps.push_back(nodes.at(name_)->output());
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
      EXPECT_TRUE(originalOrdering[idx] == curNode);
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
      EXPECT_EQ(nodes.at(toInsert)->prev(), nodes.at(insertPoint));
    } else {
      EXPECT_EQ(nodes.at(toInsert)->next(), nodes.at(insertPoint));
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::shared_ptr<Graph> graph;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unique_ptr<AliasDb> aliasDb;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unordered_map<std::string, Node*> nodes;
};

TEST_F(TopologicalMoveTest, SplitsDeps) {
  // Check that we are removing `this`'s deps properly when we need to split
  // `this` and deps (see code for what the hell that means)
  EXPECT_TRUE(moveBeforeTopologicallyValid("q", "s"));
  checkPostCondition("q", "s", false);
}

// Move after
TEST_F(TopologicalMoveTest, MoveAfterBackwardSimple) {
  // Simple move backward
  EXPECT_TRUE(moveAfterTopologicallyValid("c", "a"));
  checkPostCondition("c", "a", true);
}
TEST_F(TopologicalMoveTest, MoveAfterBackwardInvalid) {
  // simple invalid move backward
  EXPECT_FALSE(moveAfterTopologicallyValid("d", "a"));
}

TEST_F(TopologicalMoveTest, MoveAfterNoOp) {
  // doesn't actually move anything
  EXPECT_TRUE(moveAfterTopologicallyValid("f", "e"));
  checkPostCondition("f", "e", true);
}

TEST_F(TopologicalMoveTest, MoveAfterBackwardMultipleDeps) {
  // move backward with multiple dependencies
  EXPECT_TRUE(moveAfterTopologicallyValid("e", "c"));
  checkPostCondition("e", "c", true);
}

TEST_F(TopologicalMoveTest, MoveAfterBackwardNonZeroWorkingSet) {
  // Move backward with non-zero working set
  EXPECT_TRUE(moveAfterTopologicallyValid("k", "f"));
  checkPostCondition("k", "f", true);
}

TEST_F(TopologicalMoveTest, MoveAfterForwardSimple) {
  // Simple move forward
  EXPECT_TRUE(moveAfterTopologicallyValid("c", "d"));
  checkPostCondition("c", "d", true);
}

TEST_F(TopologicalMoveTest, MoveAfterForwardNonZeroWorkingSet) {
  // Move forward with non-zero working set
  EXPECT_TRUE(moveAfterTopologicallyValid("f", "l"));
  checkPostCondition("f", "l", true);
}

// Move before
TEST_F(TopologicalMoveTest, MoveBeforeForwardSimple) {
  // Simple move forward
  EXPECT_TRUE(moveBeforeTopologicallyValid("b", "d"));
  checkPostCondition("b", "d", false);
}

TEST_F(TopologicalMoveTest, MoveBeforeBackwardSimple) {
  // Simple move backward
  EXPECT_TRUE(moveBeforeTopologicallyValid("c", "a"));
  checkPostCondition("c", "a", false);
}

TEST_F(TopologicalMoveTest, MoveBeforeNoOp) {
  // doesn't actually move anything
  EXPECT_TRUE(moveBeforeTopologicallyValid("a", "b"));
  checkPostCondition("a", "b", false);
}

TEST_F(TopologicalMoveTest, MoveBeforeForwardWithDeps) {
  // move forward with deps
  EXPECT_TRUE(moveBeforeTopologicallyValid("f", "m"));
  checkPostCondition("f", "m", false);
}

TEST_F(TopologicalMoveTest, MoveBeforeBackwardWithDeps) {
  // move backward with deps
  EXPECT_TRUE(moveBeforeTopologicallyValid("l", "f"));
  checkPostCondition("l", "f", false);
}

// check that dependencies in blocks are recognized
TEST_F(TopologicalMoveTest, DepsDisallowMove) {
  EXPECT_FALSE(moveAfterTopologicallyValid("l", "m"));
  EXPECT_FALSE(moveBeforeTopologicallyValid("m", "l"));
  EXPECT_FALSE(moveAfterTopologicallyValid("n", "l"));
  EXPECT_FALSE(moveBeforeTopologicallyValid("l", "n"));
}

// Test that moveAfter(n) and moveBefore(n->next()) are not necessarily
// equivalent. Here, the dependency ordering is n -> o -> p.  So we can't
// move `n` after `o`, but we can move `n` before `p` (which pushes `o` after
// `p`)
TEST_F(TopologicalMoveTest, MoveAfterBeforeWithDeps) {
  EXPECT_FALSE(moveAfterTopologicallyValid("n", "o"));
  EXPECT_TRUE(moveBeforeTopologicallyValid("o", "p"));
  checkPostCondition("o", "p", false);
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

  EXPECT_TRUE(trueBlock->outputs().size() == falseBlock->outputs().size());
  for (auto output : trueBlock->outputs()) {
    if_->addOutput()->setType(output->type());
  }
  return if_;
}

template <class Exception, class Functor>
inline void expectThrows(Functor&& functor, const char* expectMessageContains) {
  try {
    std::forward<Functor>(functor)();
  } catch (const Exception& e) {
    if (std::string(e.what()).find(expectMessageContains) ==
        std::string::npos) {
      TORCH_CHECK(
          false,
          "Expected error message to contain \"",
          expectMessageContains,
          "\" but error message was: ",
          e.what());
    }
    return;
  }
  TORCH_CHECK(
      false,
      "Expected to throw exception containing \"",
      expectMessageContains,
      "\" but didn't throw");
}

} // namespace

TEST(AliasAnalysisTest, AliasingMutationBlocksMoves) {
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
  EXPECT_FALSE(aliasDb.moveAfterTopologicallyValid(c->node(), aMut->node()));
  EXPECT_TRUE(aliasDb.moveAfterTopologicallyValid(d->node(), c->node()));

  // b should alias to a (since they are both inputs)
  EXPECT_FALSE(
      aliasDb.moveAfterTopologicallyValid(addsB->node(), aMut->node()));
  EXPECT_TRUE(aliasDb.moveAfterTopologicallyValid(addsB->node(), c->node()));

  graph->lint();
}

TEST(AliasAnalysisTest, AliasingMutationBlocksMoves2) {
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
  EXPECT_FALSE(aliasDb.moveAfterTopologicallyValid(
      aliasesB->node(), mutatesAliasOfB->node()));
  EXPECT_FALSE(aliasDb.moveAfterTopologicallyValid(
      usesB->node(), mutatesAliasOfB->node()));
}

TEST(AliasAnalysisTest, SideEffectsBlockMoves) {
  // Test moves across side effectful nodes
  auto graph = std::make_shared<Graph>();
  auto a = graph->addInput();
  auto print1 = graph->insertNode(graph->create(prim::Print, {a}, 0));
  WithInsertPoint guard(print1);
  auto print2 = graph->insertNode(graph->create(prim::Print, {a, a}, 0));
  AliasDb aliasDb(graph);

  // def foo(a):
  //  print2(a, a)
  //  print1(a)

  // test moving across each other
  EXPECT_FALSE(aliasDb.moveAfterTopologicallyValid(print2, print1));
  EXPECT_FALSE(aliasDb.moveBeforeTopologicallyValid(print1, print2));

  // test moving where they already are
  EXPECT_TRUE(aliasDb.moveBeforeTopologicallyValid(print2, print1));
  EXPECT_TRUE(aliasDb.moveAfterTopologicallyValid(print1, print2));

  graph->insertNode(graph->create(prim::MakeTestTensor, {}, 1));
  AliasDb aliasDb2(graph);

  // def foo(a):
  //  print2(a, a)
  //  non_side_effectful = makeTestTensor()
  //  print1(a)

  // test moving with a side effectful node between
  EXPECT_FALSE(aliasDb2.moveAfterTopologicallyValid(print2, print1));
  EXPECT_FALSE(aliasDb2.moveBeforeTopologicallyValid(print2, print1));
  EXPECT_FALSE(aliasDb2.moveAfterTopologicallyValid(print1, print2));
  EXPECT_FALSE(aliasDb2.moveBeforeTopologicallyValid(print1, print2));
}

TEST(AliasAnalysisTest, MovingAcrossInnerBlocks) {
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
  EXPECT_FALSE(aliasDb.moveBeforeTopologicallyValid(c->node(), if_));
}

TEST(AliasAnalysisTest, NoneHasNoWriters) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(
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
  EXPECT_FALSE(aliasDb.hasWriters(vmap["opt"]->node()));
}

TEST(AliasAnalysisTest, SafeToChangeAliasingRelationship) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(
      R"IR(
  graph(%x : Tensor):
      %3 : int = prim::Constant[value=1]()
      %2 : int = prim::Constant[value=0]()
      %b : Tensor = aten::add(%x, %2, %3)
      %c : Tensor = aten::add(%x, %2, %3)
      %d : Tensor = aten::add(%x, %2, %3)
      %e : Tensor = aten::add(%x, %2, %3)
      %f : Tensor[] = prim::ListConstruct(%e)
      %14 : (Tensor, Tensor) = prim::TupleConstruct(%b, %c)
      return (%14)
    )IR",
      &*graph,
      vmap);

  AliasDb aliasDb(graph);
  // x, b, c escape scope, so we can't introduce an aliasing relationship
  EXPECT_FALSE(aliasDb.safeToChangeAliasingRelationship(vmap["x"], vmap["b"]));
  EXPECT_FALSE(aliasDb.safeToChangeAliasingRelationship(vmap["b"], vmap["x"]));
  EXPECT_FALSE(aliasDb.safeToChangeAliasingRelationship(vmap["b"], vmap["c"]));
  EXPECT_FALSE(aliasDb.safeToChangeAliasingRelationship(vmap["c"], vmap["b"]));

  // e aliases the wildcard set because it's contained in a list
  EXPECT_FALSE(aliasDb.safeToChangeAliasingRelationship(vmap["e"], vmap["x"]));
  EXPECT_FALSE(aliasDb.safeToChangeAliasingRelationship(vmap["x"], vmap["e"]));

  // d is a temporary with no writers, safe to change aliasing relationship
  // here
  EXPECT_TRUE(aliasDb.safeToChangeAliasingRelationship(vmap["c"], vmap["d"]));
  EXPECT_TRUE(aliasDb.safeToChangeAliasingRelationship(vmap["d"], vmap["c"]));
}

class BatchAndInstanceNormFixture
    : public ::testing::TestWithParam<std::tuple<std::string, NodeKind, bool>> {
};

TEST_P(BatchAndInstanceNormFixture, BatchAndInstanceNorm) {
  auto param = GetParam();
  auto fnName = std::get<0>(param);
  auto nodeKind = std::get<1>(param);
  auto isTraining = std::get<2>(param);
  std::string isTrainingStr = std::to_string((int)isTraining);

  auto graph = std::make_shared<Graph>();

  parseIR(
      R"IR(
  graph(%input : Tensor, %running_mean : Tensor, %running_var : Tensor):
      %none : NoneType = prim::Constant()
      %training : bool = prim::Constant[value=)IR" +
          isTrainingStr + R"IR(]()
      %momentum : float = prim::Constant[value=1.0]()
      %eps : float = prim::Constant[value=1.0e-9]()
      %cudnn_enabled : bool = prim::Constant[value=0]()
      %res : Tensor = )IR" +
          fnName +
          R"IR((%input, %none, %none, %running_mean, %running_var, %training, %momentum, %eps, %cudnn_enabled)
      return (%res)
    )IR",
      &*graph);

  graph->lint();
  DepthFirstGraphNodeIterator it(graph);

  Node* n = nullptr;
  while ((n = it.next()) != nullptr) {
    if (n->kind() == nodeKind) {
      break;
    }
  }
  EXPECT_TRUE(n != nullptr);

  AliasDb aliasDb(graph);
  EXPECT_TRUE(aliasDb.hasWriters(n) == isTraining);
}

TEST_P(BatchAndInstanceNormFixture, BatchAndInstanceNormTrainingUnknown) {
  auto param = GetParam();
  auto fnName = std::get<0>(param);
  auto nodeKind = std::get<1>(param);

  auto graph = std::make_shared<Graph>();

  parseIR(
      R"IR(
  graph(%input : Tensor, %running_mean : Tensor, %running_var : Tensor, %training : bool):
      %none : NoneType = prim::Constant()
      %momentum : float = prim::Constant[value=1.0]()
      %eps : float = prim::Constant[value=1.0e-9]()
      %cudnn_enabled : bool = prim::Constant[value=0]()
      %res : Tensor = )IR" +
          fnName +
          R"IR((%input, %none, %none, %running_mean, %running_var, %training, %momentum, %eps, %cudnn_enabled)
      return (%res)
    )IR",
      &*graph);

  graph->lint();
  DepthFirstGraphNodeIterator it(graph);

  Node* n = nullptr;
  while ((n = it.next()) != nullptr) {
    if (n->kind() == nodeKind) {
      break;
    }
  }
  EXPECT_TRUE(n != nullptr);

  AliasDb aliasDb(graph);
  EXPECT_TRUE(aliasDb.hasWriters(n));
}

TEST_P(BatchAndInstanceNormFixture, BatchNormTrainingWithNoMeanOrVar) {
  auto param = GetParam();
  auto fnName = std::get<0>(param);
  auto nodeKind = std::get<1>(param);
  auto isTraining = std::get<2>(param);
  std::string isTrainingStr = std::to_string((int)isTraining);

  auto graph = std::make_shared<Graph>();

  parseIR(
      R"IR(
  graph(%input : Tensor):
      %none : NoneType = prim::Constant()
      %training : bool = prim::Constant[value=)IR" +
          isTrainingStr + R"IR(]()
      %momentum : float = prim::Constant[value=1.0]()
      %eps : float = prim::Constant[value=1.0e-9]()
      %cudnn_enabled : bool = prim::Constant[value=0]()
      %res : Tensor = )IR" +
          fnName +
          R"IR((%input, %none, %none, %none, %none, %training, %momentum, %eps, %cudnn_enabled)
      return (%res)
    )IR",
      &*graph);

  graph->lint();
  DepthFirstGraphNodeIterator it(graph);

  Node* n = nullptr;
  while ((n = it.next()) != nullptr) {
    if (n->kind() == nodeKind) {
      break;
    }
  }
  EXPECT_TRUE(n != nullptr);

  AliasDb aliasDb(graph);
  EXPECT_FALSE(aliasDb.hasWriters(n));
}

INSTANTIATE_TEST_SUITE_P(
    AliasAnalysisTest,
    BatchAndInstanceNormFixture,
    ::testing::Values(
        std::make_tuple("aten::batch_norm", aten::batch_norm, false),
        std::make_tuple("aten::instance_norm", aten::instance_norm, false),
        std::make_tuple("aten::batch_norm", aten::batch_norm, true),
        std::make_tuple("aten::instance_norm", aten::instance_norm, true)));

TEST(WriteTrackingTest, Basic) {
  RegisterOperators reg({Operator(
      "prim::creates_alias(Tensor(a) x) -> Tensor(a)",
      [](Stack&) {},
      aliasAnalysisFromSchema())});
  const auto creates_alias = Symbol::fromQualString("prim::creates_alias");
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
  EXPECT_TRUE(aliasDb.mayAlias(aAlias, a));
  EXPECT_TRUE(aliasDb.mayAlias(a, b));
  EXPECT_FALSE(
      aliasDb.writesToAlias(pureNode, std::unordered_set<const Value*>{a}));
  EXPECT_FALSE(
      aliasDb.writesToAlias(pureNode, std::unordered_set<const Value*>{b}));
  EXPECT_TRUE(
      aliasDb.writesToAlias(writingNode, std::unordered_set<const Value*>{a}));
  EXPECT_TRUE(aliasDb.writesToAlias(
      writingNode, std::unordered_set<const Value*>{a, b}));
  EXPECT_TRUE(aliasDb.writesToAlias(
      writingNode, std::unordered_set<const Value*>{aAlias}));
}

TEST(WriteTrackingTest, IsMutable) {
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
  graph(%x: Tensor):
    %b : Tensor = aten::relu_(%x)
    return (%b)
    )IR",
      &*graph);
  auto node_iter = graph->block()->nodes().begin();
  auto relu = *node_iter;
  AliasDb aliasDb(graph);
  EXPECT_TRUE(aliasDb.isMutable(relu));
}

TEST(WriteTrackingTest, IsImmutable) {
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
  graph(%x: Tensor, %y : Tensor):
    %b : Tensor = aten::mul(%x, %y)
    return (%b)
    )IR",
      &*graph);
  auto node_iter = graph->block()->nodes().begin();
  auto mul = *node_iter;
  AliasDb aliasDb(graph);
  EXPECT_FALSE(aliasDb.isMutable(mul));
}

TEST(WriteTrackingTest, HasWriters) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(
      R"IR(
  graph(%x: Tensor, %y : Tensor):
    %c1 : int = prim::Constant[value=1]()
    %b : Tensor = aten::add_(%x, %y, %c1)
    return (%b)
    )IR",
      &*graph,
      vmap);
  auto add = vmap["b"]->node();
  AliasDb aliasDb(graph);
  EXPECT_TRUE(aliasDb.hasWriters(add));
  EXPECT_TRUE(aliasDb.isMutable(add));
}

TEST(ContainerAliasingTest, MayContainAlias) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(
      R"IR(
  graph(%inp: Tensor[]):
    %x : str = prim::Constant[value="a"]()
    %y : Tensor = prim::Constant()
    %z : Tensor = prim::Constant()
    %a : (Tensor) = prim::TupleConstruct(%y)
    %b : Dict(str, Tensor) = prim::DictConstruct(%x, %y)
    %c : Tensor[] = prim::ListConstruct(%y)
    return (%a, %b, %c)
    )IR",
      &*graph,
      vmap);

  auto str_output = vmap["x"];
  auto ten_output = vmap["y"];
  auto local_var = vmap["z"];
  AliasDb aliasDb(graph);

  EXPECT_TRUE(graph->outputs().size() == 3);
  for (auto out : graph->outputs()) {
    EXPECT_TRUE(aliasDb.mayContainAlias(ten_output, out));
    EXPECT_FALSE(aliasDb.mayContainAlias(local_var, out));
  }

  EXPECT_TRUE(aliasDb.mayContainAlias(ten_output, graph->inputs()));
  EXPECT_FALSE(aliasDb.mayContainAlias(local_var, graph->inputs()));

  EXPECT_TRUE(aliasDb.mayContainAlias(ten_output, graph->outputs()));
  EXPECT_TRUE(aliasDb.mayContainAlias(
      at::ArrayRef<Value*>{ten_output}, graph->outputs()));
  EXPECT_FALSE(aliasDb.mayContainAlias(str_output, graph->outputs()));
}

TEST(ContainerAliasingTest, MayContainAlias_cast) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(
      R"IR(
  graph(%input.1 : Tensor):
    %2 : NoneType = prim::Constant()
    %3 : bool = prim::Constant[value=0]()
    %4 : int = prim::Constant[value=6]()
    %5 : int = prim::Constant[value=1]()
    %a.1 : Tensor = aten::add(%input.1, %input.1, %5)
    %b.1 : Tensor = aten::to(%a.1, %4, %3, %3, %2)
    %c.1 : Tensor = aten::mul(%b.1, %b.1)
    return (%c.1)
    )IR",
      &*graph,
      vmap);

  auto a = vmap["a.1"];
  auto b = vmap["b.1"];
  auto c = vmap["c.1"];
  AliasDb aliasDb(graph);

  EXPECT_TRUE(graph->outputs().size() == 1);
  for (auto out : graph->outputs()) {
    EXPECT_TRUE(aliasDb.mayContainAlias(c, out));
  }

  EXPECT_TRUE(aliasDb.mayContainAlias(a, b));
  EXPECT_FALSE(aliasDb.mayContainAlias(b, graph->inputs()));

  EXPECT_TRUE(aliasDb.mayContainAlias(c, graph->outputs()));
  EXPECT_TRUE(
      aliasDb.mayContainAlias(at::ArrayRef<Value*>{c}, graph->outputs()));
  EXPECT_FALSE(aliasDb.mayContainAlias(b, graph->outputs()));
}

TEST(ContainerAliasingTest, PrimitveValuesDontAliasContainers) {
  auto graph = std::make_shared<Graph>();
  parseIR(
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

  EXPECT_TRUE(graph->outputs().size() == 3);
  // primitive values don't need to alias container
  for (auto out : graph->outputs()) {
    EXPECT_FALSE(aliasDb.mayContainAlias(int_node->output(), out));
  }
}

TEST(ContainerAliasingTest, UnionAliasing) {
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
  graph(%a : Dict(str, Tensor),
        %b : Tensor[],
        %c : Union(Dict(str, Tensor), Tensor[])):
    return (%a, %b, %c)
    )IR",
      &*graph);

  AliasDb aliasDb(graph);
  auto a = graph->outputs().at(0);
  auto b = graph->outputs().at(1);
  auto c = graph->outputs().at(2);

  EXPECT_TRUE(aliasDb.mayAlias(a, c));
  EXPECT_TRUE(aliasDb.mayAlias(b, c));
  EXPECT_TRUE(aliasDb.mayAlias(c, c));
  EXPECT_FALSE(aliasDb.mayAlias(a, b));
  EXPECT_TRUE(aliasDb.mayContainAlias(a, b));
  EXPECT_TRUE(aliasDb.mayContainAlias(a, c));
  EXPECT_TRUE(aliasDb.mayContainAlias(b, c));
}

TEST(ContainerAliasingTest, InputsCanAliasOutputs) {
  // Test input aliasing
  auto graph = std::make_shared<Graph>();
  parseIR(
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
    EXPECT_TRUE(aliasDb.mayContainAlias(input, tuple_node->output()));
  }
  EXPECT_TRUE(aliasDb.mayContainAlias(graph->inputs(), graph->outputs()));
}

// Test tuple that doesn't come from construct
TEST(ContainerAliasingTest, NestedTupleConstruct) {
  auto graph = std::make_shared<Graph>();
  parseIR(
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

    EXPECT_TRUE(aliasDb.mayContainAlias(input, graph->outputs().at(0)));
  }
}

// test nested types
TEST(ContainerAliasingTest, NestedTypes) {
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
graph():
  %a : Tensor = prim::MakeTestTensor()
  %a_list : Tensor[] = prim::ListConstruct(%a)
  %b : Tensor = prim::MakeTestTensor()
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
  EXPECT_TRUE(aliasDb.mayContainAlias(list_1, list_2));
  EXPECT_TRUE(aliasDb.mayContainAlias(list_2, list_1));

  EXPECT_TRUE(aliasDb.mayContainAlias(list_1, g_output));
  EXPECT_TRUE(aliasDb.mayContainAlias(list_2, g_output));
}

// simple example
TEST(ContainerAliasingTest, Simple) {
  auto graph = std::make_shared<Graph>();
  parseIR(
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

  EXPECT_TRUE(aliasDb.mayContainAlias(first_ten->output(), tup_node->output()));
  EXPECT_TRUE(
      !aliasDb.mayContainAlias(second_ten->output(), tup_node->output()));

  std::vector<Value*> first_st = {first_ten->output()};
  std::vector<Value*> second_st = {second_ten->output()};
  std::vector<Value*> tup_st = {tup_node->output()};
  EXPECT_TRUE(aliasDb.mayContainAlias(first_st, tup_st));
  EXPECT_FALSE(aliasDb.mayContainAlias(first_st, second_st));
  EXPECT_FALSE(aliasDb.mayContainAlias(second_st, tup_st));
}

TEST(ContainerAliasingTest, Lists) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(
      R"IR(
  graph():
    %x : str = prim::Constant[value="a"]()
    %y : Tensor = prim::Constant()
    %c : Tensor[] = prim::ListConstruct(%y)
    %d : Tensor[] = prim::ListConstruct(%y)
    return (%c, %d)
    )IR",
      &*graph,
      vmap);

  AliasDb aliasDb(graph);
  auto x = vmap["x"];
  auto c = vmap["c"];
  EXPECT_FALSE(aliasDb.mayContainAlias(x, c));
  EXPECT_FALSE(aliasDb.mayContainAlias(c, x));

  auto d = vmap["d"];

  EXPECT_TRUE(aliasDb.mayContainAlias(d, c));
  EXPECT_TRUE(aliasDb.mayContainAlias(c, d));
}

TEST(ContainerAliasingTest, Lists2) {
  // Test list container aliasing
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(
      R"IR(
graph():
  %0 : int = prim::Constant[value=2]()
  %1 : int = prim::Constant[value=3]()
  %2 : int[] = prim::ListConstruct(%0, %1)
  %x : Tensor = prim::MakeTestTensor()
  %12 : int[] = prim::ListConstruct(%0, %1)
  %y : Tensor = prim::MakeTestTensor()
  %22 : int[] = prim::ListConstruct(%0, %1)
  %z : Tensor = prim::MakeTestTensor()
  %32 : int[] = prim::ListConstruct(%0, %1)
  %fresh : Tensor = prim::MakeTestTensor()
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
  EXPECT_TRUE(aliasDb.mayAlias(x, y));
  EXPECT_TRUE(aliasDb.mayAlias(y, z));
  EXPECT_TRUE(aliasDb.mayAlias(x, z));

  // But we know `fresh` didn't go into a list, so x, y, and z should not
  // alias it.
  auto fresh = vmap["fresh"];
  EXPECT_FALSE(aliasDb.mayAlias(x, fresh));
  EXPECT_FALSE(aliasDb.mayAlias(y, fresh));
  EXPECT_FALSE(aliasDb.mayAlias(z, fresh));
}

TEST(ContainerAliasingTest, Conservative) {
  // test "conservative" analysis writes to the inside of a container.
  auto ops = torch::RegisterOperators(
      "custom::conservative", [](torch::List<at::Tensor> in) { return in; });

  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(
      R"IR(
graph():
  %0 : int = prim::Constant[value=2]()
  %1 : int = prim::Constant[value=3]()
  %2 : int[] = prim::ListConstruct(%0, %1)
  %11 : Tensor = prim::MakeTestTensor()
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
  EXPECT_TRUE(aliasDb.writesToAlias(conservativeOp, ValueSet{tensor}));
}

TEST(ContainerAliasingTest, MovesAcrossContainedWrites) {
  auto ops = torch::RegisterOperators().op(
      "uses::list",
      torch::RegisterOperators::options()
          .catchAllKernel([](torch::List<at::Tensor> in) {
            return torch::rand({2, 3});
          })
          .aliasAnalysis(AliasAnalysisKind::PURE_FUNCTION));
  // Write to the inside of a list. Check that we can't reorder a
  // print across it.
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(
      R"IR(
graph():
  %35 : int = prim::Constant[value=1]()
  %0 : int = prim::Constant[value=2]()
  %1 : int = prim::Constant[value=3]()
  %23 : int = prim::Constant[value=0]()
  %2 : int[] = prim::ListConstruct(%0, %1)
  %11 : Tensor = prim::MakeTestTensor()
  %12 : int[] = prim::ListConstruct(%0, %1)
  %21 : Tensor = prim::MakeTestTensor()
  %l : Tensor[] = prim::ListConstruct(%11, %21)
  %24 : Tensor = aten::select(%l, %23)
  %25 : int[] = prim::ListConstruct(%0, %1)
  %34 : Tensor = prim::MakeTestTensor()
  %36 : Tensor = aten::add_(%24, %34, %35)
  %37 : Tensor = uses::list(%l)
  return (%37)
)IR",
      graph.get(),
      vmap);
  AliasDb aliasDb(graph);
  auto listUse = vmap["37"]->node();
  auto internalWrite = vmap["36"]->node();
  EXPECT_FALSE(aliasDb.moveBeforeTopologicallyValid(listUse, internalWrite));
}

TEST(ContainerAliasingTest, MovesAcrossContainedWritesNested) {
  // The same as above, but with a nested list
  auto ops = torch::RegisterOperators().op(
      "uses::list",
      torch::RegisterOperators::options()
          .catchAllKernel([](torch::List<at::Tensor> in) {
            return torch::rand({2, 3});
          })
          .aliasAnalysis(AliasAnalysisKind::PURE_FUNCTION));
  // Write to the inside of a list. Check that we can't reorder a
  // print across it.
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(
      R"IR(
graph():
  %38 : int = prim::Constant[value=1]()
  %0 : int = prim::Constant[value=2]()
  %1 : int = prim::Constant[value=3]()
  %24 : int = prim::Constant[value=0]()
  %2 : int[] = prim::ListConstruct(%0, %1)
  %11 : Tensor = prim::MakeTestTensor()
  %12 : int[] = prim::ListConstruct(%0, %1)
  %21 : Tensor = prim::MakeTestTensor()
  %l : Tensor[] = prim::ListConstruct(%11, %21)
  %25 : Tensor = aten::select(%l, %24)
  %27 : Tensor = aten::select(%25, %24, %24)
  %28 : int[] = prim::ListConstruct(%0, %1)
  %37 : Tensor = prim::MakeTestTensor()
  %39 : Tensor = aten::add_(%27, %37, %38)
  %40 : Tensor = uses::list(%l)
  return (%40)
)IR",
      graph.get(),
      vmap);
  AliasDb aliasDb(graph);
  auto listUse = vmap["40"]->node();
  auto internalWrite = vmap["39"]->node();
  EXPECT_FALSE(aliasDb.moveBeforeTopologicallyValid(listUse, internalWrite));
}

TEST(WildcardsTest, Basic) {
  RegisterOperators reg(
      {Operator(
           "prim::returns_wildcard(Tensor a) -> Tensor(*)",
           [](Stack&) {},
           aliasAnalysisFromSchema()),
       Operator(
           "prim::writes(Tensor(z!) a) -> Tensor(a)",
           [](Stack&) {},
           aliasAnalysisFromSchema())});
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

    EXPECT_FALSE(aliasDb.mayAlias(a, fresh));
    EXPECT_FALSE(aliasDb.mayAlias(wildcard, fresh));
    EXPECT_TRUE(aliasDb.mayAlias(wildcard, a));
    EXPECT_FALSE(aliasDb.mayAlias(ValueSet{wildcard}, ValueSet{}));
    EXPECT_FALSE(aliasDb.hasWriters(wildcard->node()));
  }

  graph->insert(writes, {fresh2})->node();
  {
    graph->lint();
    AliasDb aliasDb(graph);
    EXPECT_FALSE(aliasDb.hasWriters(wildcard->node()));
  }

  const auto wildcardWrite = graph->insert(writes, {wildcard})->node();
  {
    graph->lint();
    AliasDb aliasDb(graph);
    // Test writes to wildcards
    EXPECT_FALSE(aliasDb.writesToAlias(
        wildcardWrite, std::unordered_set<const Value*>{fresh}));
    EXPECT_FALSE(aliasDb.writesToAlias(
        wildcardWrite, std::unordered_set<const Value*>{fresh2}));
    EXPECT_TRUE(aliasDb.writesToAlias(
        wildcardWrite, std::unordered_set<const Value*>{a}));
    EXPECT_TRUE(aliasDb.hasWriters(wildcard->node()));
  }
}

// test that wildcards are correctly divided by type
TEST(WildcardsTest, TypeIsolation) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(
      R"IR(
  graph(%ten_list : Tensor[], %int_list : int[], %opt_ten_list : Tensor[]?):
    %ten : Tensor = prim::Constant()
    %4 : Tensor[] = aten::append(%ten_list, %ten)
    %ten_ten_list : Tensor[][] = prim::Constant()
    %int_int_list : int[][] = prim::Constant()
    return ()
    )IR",
      &*graph,
      vmap);
  AliasDb aliasDb(graph);
  auto opt_ten_list = vmap["opt_ten_list"];
  auto ten_list = vmap["ten_list"];
  auto int_list = vmap["int_list"];
  EXPECT_FALSE(aliasDb.hasWriters(int_list));
  EXPECT_TRUE(aliasDb.hasWriters(opt_ten_list));
  EXPECT_TRUE(aliasDb.hasWriters(ten_list));
  EXPECT_FALSE(aliasDb.mayContainAlias(int_list, opt_ten_list));
  EXPECT_TRUE(aliasDb.mayContainAlias(ten_list, opt_ten_list));
  EXPECT_TRUE(aliasDb.mayAlias(ten_list, opt_ten_list));

  auto list_of_tensor_lists = vmap["ten_ten_list"];
  EXPECT_TRUE(aliasDb.mayContainAlias(ten_list, list_of_tensor_lists));
  EXPECT_TRUE(aliasDb.mayContainAlias(ten_list, vmap["ten"]));

  EXPECT_TRUE(
      !aliasDb.mayContainAlias(vmap["int_int_list"], list_of_tensor_lists));
}

// test invariant container aliasing
// the containers of different type cannot alias each other,
// however they may contain elements which alias each other
TEST(WildcardsTest, InvariantContainerAliasing) {
  {
    auto graph = std::make_shared<Graph>();
    std::unordered_map<std::string, Value*> vmap;
    parseIR(
        R"IR(
  graph(%ten_list : Tensor[], %ten_opt_list : Tensor?[]):
    %ten : Tensor = prim::Constant()
    %4 : Tensor[] = aten::append(%ten_list, %ten)
    return ()
    )IR",
        &*graph,
        vmap);
    AliasDb aliasDb(graph);
    auto ten_opt_list = vmap["ten_opt_list"];
    auto ten_list = vmap["ten_list"];
    EXPECT_FALSE(aliasDb.hasWriters(ten_opt_list));
    EXPECT_TRUE(aliasDb.hasWriters(ten_list));
    EXPECT_TRUE(aliasDb.mayContainAlias(ten_list, ten_opt_list));
    EXPECT_FALSE(aliasDb.mayAlias(ten_list, ten_opt_list));
  }
  {
    auto graph = std::make_shared<Graph>();
    std::unordered_map<std::string, Value*> vmap;
    parseIR(
        R"IR(
  graph(%float_3D : Float(*, *, *), %float_2D : Float(*, *)):
    return ()
    )IR",
        &*graph,
        vmap);
    AliasDb aliasDb(graph);
    EXPECT_TRUE(aliasDb.mayAlias(vmap["float_3D"], vmap["float_2D"]));
  }

  {
    auto graph = std::make_shared<Graph>();
    std::unordered_map<std::string, Value*> vmap;
    parseIR(
        R"IR(
  graph(%float_3D_list : Float(*, *, *)[], %float_2D_list : Float(*, *)[], %ten: Tensor):
    return ()
    )IR",
        &*graph,
        vmap);
    AliasDb aliasDb(graph);
    EXPECT_TRUE(aliasDb.mayAlias(vmap["float_3D_list"], vmap["float_2D_list"]));
    EXPECT_TRUE(aliasDb.mayContainAlias(vmap["float_3D_list"], vmap["ten"]));
    EXPECT_TRUE(aliasDb.mayContainAlias(vmap["float_2D_list"], vmap["ten"]));
  }
}

TEST(AliasRegistrationTest, ConservativeWithInferredSchema) {
  auto registry = torch::RegisterOperators().op(
      "foo::rand1",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor) -> at::Tensor {
            return at::rand({2, 2});
          })
          .aliasAnalysis(AliasAnalysisKind::CONSERVATIVE));
  const auto rand_op = Symbol::fromQualString("foo::rand1");
  auto graph = std::make_shared<Graph>();
  auto a = graph->addInput();
  auto b = graph->insert(rand_op, {a});
  AliasDb aliasDb(graph);
  // Conservatively we assume there is a reference
  EXPECT_TRUE(aliasDb.mayAlias(a, b));
}

TEST(AliasRegistrationTest, ConservativeWithSpecifiedSchema) {
  auto registry = torch::RegisterOperators().op(
      "foo::rand2(Tensor arg1) -> Tensor",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor) -> at::Tensor {
            return at::rand({2, 2});
          })
          .aliasAnalysis(AliasAnalysisKind::CONSERVATIVE));
  const auto rand_op = Symbol::fromQualString("foo::rand2");
  auto graph = std::make_shared<Graph>();
  auto a = graph->addInput();
  auto b = graph->insert(rand_op, {a});
  AliasDb aliasDb(graph);
  // Conservatively we assume there is a reference
  EXPECT_TRUE(aliasDb.mayAlias(a, b));
}

TEST(AliasRegistrationTest, ConservativeWithAliasingAnnotationsShouldError) {
  auto registry = torch::RegisterOperators().op(
      "foo::rand3(Tensor(a) arg1) -> Tensor(b)",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor) -> at::Tensor {
            return at::rand({2, 2});
          })
          .aliasAnalysis(AliasAnalysisKind::CONSERVATIVE));

  const auto rand_op = Symbol::fromQualString("foo::rand3");
  auto graph = std::make_shared<Graph>();
  auto a = graph->addInput();
  graph->insert(rand_op, {a});

  // Registration time is okay, but throw exception when fetch from
  // registration.
  expectThrows<c10::Error>(
      [&graph] { AliasDb aliasDb(graph); },
      "Tried to register operator foo::rand3(Tensor(a) arg1) -> Tensor(b) with aliasing information in the schema but without AliasAnalysisKind::FROM_SCHEMA");
}

TEST(AliasRegistrationTest, ConservativeWithAliasingAnnotationsShouldError2) {
  auto registry = torch::RegisterOperators().op(
      "foo::rand4(Tensor(a) arg1) -> Tensor(a)",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor) -> at::Tensor {
            return at::rand({2, 2});
          })
          .aliasAnalysis(AliasAnalysisKind::CONSERVATIVE));
  const auto rand_op = Symbol::fromQualString("foo::rand4");
  auto graph = std::make_shared<Graph>();
  auto a = graph->addInput();
  graph->insert(rand_op, {a});

  // Registration time is okay, but throw exception when fetch from
  // registration.
  expectThrows<c10::Error>(
      [&graph] { AliasDb aliasDb(graph); },
      "Tried to register operator foo::rand4(Tensor(a) arg1) -> Tensor(a) with aliasing information in the schema but without AliasAnalysisKind::FROM_SCHEMA");
}

TEST(AliasRegistrationTest, FromSchemaWithInferredSchemaShouldError) {
  expectThrows<c10::Error>(
      [] {
        torch::RegisterOperators().op(
            "foo::rand5",
            torch::RegisterOperators::options()
                .catchAllKernel([](at::Tensor) -> at::Tensor {
                  return at::rand({2, 2});
                })
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA));
      },
      "Tried to register operator foo::rand5(Tensor _0) -> Tensor _0 with AliasAnalysisKind::FROM_SCHEMA, but the schema is inferred");
}

TEST(AliasRegistrationTest, FromSchemaInferredPure) {
  auto registry = torch::RegisterOperators().op(
      "foo::rand6(Tensor arg1) -> Tensor",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor) -> at::Tensor {
            return at::rand({2, 2});
          })
          .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA));
  const auto rand_op = Symbol::fromQualString("foo::rand6");
  auto graph = std::make_shared<Graph>();
  auto a = graph->addInput();
  auto b = graph->insert(rand_op, {a});
  AliasDb aliasDb(graph);
  // The schema doesn't contain alias information, which means it's pure
  // (meh!)
  EXPECT_FALSE(aliasDb.mayAlias(a, b));
}

TEST(AliasRegistrationTest, FromSchemaAliased) {
  auto registry = torch::RegisterOperators().op(
      "foo::rand7(Tensor(a) arg1) -> Tensor(a)",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor t) -> at::Tensor { return t * 2; })
          .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA));
  const auto rand_op = Symbol::fromQualString("foo::rand7");

  auto graph = std::make_shared<Graph>();
  auto a = graph->addInput();
  auto b = graph->insert(rand_op, {a});
  AliasDb aliasDb(graph);
  // The schema has an alias reference
  EXPECT_TRUE(aliasDb.mayAlias(a, b));
}

TEST(AliasRegistrationTest, FromSchemaPure) {
  auto registry = torch::RegisterOperators().op(
      "foo::rand8(Tensor(a) arg1) -> Tensor(b)",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor t) -> at::Tensor { return t * 2; })
          .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA));
  const auto rand_op = Symbol::fromQualString("foo::rand8");
  auto graph = std::make_shared<Graph>();
  auto a = graph->addInput();
  auto b = graph->insert(rand_op, {a});
  AliasDb aliasDb(graph);
  // The schema does not have an alias reference
  EXPECT_FALSE(aliasDb.mayAlias(a, b));
}

TEST(AliasRegistrationTest, PureNoSchema) {
  auto registry = torch::RegisterOperators().op(
      "foo::rand9",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor) -> at::Tensor {
            return at::rand({2, 2});
          })
          .aliasAnalysis(AliasAnalysisKind::PURE_FUNCTION));
  const auto rand_op = Symbol::fromQualString("foo::rand9");
  auto graph = std::make_shared<Graph>();
  auto a = graph->addInput();
  auto b = graph->insert(rand_op, {a});
  AliasDb aliasDb(graph);
  // The schema is pure, there cannot be any alias
  EXPECT_FALSE(aliasDb.mayAlias(a, b));
}

TEST(AliasRegistrationTest, PureWithSchema) {
  auto registry = torch::RegisterOperators().op(
      "foo::rand10(Tensor arg1) -> Tensor",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor) -> at::Tensor {
            return at::rand({2, 2});
          })
          .aliasAnalysis(AliasAnalysisKind::PURE_FUNCTION));
  const auto rand_op = Symbol::fromQualString("foo::rand10");
  auto graph = std::make_shared<Graph>();
  auto a = graph->addInput();
  auto b = graph->insert(rand_op, {a});
  AliasDb aliasDb(graph);
  // The schema is pure, there cannot be any alias
  EXPECT_FALSE(aliasDb.mayAlias(a, b));
}

TEST(AliasRegistrationTest, PureWithAnnotationsShouldError) {
  auto registry = torch::RegisterOperators().op(
      "foo::rand11(Tensor(a) arg1) -> Tensor(a)",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor t) -> at::Tensor { return t * 2; })
          .aliasAnalysis(AliasAnalysisKind::PURE_FUNCTION));
  const auto rand_op = Symbol::fromQualString("foo::rand11");
  auto graph = std::make_shared<Graph>();
  auto a = graph->addInput();
  graph->insert(rand_op, {a});

  // Registration time is okay, but throw exception when fetch from
  // registration.
  expectThrows<c10::Error>(
      [&graph] { AliasDb aliasDb(graph); },
      "Tried to register operator foo::rand11(Tensor(a) arg1) -> Tensor(a) with aliasing information in the schema but without AliasAnalysisKind::FROM_SCHEMA");
}

TEST(AliasRegistrationTest, AliasMoveAtenListOp) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  auto graph_string = R"IR(
  graph():
    %x : Tensor = prim::MakeTestTensor()
    %8 : int = prim::Constant[value=0]()
    %5 : int = prim::Constant[value=1]()
    %4 : int = prim::Constant[value=2]()
    %y : Tensor[] = prim::ListConstruct(%x)
    %6 : Tensor = aten::add_(%x, %4, %5)
    %9 : Tensor = aten::cat(%y, %8)
    return (%9))IR";

  torch::jit::parseIR(graph_string, graph.get(), vmap);
  AliasDb aliasDb(graph);

  // bc y.1 has a single used in a single non-aliasing aten op,
  // x is added to y.1 contained elements instead of wildcard set
  EXPECT_TRUE(!aliasDb.mayAlias(vmap["x"], vmap["9"]));

  // write to contained element should prevent move
  EXPECT_TRUE(!aliasDb.moveBeforeTopologicallyValid(
      vmap["y"]->node(), vmap["9"]->node()));
}

TEST(
    AliasRegistrationTest,
    AliasMoveForTupleConstructWithSingleUseAsGraphOutput) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  auto graph_string = R"IR(
  graph():
    %x : Tensor = prim::MakeTestTensor()
    %y : Tensor = prim::MakeTestTensor()
    %z : (Tensor) = prim::TupleConstruct(%x, %y)
    return (%z))IR";

  torch::jit::parseIR(graph_string, graph.get(), vmap);
  AliasDb aliasDb(graph, /*isFrozen=*/false);

  EXPECT_TRUE(!aliasDb.mayAlias(vmap["x"], vmap["y"]));
  EXPECT_TRUE(aliasDb.mayContainAlias(vmap["z"], vmap["x"]));
  EXPECT_TRUE(aliasDb.mayContainAlias(vmap["z"], vmap["y"]));
}

TEST(AliasRegistrationTest, RecursiveSubgraphTupleContainment) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  auto graph_string = R"IR(
  graph():
    %x : Tensor = prim::MakeTestTensor()
    %y : Tensor = prim::MakeTestTensor()
    %z : (Tensor, Tensor) = prim::TupleConstruct(%x, %y)
    return (%z))IR";

  torch::jit::parseIR(graph_string, graph.get(), vmap);
  auto node = vmap["z"]->node();
  auto subgraph =
      SubgraphUtils::createSingletonSubgraph(node, prim::FunctionalGraph);
  AliasDb aliasDb(graph);

  EXPECT_TRUE(aliasDb.mayContainAlias(subgraph->output(), vmap["x"]));
  EXPECT_TRUE(aliasDb.mayContainAlias(subgraph->output(), vmap["y"]));
  EXPECT_TRUE(aliasDb.mayAlias(vmap["x"], vmap["y"]));
}

TEST(AliasRegistrationTest, WildcardAliasForTupleConstructWithUses) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  auto graph_string = R"IR(
  graph():
    %x : Tensor = prim::MakeTestTensor()
    %y : Tensor = prim::MakeTestTensor()
    %z : Tensor = prim::MakeTestTensor()
    %0 : int = prim::Constant[value=0]()
    %a : (Tensor) = prim::TupleConstruct(%x, %y)
    %b : (Tensor) = prim::TupleConstruct(%z)
    %c : Tensor = prim::TupleIndex(%a, %0)
    %d : Tensor = prim::TupleIndex(%b, %0)
    return (%c, %d))IR";

  torch::jit::parseIR(graph_string, graph.get(), vmap);
  AliasDb aliasDb(graph, /*isFrozen=*/false);

  EXPECT_TRUE(aliasDb.mayAlias(vmap["x"], vmap["y"]));
  EXPECT_TRUE(aliasDb.mayAlias(vmap["x"], vmap["z"]));
  EXPECT_TRUE(aliasDb.mayAlias(vmap["y"], vmap["z"]));
  EXPECT_TRUE(aliasDb.mayContainAlias(vmap["a"], vmap["x"]));
  EXPECT_TRUE(aliasDb.mayContainAlias(vmap["a"], vmap["y"]));
  EXPECT_TRUE(aliasDb.mayContainAlias(vmap["a"], vmap["z"]));
  EXPECT_TRUE(aliasDb.mayContainAlias(vmap["b"], vmap["x"]));
  EXPECT_TRUE(aliasDb.mayContainAlias(vmap["b"], vmap["y"]));
  EXPECT_TRUE(aliasDb.mayContainAlias(vmap["b"], vmap["z"]));
}

TEST(AliasRegistrationTest, ATenSplitIntListAliasCheck) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  auto graph_string = R"IR(
  graph():
    %x : Tensor = prim::MakeTestTensor()
    %0 : int = prim::Constant[value=0]()
    %1 : int = prim::Constant[value=1]()
    %2 : int = prim::Constant[value=2]()
    %y : Tensor = aten::add(%x, %x, %0)
    %lengths_list : int[] = prim::tolist(%1, %2)
    %a : Tensor[] = aten::split(%y, %lengths_list, %0)
    %b : Tensor, %c : Tensor = prim::ListUnpack(%a)
    %b1 : Tensor = aten::flatten(%b, %0, %1)
    %c1 : Tensor = aten::flatten(%c, %0, %1)
    %d : Tensor = aten::add(%b1, %c1, %0)
    return (%d))IR";

  torch::jit::parseIR(graph_string, graph.get(), vmap);
  AliasDb aliasDb(graph, /*isFrozen=*/false);

  EXPECT_TRUE(aliasDb.mayAlias(vmap["y"], vmap["b"]));
  EXPECT_TRUE(aliasDb.mayAlias(vmap["y"], vmap["c"]));
  EXPECT_TRUE(aliasDb.mayAlias(vmap["y"], vmap["b1"]));
  EXPECT_TRUE(aliasDb.mayAlias(vmap["y"], vmap["c1"]));
}

TEST(AliasRegistrationTest, ATenSplitIntAliasCheck) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  auto graph_string = R"IR(
  graph():
    %x : Tensor = prim::MakeTestTensor()
    %0 : int = prim::Constant[value=0]()
    %1 : int = prim::Constant[value=1]()
    %2 : int = prim::Constant[value=2]()
    %y : Tensor = aten::add(%x, %x, %0)
    %a : Tensor[] = aten::split(%y, %2, %0)
    %b : Tensor, %c : Tensor = prim::ListUnpack(%a)
    %b1 : Tensor = aten::flatten(%b, %0, %1)
    %c1 : Tensor = aten::flatten(%c, %0, %1)
    %d : Tensor = aten::add(%b1, %c1, %0)
    return (%d))IR";

  torch::jit::parseIR(graph_string, graph.get(), vmap);
  AliasDb aliasDb(graph, /*isFrozen=*/false);

  EXPECT_TRUE(aliasDb.mayAlias(vmap["y"], vmap["b"]));
  EXPECT_TRUE(aliasDb.mayAlias(vmap["y"], vmap["c"]));
  EXPECT_TRUE(aliasDb.mayAlias(vmap["y"], vmap["b1"]));
  EXPECT_TRUE(aliasDb.mayAlias(vmap["y"], vmap["c1"]));
}

TEST(AliasRegistrationTest, PureWithAnnotationsShouldError2) {
  auto registry = torch::RegisterOperators().op(
      "foo::rand12(Tensor(a) arg1) -> Tensor(b)",
      torch::RegisterOperators::options()
          .catchAllKernel([](at::Tensor t) -> at::Tensor { return t * 2; })
          .aliasAnalysis(AliasAnalysisKind::PURE_FUNCTION));
  const auto rand_op = Symbol::fromQualString("foo::rand12");
  auto graph = std::make_shared<Graph>();
  auto a = graph->addInput();
  graph->insert(rand_op, {a});

  // Registration time is okay, but throw exception when fetch from
  // registration.
  expectThrows<c10::Error>(
      [&graph] { AliasDb aliasDb(graph); },
      "Tried to register operator foo::rand12(Tensor(a) arg1) -> Tensor(b) with aliasing information in the schema but without AliasAnalysisKind::FROM_SCHEMA");
}

TEST(IRNonDeterminismTest, Basic) {
  auto graph = std::make_shared<Graph>();
  auto graph_string = R"IR(
  graph():
    %x : Tensor = prim::MakeTestTensor()
    %0 : int = prim::Constant[value=0]()
    %1 : NoneType = prim::Constant()
    %2 : Tensor = aten::bernoulli(%x, %1)
    %3 : Tensor = aten::add(%x, %2, %0)
    return (%3))IR";
  parseIR(graph_string, graph.get());

  for (Node* n : graph->nodes()) {
    if (n->kind() == aten::bernoulli) {
      ASSERT_TRUE(n->isNondeterministic());
    } else {
      ASSERT_FALSE(n->isNondeterministic());
    }
  }
}

TEST(IRNonDeterminismTest, DropoutSpecialCase) {
  auto graph = std::make_shared<Graph>();
  auto graph_string = R"IR(
  graph():
    %x : Tensor = prim::MakeTestTensor()
    %0 : bool = prim::Constant[value=0]()
    %1 : bool = prim::Constant[value=1]()
    %3 : int = prim::Constant[value=1]()
    %3 : float = prim::Constant[value=1.0]()
    %4 : Tensor = aten::dropout(%x, %3, %0)
    %5 : Tensor = aten::dropout(%x, %3, %1)
    %6 : Tensor = aten::add(%4, %5, %3)
    return (%6))IR";
  parseIR(graph_string, graph.get());

  bool train = false;
  for (Node* n : graph->nodes()) {
    if (n->kind() == aten::dropout) {
      if (!train) {
        ASSERT_FALSE(n->isNondeterministic());
        train = true;
      } else {
        ASSERT_TRUE(n->isNondeterministic());
      }
    } else {
      ASSERT_FALSE(n->isNondeterministic());
    }
  }
}

TEST(NonDeterminismBackwardsCompatibility, BackwardsCompatibility) {
  static const std::vector<std::string> nondeterministic_ops = {
      "aten::dropout(Tensor input, float p, bool train) -> Tensor",
      "aten::_fused_dropout(Tensor self, float p, Generator? generator) -> (Tensor, Tensor)",
      "aten::_standard_gamma(Tensor self, Generator? generator) -> Tensor",
      "aten::bernoulli(Tensor self, *, Generator? generator) -> Tensor",
      "aten::bernoulli(Tensor self, float p, *, Generator? generator) -> Tensor",
      "aten::multinomial(Tensor self, int num_samples, bool replacement, *, Generator? generator) -> Tensor",
      "aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)",
      "aten::normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator) -> Tensor",
      "aten::normal.float_Tensor(float mean, Tensor std, *, Generator? generator) -> Tensor",
      "aten::normal.Tensor_float(Tensor mean, float std, *, Generator? generator) -> Tensor",
      "aten::poisson(Tensor self, Generator? generator) -> Tensor",
      "aten::binomial(Tensor count, Tensor prob, Generator? generator=None) -> Tensor",
      "aten::rrelu(Tensor self, Scalar lower, Scalar upper, bool training, Generator? generator) -> Tensor",
      "aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, Generator? generator) -> Tensor",
      "aten::rand(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
      "aten::rand_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randint(int high, int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
      "aten::randint(int low, int high, int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
      "aten::randint_like(Tensor self, int high, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randint_like(Tensor self, int low, int high, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randn(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
      "aten::randn_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randperm(int n, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor"};
  for (const std::string& op : nondeterministic_ops) {
    const c10::FunctionSchema& schema = torch::jit::parseSchema(op);
    const auto& op_handle = c10::Dispatcher::singleton().findOp(
        c10::OperatorName(schema.name(), schema.overload_name()));
    ASSERT_TRUE(op_handle->hasTag(at::Tag::nondeterministic_seeded));
  }
}

TEST(TypeHashing, HashTypes) {
  HashType hasher;

  const TypePtr int_type = IntType::get();
  const TypePtr float_type = FloatType::get();
  ASSERT_NE(hasher(int_type), hasher(float_type));

  const TypePtr int2_type = TupleType::create({int_type, int_type});
  const TypePtr int3_type = TupleType::create({int_type, int_type, int_type});
  ASSERT_NE(hasher(int2_type), hasher(int3_type));
}

} // namespace jit
} // namespace torch
