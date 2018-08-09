#include <algorithm>

#include "test_util.h"

#include "nomnigraph/Transformations/SubgraphMatcher.h"

#include <gtest/gtest.h>

namespace nom {

namespace matcher {

using NodeType = std::string;
using Criteria = std::string;

// Node matches a criteria (string) if the data string is the same as the
// criteria. Special case: "*" will match any thing.
struct TestNodeMatch {
  static bool isMatch(
      const nom::Graph<NodeType>::NodeRef& node,
      const Criteria& criteria) {
    return criteria == "*" || criteria == node->data();
  }
};

using TestGraph = Graph<NodeType>;
using TestMatcher = SubgraphMatcher<TestGraph, Criteria, TestNodeMatch>;
using Tree = SubtreeMatchCriteria<Criteria>;

Criteria any() {
  return Criteria("*");
}

// Make it more concise to create matching criteria in dataflow graph.
// For example, operatorTree("opA", ...) will refer to a tree like this:
// ... -> opA -> opA_Output
SubtreeMatchCriteria<Criteria> operatorTree(
    const Criteria& root,
    const std::vector<SubtreeMatchCriteria<Criteria>>& childrenCriteria = {},
    int count = 1) {
  return Tree(any(), {Tree(root, childrenCriteria)}, count);
}

std::map<std::string, std::string> TestGraphNodePrinter(
    TestGraph::NodeRef node) {
  std::map<std::string, std::string> labelMap;
  labelMap["label"] = node->data();
  return labelMap;
};

// Attempts to create a realistic dataflow graph that shows a fuse procedure.
struct DataFlowTestGraph {
  const int numInputs = 4;

  TestGraph graph;

  TestGraph::NodeRef opB;
  TestGraph::NodeRef opF;
  TestGraph::NodeRef opC;
  TestGraph::NodeRef opG;
  TestGraph::NodeRef dataOut;

  // Realistic data flow test graph.
  /*


                          +---------------+
                          |               |
                          |  +---------+  |  +---------+
    +---------------------+  | input_A |  |  | input_B |
    |                        +---------+  |  +---------+
    |                          |          |    |
    |                          |          |    |
    |                          v          v    v
  +---------++---------+     +-------------------------+     +--------+
  | input_C || input_D | --> |           opC           | --> | dataC2 |
  +---------++---------+     +-------------------------+     +--------+
                               |
                               |
                               v
                             +---------+
                             |  dataC  | -+
                             +---------+  |
                               |          |
                               |          |
                               v          |
                             +---------+  |
                             |   opB   | <+
                             +---------+
                               |
                               |
                               v
                             +---------+
                             |  dataB  |
                             +---------+
                               |
                               |
                               v
                             +---------+
                             |   opF   |
                             +---------+
                               |
                               |
                               v
                             +---------+
                             |  dataF  |
                             +---------+
                               |
                               |
                               v
             +---------+     +---------+
             |  dataI  | --> |   opG   |
             +---------+     +---------+
                               |
                               |
                               v
                             +---------+
                             | dataOut |
                             +---------+
  */
  DataFlowTestGraph() {
    opC = graph.createNode("opC");

    for (int i = 0; i < numInputs; i++) {
      auto dataInput = graph.createNode("input");
      graph.createEdge(dataInput, opC);
    }

    auto dataC = graph.createNode("dataC");
    auto dataC2 = graph.createNode("dataC2");
    graph.createEdge(opC, dataC);
    graph.createEdge(opC, dataC2);

    opB = graph.createNode("opB");
    // There are 2 edges
    graph.createEdge(dataC, opB);
    graph.createEdge(dataC, opB);

    auto dataB = graph.createNode("dataB");
    graph.createEdge(opB, dataB);

    opF = graph.createNode("opF");
    graph.createEdge(dataB, opF);

    auto dataF = graph.createNode("dataF");
    graph.createEdge(opF, dataF);

    auto dataI = graph.createNode("dataI");

    opG = graph.createNode("opG");
    graph.createEdge(dataF, opG);
    graph.createEdge(dataI, opG);

    dataOut = graph.createNode("dataOut");
    graph.createEdge(opG, dataOut);

    // Use nom::converters::convertToDotString(&graph, TestGraphNodePrinter)
    // to visualize the graph.
  }
};

SubtreeMatchCriteria<Criteria> DataFlowTestGraphCriteria() {
  // clang-format off
  return Tree(
      Criteria("opG"),{
        operatorTree("opF", {
            // Note: we currently don't enforce that these 2 opC nodes
            // have to be the same.
            operatorTree("opB", {
              operatorTree("opC", {
                Tree(Criteria("input"), {}, Tree::kStarCount)
              }, 2),
            })
        }),
        Tree(any()) // matches dataI
      });
  // clang-format on
}

TestGraph::NodeRef getInNode(TestGraph::NodeRef node, int index) {
  return node->getInEdges()[index]->tail();
}

} // namespace matcher

} // namespace nom

using namespace nom::matcher;

// Simple test cases for node matching criteria.
TEST(SubgraphMatcher, IsNodeMatch) {
  TestGraph graph;
  auto n1 = graph.createNode("Hello");
  auto n2 = graph.createNode("Le");
  graph.createEdge(n1, n2);

  EXPECT_TRUE(TestMatcher::isNodeMatch(n1, "Hello"));
  EXPECT_FALSE(TestMatcher::isNodeMatch(n1, "G"));
  EXPECT_TRUE(TestMatcher::isNodeMatch(n2, "Le"));
  EXPECT_FALSE(TestMatcher::isNodeMatch(n2, "le"));
}

// Test subtree matching with a simple tree graph.
TEST(SubgraphMatcher, IsSubtreeMatch) {
  TestGraph graph;
  auto n1 = graph.createNode("1");
  auto n2 = graph.createNode("2");
  auto n3 = graph.createNode("3");
  auto n4 = graph.createNode("4");
  auto n5 = graph.createNode("5");
  auto n6 = graph.createNode("6");
  auto n7 = graph.createNode("7");

  graph.createEdge(n1, n2);
  graph.createEdge(n2, n3);
  graph.createEdge(n2, n4);
  graph.createEdge(n1, n5);
  graph.createEdge(n5, n6);
  graph.createEdge(n5, n7);
  /*       N1
         /     \
      N2         N5
    /    \     /    \
  N3     N4   N6   N7
  */

  auto subtree = Tree(any(), {Tree(any()), Tree(any())});
  EXPECT_FALSE(TestMatcher::isSubtreeMatch(n1, subtree, false));
  EXPECT_FALSE(TestMatcher::isSubtreeMatch(n4, subtree, false));

  EXPECT_TRUE(TestMatcher::isSubtreeMatch(n2, subtree, false));
  EXPECT_TRUE(TestMatcher::isSubtreeMatch(n5, subtree, false));

  subtree = Tree(Criteria("5"), {Tree(any()), Tree(any())});
  EXPECT_FALSE(TestMatcher::isSubtreeMatch(n2, subtree, false));
  EXPECT_TRUE(TestMatcher::isSubtreeMatch(n5, subtree, false));

  subtree = Tree(any(), {Tree(any()), Tree(Criteria("4"))});
  EXPECT_TRUE(TestMatcher::isSubtreeMatch(n2, subtree, false));
  EXPECT_FALSE(TestMatcher::isSubtreeMatch(n5, subtree, false));

  // Accepts non terminal node
  subtree = Tree(any(), {Tree::nonTerminal(any()), Tree::nonTerminal(any())});
  EXPECT_TRUE(TestMatcher::isSubtreeMatch(n1, subtree, false));
  EXPECT_TRUE(TestMatcher::isSubtreeMatch(n2, subtree, false));
  EXPECT_TRUE(TestMatcher::isSubtreeMatch(n5, subtree, false));
  EXPECT_FALSE(TestMatcher::isSubtreeMatch(n3, subtree, false));
  EXPECT_FALSE(TestMatcher::isSubtreeMatch(n4, subtree, false));
  EXPECT_FALSE(TestMatcher::isSubtreeMatch(n6, subtree, false));
  EXPECT_FALSE(TestMatcher::isSubtreeMatch(n7, subtree, false));
}

// Test subtree matching in which * (repeated) matching of children is allowed.
TEST(SubgraphMatcher, IsSubtreeMatchRepeated) {
  TestGraph graph;
  auto n1 = graph.createNode("1");
  auto n2 = graph.createNode("2");
  auto n3A = graph.createNode("3");
  auto n3B = graph.createNode("3");
  auto n4 = graph.createNode("4");
  auto n5A = graph.createNode("5");
  auto n5B = graph.createNode("5");
  auto n5C = graph.createNode("5");
  graph.createEdge(n1, n2);
  graph.createEdge(n1, n3A);
  graph.createEdge(n1, n3B);
  graph.createEdge(n1, n4);
  graph.createEdge(n1, n4);
  graph.createEdge(n1, n5A);
  graph.createEdge(n1, n5B);
  graph.createEdge(n1, n5C);

  auto subtree = Tree(any(), {Tree(Criteria("2"))});
  EXPECT_FALSE(TestMatcher::isSubtreeMatch(n1, subtree, false));

  subtree = Tree(any(), {Tree(Criteria("2"), {}, Tree::kStarCount)});
  EXPECT_FALSE(TestMatcher::isSubtreeMatch(n1, subtree, false));

  // clang-format off
  subtree = Tree(any(), {
    Tree(Criteria("2")),
    Tree(Criteria("3"), {}, 2),
    Tree(Criteria("4"), {}, 2),
    Tree(Criteria("5"), {}, 3)
  });
  EXPECT_TRUE(TestMatcher::isSubtreeMatch(n1, subtree, false));

  subtree = Tree(any(), {
    Tree(Criteria("2")),
    Tree(Criteria("3"), {}, 2),
    Tree(Criteria("4"), {}, 2),
    Tree(Criteria("5"), {}, Tree::kStarCount)
  });
  EXPECT_TRUE(TestMatcher::isSubtreeMatch(n1, subtree, false));

  subtree = Tree(any(), {
    Tree(Criteria("2")),
    Tree(Criteria("3"), {}, Tree::kStarCount),
    Tree(Criteria("4"), {}, 2),
    Tree(Criteria("5"), {}, Tree::kStarCount)
  });
  EXPECT_TRUE(TestMatcher::isSubtreeMatch(n1, subtree, false));

  subtree = Tree(any(), {
    Tree(Criteria("2")),
    Tree(Criteria("3"), {}, Tree::kStarCount),
  });
  // Fails because there are unmatched edges.
  EXPECT_FALSE(TestMatcher::isSubtreeMatch(n1, subtree, false));

  subtree = Tree(any(), {
    Tree(Criteria("2")),
    Tree(Criteria("3"), {}, 2),
    Tree(Criteria("4")),
    Tree(Criteria("5"), {}, 3)
  });
  // Fails because the count is wrong; we have 2 edges to node N4 while
  // the pattern expects only 1.
  EXPECT_FALSE(TestMatcher::isSubtreeMatch(n1, subtree, false));
  // clang-format on
}

TEST(SubgraphMatcher, IsSubtreeMatchRealistic) {
  auto graph = DataFlowTestGraph();
  auto subtree = DataFlowTestGraphCriteria();

  EXPECT_FALSE(TestMatcher::isSubtreeMatch(graph.opF, subtree));
  EXPECT_FALSE(TestMatcher::isSubtreeMatch(graph.opC, subtree));
  EXPECT_FALSE(TestMatcher::isSubtreeMatch(graph.opB, subtree));
  EXPECT_FALSE(TestMatcher::isSubtreeMatch(graph.dataOut, subtree));

  EXPECT_TRUE(TestMatcher::isSubtreeMatch(graph.opG, subtree));
}

TEST(SubgraphMatcher, ReplaceSubtreeRealistic) {
  auto graph = DataFlowTestGraph();
  auto subtree = DataFlowTestGraphCriteria();

  TestMatcher::replaceSubtree(
      graph.graph, subtree, [](TestGraph& g, TestGraph::NodeRef opG) {
        auto opFused = g.createNode("opFused");

        auto dataF = getInNode(opG, 0);
        auto opF = getInNode(dataF, 0);
        auto dataB = getInNode(opF, 0);
        auto opB = getInNode(dataB, 0);
        auto dataC = getInNode(opB, 0);
        auto opC = getInNode(dataC, 0);

        g.deleteNode(dataF);
        g.replaceNode(opG, opFused);

        auto outEdgesC = opC->getOutEdges();
        g.deleteNode(outEdgesC[0]->head());
        g.deleteNode(outEdgesC[1]->head());
        g.replaceNode(opC, opFused);

        g.deleteNode(opC);
        g.deleteNode(opB);
        g.deleteNode(dataB);
        g.deleteNode(opF);
        g.deleteNode(opG);

        return true;
      });

  // Now the nodes are:
  // - NumInputs input nodes
  // - dataI node
  // - fused node
  // - output node
  auto nodes = graph.graph.getMutableNodes();

  // Test that the graph is transformed as expected.
  EXPECT_EQ(nodes.size(), graph.numInputs + 3);
  TestGraph::NodeRef opFused;
  TestGraph::NodeRef dataI;
  TestGraph::NodeRef dataOut;
  for (auto node : nodes) {
    if (node->data() == "opFused") {
      opFused = node;
    } else if (node->data() == "dataOut") {
      dataOut = node;
    } else if (node->data() == "dataI") {
      dataI = node;
    }
  }

  EXPECT_EQ(getInNode(dataOut, 0), opFused);
  EXPECT_EQ(getInNode(opFused, 0), dataI);
  for (int i = 1; i <= graph.numInputs; i++) {
    EXPECT_EQ(getInNode(opFused, i)->data(), "input");
  }

  // Use nom::converters::convertToDotString(&graph.graph, TestGraphNodePrinter)
  // to visualize. The transformed graph looks like This
  /*

                +---------++---------+
                | input_A || input_D |
                +---------++---------+
                  |          |
                  |          |
                  v          v
+---------+     +--------------------+     +---------+
| input_B | --> |      opFused       | <-- | input_C |
+---------+     +--------------------+     +---------+
                  |          ^
                  |          |
                  v          |
                +---------++---------+
                | dataOut ||  dataI  |
                +---------++---------+
   */
}
