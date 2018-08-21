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
using TestMatchGraph = MatchGraph<Criteria>;
using TestMatchNode = MatchNode<Criteria>;

// Have just one TestMatchGraph in the tests to make it less verbose to create
// the match graphs.
TestMatchGraph graph;
// Call reset before creating a new TestMatchGraph.
void reset() {
  graph = TestMatchGraph();
}

// Helper methods to make it less verbose to create match graphs.
TestMatchGraph::NodeRef Tree(
    const Criteria& root,
    const std::vector<TestMatchGraph::NodeRef>& children = {},
    int count = 1) {
  return subgraph(graph, root, children, count, false);
}

TestMatchGraph::NodeRef NonTerminal(const Criteria& root, int count = 1) {
  return subgraph(graph, root, {}, count, true);
}

Criteria any() {
  return Criteria("*");
}

// Make it more concise to create matching criteria in dataflow graph.
// For example, operatorTree("opA", ...) will refer to a tree like this:
// ... -> opA -> opA_Output
TestMatchGraph::NodeRef operatorTree(
    const Criteria& root,
    const std::vector<TestMatchGraph::NodeRef>& childrenCriteria = {},
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

TestMatchGraph::NodeRef DataFlowTestGraphCriteria() {
  // clang-format off
  return Tree(
    Criteria("opG"),{
      operatorTree("opF", {
          // Note: we currently don't enforce that these 2 opC nodes
          // have to be the same.
          operatorTree("opB", {
            operatorTree("opC", {
              Tree(Criteria("input"), {}, TestMatchNode::kStarCount)
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

bool isSubgraphMatch(
    TestGraph::NodeRef nodeRef,
    const TestMatchGraph::NodeRef& criteria,
    bool invertGraphTraversal = true) {
  return TestMatcher::isSubgraphMatch(nodeRef, criteria, invertGraphTraversal)
      .isMatch();
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

  reset();
  auto subtree = Tree(any(), {Tree(any()), Tree(any())});
  EXPECT_FALSE(isSubgraphMatch(n1, subtree, false));
  EXPECT_FALSE(isSubgraphMatch(n4, subtree, false));

  EXPECT_TRUE(isSubgraphMatch(n2, subtree, false));
  EXPECT_TRUE(isSubgraphMatch(n5, subtree, false));

  reset();
  subtree = Tree(Criteria("5"), {Tree(any()), Tree(any())});
  EXPECT_FALSE(isSubgraphMatch(n2, subtree, false));
  EXPECT_TRUE(isSubgraphMatch(n5, subtree, false));

  reset();
  subtree = Tree(any(), {Tree(any()), Tree(Criteria("4"))});
  EXPECT_TRUE(isSubgraphMatch(n2, subtree, false));
  EXPECT_FALSE(isSubgraphMatch(n5, subtree, false));

  reset();
  // Accepts non terminal node
  subtree = Tree(any(), {NonTerminal(any()), NonTerminal(any())});
  EXPECT_TRUE(isSubgraphMatch(n1, subtree, false));
  EXPECT_TRUE(isSubgraphMatch(n2, subtree, false));
  EXPECT_TRUE(isSubgraphMatch(n5, subtree, false));
  EXPECT_FALSE(isSubgraphMatch(n3, subtree, false));
  EXPECT_FALSE(isSubgraphMatch(n4, subtree, false));
  EXPECT_FALSE(isSubgraphMatch(n6, subtree, false));
  EXPECT_FALSE(isSubgraphMatch(n7, subtree, false));
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

  reset();
  auto subtree = Tree(any(), {Tree(Criteria("2"))});
  EXPECT_FALSE(isSubgraphMatch(n1, subtree, false));

  reset();
  subtree = Tree(any(), {Tree(Criteria("2"), {}, TestMatchNode::kStarCount)});
  EXPECT_FALSE(isSubgraphMatch(n1, subtree, false));

  reset();
  // clang-format off
  subtree = Tree(any(), {
    Tree(Criteria("2")),
    Tree(Criteria("3"), {}, 2),
    Tree(Criteria("4"), {}, 2),
    Tree(Criteria("5"), {}, 3)
  });
  EXPECT_TRUE(isSubgraphMatch(n1, subtree, false));

  reset();
  subtree = Tree(any(), {
    Tree(Criteria("2")),
    Tree(Criteria("3"), {}, 2),
    Tree(Criteria("4"), {}, 2),
    Tree(Criteria("5"), {}, 4)
  });
  // Failes because exepected 4 matches of n5 but found 3.
  EXPECT_FALSE(isSubgraphMatch(n1, subtree, false));

  reset();
  subtree = Tree(any(), {
    Tree(Criteria("2")),
    Tree(Criteria("3"), {}, 2),
    Tree(Criteria("4"), {}, 2),
    Tree(Criteria("5"), {}, TestMatchNode::kStarCount)
  });
  EXPECT_TRUE(isSubgraphMatch(n1, subtree, false));

  reset();
  subtree = Tree(any(), {
    Tree(Criteria("2")),
    Tree(Criteria("3"), {}, TestMatchNode::kStarCount),
    Tree(Criteria("4"), {}, 2),
    Tree(Criteria("5"), {}, TestMatchNode::kStarCount)
  });
  EXPECT_TRUE(isSubgraphMatch(n1, subtree, false));

  reset();
  subtree = Tree(any(), {
    Tree(Criteria("2")),
    Tree(Criteria("3"), {}, TestMatchNode::kStarCount),
  });
  // Fails because there are unmatched edges.
  EXPECT_FALSE(isSubgraphMatch(n1, subtree, false));

  reset();
  subtree = Tree(any(), {
    Tree(Criteria("2")),
    Tree(Criteria("3"), {}, 2),
    Tree(Criteria("4")),
    Tree(Criteria("5"), {}, 3)
  });
  // Fails because the count is wrong; we have 2 edges to node N4 while
  // the pattern expects only 1.
  EXPECT_FALSE(isSubgraphMatch(n1, subtree, false));
  // clang-format on
}

TEST(SubgraphMatcher, DagMatching) {
  reset();

  // clang-format off
  auto n4match = Tree(Criteria("4"), {
    Tree(Criteria("5"))
  });
  auto subgraph = Tree(Criteria("1"), {
    Tree(Criteria("2"), {
      n4match
    }),
    Tree(Criteria("3"), {
      n4match
    }),
  });
  // clang-format on

  {
    TestGraph graph;
    auto n1 = graph.createNode("1");
    auto n2 = graph.createNode("2");
    auto n3 = graph.createNode("3");
    auto n4 = graph.createNode("4");
    auto n5 = graph.createNode("5");

    graph.createEdge(n1, n2);
    graph.createEdge(n1, n3);
    graph.createEdge(n2, n4);
    graph.createEdge(n3, n4);
    graph.createEdge(n4, n5);

    /*       N1
           /     \
        N2         N3
            \   /
             N4
             |
             N5
    */

    EXPECT_TRUE(isSubgraphMatch(n1, subgraph, false));
  }

  {
    TestGraph graph;
    auto n1 = graph.createNode("1");
    auto n2 = graph.createNode("2");
    auto n3 = graph.createNode("3");
    auto n4A = graph.createNode("4");
    auto n4B = graph.createNode("4");
    auto n5 = graph.createNode("5");

    graph.createEdge(n1, n2);
    graph.createEdge(n1, n3);
    graph.createEdge(n2, n4A);
    graph.createEdge(n3, n4B);
    graph.createEdge(n4A, n5);
    graph.createEdge(n4B, n5);

    /*       N1
           /    \
        N2       N3
        /          \
       N4A        N4B
          \     /
            N5
    */

    // This should fail because n4A and n4B are not the same node.
    EXPECT_FALSE(isSubgraphMatch(n1, subgraph, false));
  }
}

TEST(SubgraphMatcher, DagMatchingMultiEdges) {
  reset();

  // clang-format off
  auto n2match = Tree(Criteria("2"));
  auto subgraph = Tree(Criteria("1"), {
    n2match,
    n2match
  });
  // clang-format on

  {
    TestGraph graph;
    auto n1 = graph.createNode("1");
    auto n2 = graph.createNode("2");

    graph.createEdge(n1, n2);
    graph.createEdge(n1, n2);

    EXPECT_TRUE(isSubgraphMatch(n1, subgraph, false));
  }

  {
    TestGraph graph;
    auto n1 = graph.createNode("1");
    auto n2A = graph.createNode("2");
    auto n2B = graph.createNode("2");

    graph.createEdge(n1, n2A);
    graph.createEdge(n1, n2B);

    EXPECT_FALSE(isSubgraphMatch(n1, subgraph, false));
  }
}

TEST(SubgraphMatcher, DagMatchingRandomLargeGraph) {
  reset();
  // clang-format off
  auto n4match = Tree(any(), {
    NonTerminal(any(), 1)
  });
  auto subtree = Tree(any(), {
    Tree(any(), {
      n4match
    }),
    Tree(any(), {
      n4match
    }),
  });
  // clang-format on
  /*       N1
         /     \
      N2         N3
          \   /
           N4
           |
           N5
  */

  // Look for the diamond pattern in a random large graph.
  TestGraph graph;
  std::vector<nom::Graph<std::string>::NodeRef> nodes;

  // Here we create a test graph and then randomly embed the above
  // pattern into the graph repeatedly (numPatterns times).
  // The actual number of match will be less than numPatterns because the
  // embedded patterns can overlap which become unmatched subgraphs.
  const int numNodes = 50000;
  const int numPatterns = 5000;

  for (int i = 0; i < numNodes; i++) {
    auto node = graph.createNode("Node");
    nodes.emplace_back(node);
  }

  TestRandom random(517);
  for (int i = 0; i < numPatterns; i++) {
    std::vector<int> nodeIdx;
    for (int k = 0; k < 5; k++) {
      nodeIdx.emplace_back(random.nextInt() % numNodes);
    }
    graph.createEdge(nodes[nodeIdx[0]], nodes[nodeIdx[1]]);
    graph.createEdge(nodes[nodeIdx[0]], nodes[nodeIdx[2]]);
    graph.createEdge(nodes[nodeIdx[1]], nodes[nodeIdx[3]]);
    graph.createEdge(nodes[nodeIdx[2]], nodes[nodeIdx[3]]);
    graph.createEdge(nodes[nodeIdx[3]], nodes[nodeIdx[4]]);
  }
  EXPECT_EQ(graph.getEdgesCount(), 5 * numPatterns);

  int countMatch = 0;
  for (auto node : graph.getMutableNodes()) {
    if (isSubgraphMatch(node, subtree, false)) {
      countMatch++;
    }
  }
  EXPECT_EQ(countMatch, 1072);
}

TEST(SubgraphMatcher, IsSubtreeMatchRealistic) {
  reset();
  auto graph = DataFlowTestGraph();
  auto subtree = DataFlowTestGraphCriteria();

  EXPECT_FALSE(isSubgraphMatch(graph.opF, subtree));
  EXPECT_FALSE(isSubgraphMatch(graph.opC, subtree));
  EXPECT_FALSE(isSubgraphMatch(graph.opB, subtree));
  EXPECT_FALSE(isSubgraphMatch(graph.dataOut, subtree));

  EXPECT_TRUE(isSubgraphMatch(graph.opG, subtree));
}

TEST(SubgraphMatcher, ReplaceSubtreeRealistic) {
  reset();
  auto graph = DataFlowTestGraph();
  auto subtree = DataFlowTestGraphCriteria();

  TestMatcher::replaceSubgraph(
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
