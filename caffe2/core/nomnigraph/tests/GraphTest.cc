#include "test_util.h"

#include "nomnigraph/Converters/Dot.h"
#include "nomnigraph/Graph/Algorithms.h"
#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Support/Casting.h"

#include <gtest/gtest.h>

using TestGraph = nom::Graph<TestClass>;

TEST(Basic, CreateNodeAndEdge) {
  TestGraph g;
  auto n1 = createTestNode(g);
  auto n2 = createTestNode(g);
  g.createEdge(n1, n2);
  EXPECT_TRUE(g.hasNode(n1));
  EXPECT_TRUE(g.hasNode(n2));
}

TEST(Basic, DeleteNode) {
  TestGraph g;
  auto n1 = createTestNode(g);
  auto n2 = createTestNode(g);
  g.createEdge(n1, n2);
  EXPECT_TRUE(g.hasNode(n1));
  g.deleteNode(n1);
  EXPECT_FALSE(g.hasNode(n1));
}

TEST(Basic, DeleteEdge) {
  TestGraph g;
  auto n1 = createTestNode(g);
  auto n2 = createTestNode(g);
  auto e = g.createEdge(n1, n2);
  g.deleteEdge(e);
}

TEST(Basic, ReplaceEdges) {
  TestGraph g;
  auto n1 = createTestNode(g);
  auto n2 = createTestNode(g);
  auto n3 = createTestNode(g);
  auto n4 = createTestNode(g);
  auto n5 = createTestNode(g);

  g.createEdge(n1, n3);
  g.createEdge(n2, n3);
  g.createEdge(n3, n4);
  /*
       1     2        5
          |
          3
          |
          4
   */

  EXPECT_FALSE(g.hasEdge(n1, n5));
  EXPECT_FALSE(g.hasEdge(n2, n5));
  g.replaceInEdges(n3, n5);
  /*
       1     2        3
          |           |
          5           4
   */
  EXPECT_TRUE(g.hasEdge(n1, n5));
  EXPECT_TRUE(g.hasEdge(n2, n5));

  EXPECT_FALSE(g.hasEdge(n5, n4));
  g.replaceOutEdges(n3, n5);
  /*
       1     2        3
          |
          5
          |
          4
   */
  EXPECT_TRUE(g.hasEdge(n5, n4));

  g.replaceNode(n5, n3);
  // Back to the original graph.
  /*
       1     2        5
          |
          3
          |
          4
   */
  EXPECT_TRUE(g.hasEdge(n1, n3));
  EXPECT_TRUE(g.hasEdge(n2, n3));
}

TEST(Basic, HasNode) {
  TestGraph g;
  auto n1 = createTestNode(g);
  auto n2 = createTestNode(g);
  auto n3 = createTestNode(g);
  g.createEdge(n1, n2);
  g.createEdge(n1, n3);
  // Current graph: 1 -> 2 -> 3
  EXPECT_TRUE(g.hasNode(n1));
  EXPECT_TRUE(g.hasNode(n2));
  EXPECT_TRUE(g.hasNode(n3));
  g.swapNodes(n1, n3);
  // Current graph: 3 -> 2 -> 1
  EXPECT_TRUE(g.hasNode(n1));
  EXPECT_TRUE(g.hasNode(n3));
  g.deleteNode(n1);
  // Current graph: 3 -> 2
  EXPECT_FALSE(g.hasNode(n1));
  auto n4 = createTestNode(g);
  EXPECT_TRUE(g.hasNode(n4));
  g.replaceNode(n2, n4);
  // Current graph: 3 -> 4  ,   2
  // replaceNode doesn't delete n2.
  EXPECT_TRUE(g.hasNode(n2));

  // Create a second graph g2, and move the nodes from g2 to g.
  TestClass t5;
  nom::Graph<TestClass> g2;
  nom::Graph<TestClass>::NodeRef n5 = g2.createNode(std::move(t5));
  EXPECT_TRUE(g2.hasNode(n5));

  EXPECT_FALSE(g.hasNode(n5));
  g2.moveNode(n5, &g);
  // Current graph (g1): 3 -> 4, 2, 5
  EXPECT_TRUE(g.hasNode(n5));
}

TEST(Basic, Moves) {
  TestGraph g;
  auto n1 = createTestNode(g);
  auto n2 = createTestNode(g);
  auto n3 = createTestNode(g);
  auto e1 = g.createEdge(n1, n2);
  auto e2 = g.createEdge(n1, n3);
  // Current graph: 1 -> 2 -> 3

  TestGraph g2;
  g.deleteEdge(e2);
  g.moveNode(n1, &g2);
  g.moveNode(n2, &g2);
  g.moveEdge(e1, &g2);
  EXPECT_TRUE(g.isValid());
  EXPECT_TRUE(g2.isValid());
  EXPECT_EQ(g.getMutableNodes().size(), 1);
  EXPECT_EQ(g2.getMutableNodes().size(), 2);
  EXPECT_EQ(g.getMutableEdges().size(), 0);
  EXPECT_EQ(g2.getMutableEdges().size(), 1);
}

TEST(Basic, MoveSubgraph) {
  TestGraph g;
  auto n1 = createTestNode(g);
  auto n2 = createTestNode(g);
  auto n3 = createTestNode(g);
  auto e1 = g.createEdge(n1, n2);
  auto e2 = g.createEdge(n1, n3);
  // Current graph: 1 -> 2 -> 3

  TestGraph g2;

  g.deleteEdge(e2);

  TestGraph::SubgraphType sg;
  sg.addNode(n1);
  sg.addNode(n2);
  sg.addEdge(e1);

  g.moveSubgraph(sg, &g2);

  EXPECT_TRUE(g.isValid());
  EXPECT_TRUE(g2.isValid());
  EXPECT_EQ(g.getMutableNodes().size(), 1);
  EXPECT_EQ(g2.getMutableNodes().size(), 2);
  EXPECT_EQ(g.getMutableEdges().size(), 0);
  EXPECT_EQ(g2.getMutableEdges().size(), 1);
}

TEST(Basic, DotGenerator) {
  TestGraph g;
  auto n1 = createTestNode(g);
  auto n2 = createTestNode(g);
  auto n3 = createTestNode(g);
  auto e12 = g.createEdge(n1, n2);
  g.createEdge(n1, n3);

  std::string dot = nom::converters::convertToDotString(&g, TestNodePrinter);

  // sanity check
  std::string prefix = "digraph G";
  // Full string comparison of the output is not stable because the dot
  // string includes node pointer address as node id. We should switch to
  // comparing full output once dot generator no longer uses addresses.
  EXPECT_TRUE(dot.compare(0, prefix.length(), prefix) == 0);

  TestGraph::SubgraphType sg;
  sg.addNode(n1);
  sg.addNode(n2);
  sg.addEdge(e12);

  // Convert to dot with subgraph clusters.
  dot = nom::converters::convertToDotString<TestGraph>(
      &g, {&sg}, TestNodePrinter);

  // sanity check
  EXPECT_TRUE(dot.compare(0, prefix.length(), prefix) == 0);

  // Convert a single subgraph to dot.
  dot = nom::converters::convertToDotString<TestGraph>(sg, TestNodePrinter);

  // sanity check
  EXPECT_TRUE(dot.compare(0, prefix.length(), prefix) == 0);

  dot =
      nom::converters::convertToDotRecordString<TestGraph>(&g, TestNodePrinter);
  // sanity check
  EXPECT_TRUE(dot.compare(0, prefix.length(), prefix) == 0);
}
