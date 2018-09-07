#include "test_util.h"

#include "nomnigraph/Converters/Dot.h"
#include "nomnigraph/Graph/Algorithms.h"
#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Support/Casting.h"

#include <gtest/gtest.h>

using TestGraph = nom::Graph<TestClass>;
TestGraph::NodeRef createTestNode(TestGraph& g) {
  return g.createNode(TestClass());
}

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
}
