#include "test_util.h"

#include "nomnigraph/Converters/Dot.h"
#include "nomnigraph/Graph/Algorithms.h"
#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Transformations/Match.h"
#include "nomnigraph/Support/Casting.h"

#include <gtest/gtest.h>

TEST(Basic, CreateNodeAndEdge) {
  TestClass t1;
  TestClass t2;
  nom::Graph<TestClass> g;
  nom::Graph<TestClass>::NodeRef n1 = g.createNode(std::move(t1));
  nom::Graph<TestClass>::NodeRef n2 = g.createNode(std::move(t2));
  g.createEdge(n1, n2);
  EXPECT_TRUE(g.hasNode(n1));
  EXPECT_TRUE(g.hasNode(n2));
}

TEST(Basic, DeleteNode) {
  TestClass t1;
  TestClass t2;
  nom::Graph<TestClass> g;
  nom::Graph<TestClass>::NodeRef n1 = g.createNode(std::move(t1));
  nom::Graph<TestClass>::NodeRef n2 = g.createNode(std::move(t2));
  g.createEdge(n1, n2);
  EXPECT_TRUE(g.hasNode(n1));
  g.deleteNode(n1);
  EXPECT_FALSE(g.hasNode(n1));
}

TEST(Basic, DeleteEdge) {
  TestClass t1;
  TestClass t2;
  nom::Graph<TestClass> g;
  nom::Graph<TestClass>::NodeRef n1 = g.createNode(std::move(t1));
  nom::Graph<TestClass>::NodeRef n2 = g.createNode(std::move(t2));
  nom::Graph<TestClass>::EdgeRef e = g.createEdge(n1, n2);
  g.deleteEdge(e);
}

TEST(Basic, HasNode) {
  TestClass t1;
  TestClass t2;
  TestClass t3;
  nom::Graph<TestClass> g;
  nom::Graph<TestClass>::NodeRef n1 = g.createNode(std::move(t1));
  nom::Graph<TestClass>::NodeRef n2 = g.createNode(std::move(t2));
  nom::Graph<TestClass>::NodeRef n3 = g.createNode(std::move(t3));
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
  TestClass t4;
  nom::Graph<TestClass>::NodeRef n4 = g.createNode(std::move(t4));
  EXPECT_TRUE(g.hasNode(n4));
  g.replaceNode(n2, n4);
  // Current graph: 3 -> 4  ,   2
  // replaceNode doesn't delete n2.
  EXPECT_TRUE(g.hasNode(n2));

  // Create a second graph g2, and import the nodes from g2 to g.
  TestClass t5;
  nom::Graph<TestClass> g2;
  nom::Graph<TestClass>::NodeRef n5 = g2.createNode(std::move(t5));
  EXPECT_TRUE(g2.hasNode(n5));

  EXPECT_FALSE(g.hasNode(n5));
  g2.importNode(n5, g);
  // Current graph (g1): 3 -> 4, 2, 5
  EXPECT_TRUE(g.hasNode(n5));
}
