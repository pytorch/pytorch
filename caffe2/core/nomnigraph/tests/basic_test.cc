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
}

TEST(Basic, DeleteNode) {
  TestClass t1;
  TestClass t2;
  nom::Graph<TestClass> g;
  nom::Graph<TestClass>::NodeRef n1 = g.createNode(std::move(t1));
  nom::Graph<TestClass>::NodeRef n2 = g.createNode(std::move(t2));
  g.createEdge(n1, n2);
  g.deleteNode(n1);
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

