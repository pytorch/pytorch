#include <gtest/gtest.h>

#include "test_util.h"

#include "nomnigraph/Graph/Graph.h"

TEST(Tarjans, Simple) {
    TestClass t1;
    TestClass t2;
    nom::Graph<TestClass, int> g;
    nom::Graph<TestClass, int>::NodeRef n1 = g.createNode(std::move(t1));
    nom::Graph<TestClass, int>::NodeRef n2 = g.createNode(std::move(t2));
    g.createEdge(n1, n2);
    g.createEdge(n2, n1);
    auto sccs = nom::algorithm::tarjans(&g);
    EXPECT_EQ(sccs.size(), 1);
}

TEST(Tarjans, DAG) {
  auto graph = createGraph();
  auto sccs = nom::algorithm::tarjans(&graph);
  EXPECT_EQ(sccs.size(), 9);
}

TEST(Tarjans, Cycle) {
  auto graph = createGraphWithCycle();
  auto sccs = nom::algorithm::tarjans(&graph);
  EXPECT_EQ(sccs.size(), 8);
}

TEST(Tarjans, Random) {
  nom::Graph<TestClass, int> g;
  std::vector<nom::Graph<TestClass, int>::NodeRef> nodes;
  for (auto i = 0; i < 10; ++i) {
    TestClass t;
    nodes.emplace_back(g.createNode(std::move(t)));
  }
  for (auto i = 0; i < 30; ++i) {
    int ri1 = rand() % nodes.size();
    int ri2 = rand() % nodes.size();
    g.createEdge(nodes[ri1], nodes[ri2]);
  }

  auto sccs = nom::algorithm::tarjans(&g);
  EXPECT_GE(sccs.size(), 1);
}

