#include <gtest/gtest.h>

#include "test_util.h"

#include "nomnigraph/Graph/Graph.h"

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Tarjans, Simple) {
  TestClass t1;
  TestClass t2;
  nom::Graph<TestClass> g;
  nom::Graph<TestClass>::NodeRef n1 = g.createNode(std::move(t1));
  nom::Graph<TestClass>::NodeRef n2 = g.createNode(std::move(t2));
  g.createEdge(n1, n2);
  g.createEdge(n2, n1);
  auto sccs = nom::algorithm::tarjans(&g);
  EXPECT_EQ(sccs.size(), 1);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Tarjans, WithEdgeStorage) {
  TestClass t1;
  TestClass t2;
  nom::Graph<TestClass, TestClass> g;
  nom::Graph<TestClass, TestClass>::NodeRef n1 = g.createNode(std::move(t1));
  nom::Graph<TestClass, TestClass>::NodeRef n2 = g.createNode(std::move(t2));
  g.createEdge(n1, n2, TestClass());
  g.createEdge(n2, n1, TestClass());
  auto sccs = nom::algorithm::tarjans(&g);
  EXPECT_EQ(sccs.size(), 1);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Tarjans, DAG) {
  auto graph = createGraph();
  auto sccs = nom::algorithm::tarjans(&graph);
  EXPECT_EQ(sccs.size(), 9);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Tarjans, Cycle) {
  auto graph = createGraphWithCycle();
  auto sccs = nom::algorithm::tarjans(&graph);
  EXPECT_EQ(sccs.size(), 8);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Tarjans, Random) {
  nom::Graph<TestClass> g;
  std::vector<nom::Graph<TestClass>::NodeRef> nodes;
  for (auto i = 0; i < 10; ++i) {
    TestClass t;
    nodes.emplace_back(g.createNode(std::move(t)));
  }
  for (auto i = 0; i < 30; ++i) {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,clang-analyzer-security.insecureAPI.rand)
    int ri1 = rand() % nodes.size();
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,clang-analyzer-security.insecureAPI.rand)
    int ri2 = rand() % nodes.size();
    g.createEdge(nodes[ri1], nodes[ri2]);
  }

  auto sccs = nom::algorithm::tarjans(&g);
  EXPECT_GE(sccs.size(), 1);
}
