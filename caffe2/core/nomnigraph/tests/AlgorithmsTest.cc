#include "test_util.h"

#include <gtest/gtest.h>

TEST(DominatorTree, Test1) {
  nom::Graph<std::string> graph;
  auto r = graph.createNode(std::string("r"));
  auto a = graph.createNode(std::string("a"));
  auto b = graph.createNode(std::string("b"));
  auto c = graph.createNode(std::string("c"));
  auto d = graph.createNode(std::string("d"));
  auto e = graph.createNode(std::string("e"));
  auto f = graph.createNode(std::string("f"));
  auto g = graph.createNode(std::string("g"));
  auto l = graph.createNode(std::string("l"));
  auto h = graph.createNode(std::string("h"));
  auto i = graph.createNode(std::string("i"));
  auto j = graph.createNode(std::string("j"));
  auto k = graph.createNode(std::string("k"));
  graph.createEdge(r, a);
  graph.createEdge(r, b);
  graph.createEdge(r, c);
  graph.createEdge(c, f);
  graph.createEdge(c, g);
  graph.createEdge(g, j);
  graph.createEdge(g, i);
  graph.createEdge(f, i);
  graph.createEdge(i, k);
  graph.createEdge(k, i);
  graph.createEdge(k, r);
  graph.createEdge(a, d);
  graph.createEdge(b, d);
  graph.createEdge(b, a);
  graph.createEdge(b, e);
  graph.createEdge(d, l);
  graph.createEdge(l, h);
  graph.createEdge(h, k);
  graph.createEdge(h, e);
  graph.createEdge(e, h);

  auto tree = nom::algorithm::dominatorTree(&graph, r);
  auto map = nom::algorithm::immediateDominatorMap(&graph, r);

  EXPECT_EQ(map[j], g);
  EXPECT_EQ(map[g], c);
  EXPECT_EQ(map[f], c);
  EXPECT_EQ(map[l], d);
  EXPECT_EQ(map[a], r);
  EXPECT_EQ(map[b], r);
  EXPECT_EQ(map[c], r);
  EXPECT_EQ(map[d], r);
  EXPECT_EQ(map[e], r);
  EXPECT_EQ(map[h], r);
  EXPECT_EQ(map[i], r);
  EXPECT_EQ(map[k], r);
  auto domFrontMap = nom::algorithm::dominanceFrontierMap(&graph, r);
}

// https://www.seas.harvard.edu/courses/cs252/2011sp/slides/Lec04-SSA.pdf
// using example on page 24
TEST(DominatorTree, Test2) {
  nom::Graph<std::string> graph;
  auto entry = graph.createNode(std::string("entry"));
  auto n1 = graph.createNode(std::string("1"));
  auto n2 = graph.createNode(std::string("2"));
  auto n3 = graph.createNode(std::string("3"));
  auto n4 = graph.createNode(std::string("4"));
  auto n5 = graph.createNode(std::string("5"));
  auto n6 = graph.createNode(std::string("6"));
  auto n7 = graph.createNode(std::string("7"));
  auto exit = graph.createNode(std::string("exit"));
  graph.createEdge(entry, n1);
  graph.createEdge(n1, n2);
  graph.createEdge(n1, n5);
  graph.createEdge(n5, n1);
  graph.createEdge(n2, n3);
  graph.createEdge(n2, n4);
  graph.createEdge(n3, n6);
  graph.createEdge(n4, n6);
  graph.createEdge(n6, n7);
  graph.createEdge(n5, n7);
  graph.createEdge(n7, exit);

  auto domFrontMap = nom::algorithm::dominanceFrontierMap(&graph, entry);
  using noderef = nom::Graph<std::string>::NodeRef;
  std::unordered_map<noderef, std::unordered_set<noderef>> checkMap = {
    {n1, {n1}},
    {n2, {n7}},
    {n3, {n6}},
    {n4, {n6}},
    {n5, {n1, n7}},
    {n6, {n7}}
  };
  for (auto pair : domFrontMap) {
    EXPECT_EQ(pair.second, checkMap[pair.first]);
  }
}

TEST(Subgraph, InduceEdges) {
  auto g = createGraph();
  auto sg = decltype(g)::SubgraphType();
  for (const auto& node : g.getMutableNodes()) {
    sg.addNode(node);
  }

  nom::algorithm::induceEdges(&sg);

  for (const auto& edge : g.getMutableEdges()) {
    EXPECT_TRUE(sg.hasEdge(edge));
  }
}

TEST(Subgraph, InduceEdgesCycle) {
  auto g = createGraphWithCycle();
  auto sg = decltype(g)::SubgraphType();
  for (const auto& node : g.getMutableNodes()) {
    sg.addNode(node);
  }

  nom::algorithm::induceEdges(&sg);

  for (const auto& edge : g.getMutableEdges()) {
    EXPECT_TRUE(sg.hasEdge(edge));
  }
}
