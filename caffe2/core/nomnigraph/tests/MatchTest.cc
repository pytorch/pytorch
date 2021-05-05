#include "test_util.h"

#include "nomnigraph/Transformations/Match.h"

#include <gtest/gtest.h>

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Match, Basic) {
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

  nom::Graph<std::string> match_graph;
  auto m1 = match_graph.createNode(std::string("1"));
  auto m2 = match_graph.createNode(std::string("2"));
  match_graph.createEdge(m1, m2);

  nom::Match<decltype(graph)> m(match_graph);
  EXPECT_EQ(m.match(graph).size(), 1);
}
