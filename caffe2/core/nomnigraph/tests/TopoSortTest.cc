#include <gtest/gtest.h>

#include "test_util.h"

#include "nomnigraph/Graph/Graph.h"

using GraphT = nom::Graph<TestClass>;
using TopoSortT = nom::algorithm::TopoSort<GraphT>;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TopoSort, Simple) {
  GraphT g;
  auto n1 = createTestNode(g);
  auto n2 = createTestNode(g);
  g.createEdge(n1, n2);
  auto res = nom::algorithm::topoSort(&g);
  EXPECT_EQ(res.status, TopoSortT::Result::OK);
  EXPECT_EQ(res.nodes.size(), 2);
  EXPECT_EQ(res.nodes[0], n1);
  EXPECT_EQ(res.nodes[1], n2);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TopoSort, DAG) {
  GraphT g;
  auto n1 = createTestNode(g);
  auto n2 = createTestNode(g);
  auto n3 = createTestNode(g);
  auto n4 = createTestNode(g);
  g.createEdge(n1, n2);
  g.createEdge(n1, n3);
  g.createEdge(n2, n4);
  g.createEdge(n3, n4);
  auto res = nom::algorithm::topoSort(&g);
  EXPECT_EQ(res.status, TopoSortT::Result::OK);
  EXPECT_EQ(res.nodes.size(), 4);
  auto i1 = std::find(res.nodes.begin(), res.nodes.end(), n1);
  auto i2 = std::find(res.nodes.begin(), res.nodes.end(), n2);
  auto i3 = std::find(res.nodes.begin(), res.nodes.end(), n3);
  auto i4 = std::find(res.nodes.begin(), res.nodes.end(), n4);
  ASSERT_TRUE(i1 != res.nodes.end());
  ASSERT_TRUE(i2 != res.nodes.end());
  ASSERT_TRUE(i3 != res.nodes.end());
  ASSERT_TRUE(i4 != res.nodes.end());
  ASSERT_LT(i1, i2);
  ASSERT_LT(i1, i3);
  ASSERT_LT(i2, i4);
  ASSERT_LT(i3, i4);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TopoSort, Cycle1) {
  GraphT g;
  auto n1 = createTestNode(g);
  auto n2 = createTestNode(g);
  g.createEdge(n1, n2);
  g.createEdge(n2, n1);
  auto res = nom::algorithm::topoSort(&g);
  EXPECT_EQ(res.status, TopoSortT::Result::CYCLE);
  EXPECT_EQ(res.nodes.size(), 0);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TopoSort, Cycle2) {
  GraphT g;
  auto n1 = createTestNode(g);
  auto n2 = createTestNode(g);
  auto n3 = createTestNode(g);
  auto n4 = createTestNode(g);
  g.createEdge(n1, n2);
  g.createEdge(n2, n3);
  g.createEdge(n3, n4);
  g.createEdge(n4, n2);
  auto res = nom::algorithm::topoSort(&g);
  EXPECT_EQ(res.status, TopoSortT::Result::CYCLE);
  EXPECT_EQ(res.nodes.size(), 0);
}
