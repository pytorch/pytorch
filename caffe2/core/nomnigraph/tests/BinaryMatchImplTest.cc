#include <gtest/gtest.h>
#include <set>

#include "test_util.h"

#include "nomnigraph/Converters/Dot.h"
#include "nomnigraph/Graph/Algorithms.h"
#include "nomnigraph/Graph/Graph.h"

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(BinaryMatch, NoMatch) {
  auto graph = createGraph();
  auto matches = nom::algorithm::binaryMatch(
      &graph, [](decltype(graph)::NodeRef n) { return false; });
  EXPECT_EQ(matches.size(), 0);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(BinaryMatch, AllMatch) {
  auto graph = createGraph();
  auto matches = nom::algorithm::binaryMatch(
      &graph, [](decltype(graph)::NodeRef n) { return true; });
  EXPECT_EQ(matches.size(), 1);
  EXPECT_EQ(matches.front().getNodesCount(), graph.getMutableNodes().size());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(BinaryMatch, EmptyGraph) {
  nom::Graph<std::string> graph;
  auto matches = nom::algorithm::binaryMatch(
      &graph, [](decltype(graph)::NodeRef n) { return true; });
  EXPECT_EQ(matches.size(), 0);
}

// We should get this back:
// +---+     +-------+
// | 4 | <-- |   2   |
// +---+     +-------+
//   |         |
//   |         |
//   |         v
//   |       +-------+
//   |       |   3   |
//   |       +-------+
//   |         |
//   |         |
//   |         v
//   |       +-------+
//   +-----> |   6   |
//           +-------+
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(BinaryMatch, Basic) {
  auto graph = createGraph();
  auto matches =
      nom::algorithm::binaryMatch(&graph, [](decltype(graph)::NodeRef n) {
        if (n->data() == "2" || n->data() == "3" || n->data() == "4" ||
            n->data() == "6") {
          return true;
        }
        return false;
      });

  EXPECT_EQ(matches.size(), 1);
  auto match = matches.front();
  EXPECT_EQ(match.getNodesCount(), 4);
  std::set<std::string> exp{"2", "3", "4", "6"};
  for (auto n : match.getNodes()) {
    EXPECT_EQ(exp.count(n->data()), 1);
    exp.erase(n->data());
  }

  // We found all the those nodes.
  EXPECT_EQ(exp.size(), 0);
}

// The interesting bit about this test case is that
// the predicate does not match on 3.
//
// As such, this part of the graph
// +---+     +-------+
// | 4 | <-- |   2   |
// +---+     +-------+
//   |         |
//   |         |
//   |         v
//   |       +-------+
//   |       |   3   |
//   |       +-------+
//   |         |
//   |         |
//   |         v
//   |       +-------+
//   +-----> |   6   |
//           +-------+
//
// should match as { 4, 2 }, { 6 } not { 4, 2, 6 }
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(BinaryMatch, RemovedMiddleNode) {
  auto graph = createGraph();
  auto matches =
      nom::algorithm::binaryMatch(&graph, [](decltype(graph)::NodeRef n) {
        if (n->data() == "2" || n->data() == "4" || n->data() == "6") {
          return true;
        }
        return false;
      });

  EXPECT_EQ(matches.size(), 2);
  auto match1 = matches.front();
  auto match2 = matches.back();

  EXPECT_EQ(match1.getNodesCount(), 2);
  EXPECT_EQ(match2.getNodesCount(), 1);

  std::set<std::string> exp1{"2", "4"};
  std::set<std::string> exp2{"6"};
  for (auto n : match1.getNodes()) {
    EXPECT_EQ(exp1.count(n->data()), 1);
    exp1.erase(n->data());
  }
  for (auto n : match2.getNodes()) {
    EXPECT_EQ(exp2.count(n->data()), 1);
    exp2.erase(n->data());
  }

  EXPECT_EQ(exp1.size(), 0);
  EXPECT_EQ(exp2.size(), 0);
}
