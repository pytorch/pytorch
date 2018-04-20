#include <gtest/gtest.h>
#include <set>

#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Graph/Algorithms.h"
#include "nomnigraph/Converters/Dot.h"

/** Our test graph looks like this:
 *           +-------+
 *           | entry |
 *           +-------+
 *             |
 *             |
 *             v
 *           +-------+
 *           |   1   |
 *           +-------+
 *             |
 *             |
 *             v
 * +---+     +-------+
 * | 4 | <-- |   2   |
 * +---+     +-------+
 *   |         |
 *   |         |
 *   |         v
 *   |       +-------+
 *   |       |   3   |
 *   |       +-------+
 *   |         |
 *   |         |
 *   |         v
 *   |       +-------+
 *   +-----> |   6   |
 *           +-------+
 *             |
 *             |
 *             v
 * +---+     +-------+
 * | 5 | --> |   7   |
 * +---+     +-------+
 *             |
 *             |
 *             v
 *           +-------+
 *           | exit  |
 *           +-------+
 *
 * Here is the code used to generate the dot file for it:
 *
 *  auto str = nom::converters::convertToDotString(&graph,
 *    [](nom::Graph<std::string>::NodeRef node) {
 *      std::map<std::string, std::string> labelMap;
 *      labelMap["label"] = node->data();
 *      return labelMap;
 *    });
 */
nom::Graph<std::string> createGraph() {
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
  graph.createEdge(n2, n3);
  graph.createEdge(n2, n4);
  graph.createEdge(n3, n6);
  graph.createEdge(n4, n6);
  graph.createEdge(n6, n7);
  graph.createEdge(n5, n7);
  graph.createEdge(n7, exit);
  return graph;
}

TEST(BinaryMatch, NoMatch) {
  auto graph = createGraph();
  auto matches = nom::algorithm::binaryMatch(&graph,
  [](decltype(graph)::NodeRef n) {
    return false;
  });
  EXPECT_EQ(matches.size(), 0);
}

TEST(BinaryMatch, AllMatch) {
  auto graph = createGraph();
  auto matches = nom::algorithm::binaryMatch(&graph,
  [](decltype(graph)::NodeRef n) {
    return true;
  });
  EXPECT_EQ(matches.size(), 1);
  EXPECT_EQ(matches.front().Nodes.size(),
            graph.getMutableNodes().size());
}

TEST(BinaryMatch, EmptyGraph) {
  nom::Graph<std::string> graph;
  auto matches = nom::algorithm::binaryMatch(&graph,
  [](decltype(graph)::NodeRef n) {
    return true;
  });
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
TEST(BinaryMatch, Basic) {
  auto graph = createGraph();
  auto matches = nom::algorithm::binaryMatch(&graph,
  [](decltype(graph)::NodeRef n) {
    if (n->data() == "2" ||
        n->data() == "3" ||
        n->data() == "4" ||
        n->data() == "6") {
      return true;
    }
    return false;
  });

  EXPECT_EQ(matches.size(), 1);
  auto match = matches.front();
  EXPECT_EQ(match.Nodes.size(), 4);
  std::set<std::string> exp{"2", "3", "4", "6"};
  for (auto n : match.Nodes) {
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
TEST(BinaryMatch, RemovedMiddleNode) {
  auto graph = createGraph();
  auto matches = nom::algorithm::binaryMatch(&graph,
  [](decltype(graph)::NodeRef n) {
    if (n->data() == "2" ||
        n->data() == "4" ||
        n->data() == "6") {
      return true;
    }
    return false;
  });

  EXPECT_EQ(matches.size(), 2);
  auto match1 = matches.front();
  auto match2 = matches.back();

  EXPECT_EQ(match1.Nodes.size(), 2);
  EXPECT_EQ(match2.Nodes.size(), 1);

  std::set<std::string> exp1{"2", "4"};
  std::set<std::string> exp2{"6"};
  for (auto n : match1.Nodes) {
    EXPECT_EQ(exp1.count(n->data()), 1);
    exp1.erase(n->data());
  }
  for (auto n : match2.Nodes) {
    EXPECT_EQ(exp2.count(n->data()), 1);
    exp2.erase(n->data());
  }

  EXPECT_EQ(exp1.size(), 0);
  EXPECT_EQ(exp2.size(), 0);
}
