#include <c10/test/util/Macros.h>
#include <c10/util/NetworkFlow.h>
#include <gtest/gtest.h>
#include <cstdlib>

namespace {

template <typename T>
bool vector_contains(const std::vector<T>& vec, const T& element) {
  for (const auto& e : vec) {
    if (e == element) {
      return true;
    }
  }
  return false;
}

template <typename T>
void expect_vector_contains_subset(
    const std::vector<T>& vec,
    const std::vector<T>& subset) {
  for (auto& element : subset) {
    if (!vector_contains(vec, element)) {
      std::stringstream ss;
      ss << "Failed: checking whether {";
      for (auto& e : subset) {
        ss << e << ", ";
      }
      ss << "} is a subset of {";
      for (auto& e : vec) {
        ss << e << ", ";
      }
      ss << "}, but couldn't find " << element;
      FAIL() << ss.str();
    }
  }
}

namespace test_network_flow {

TEST(NetworkFlowTest, basic) {
  /*
   *     3    1       2
   *      -->b--  ->e--
   *     /  1|  \/     \
   *    / 2  v 2/\   2  \
   *   a---->c-/  ->f---->h
   *    \      2\/      /
   *     \3    1/\    2/
   *      -->d--  ->g--
   *
   * Consider these augmenting paths that constitute a blocking flow:
   * a -> d -> f -> h (capacity 1), saturates d->f
   * a -> c -> g -> h (capacity 2), saturates a->c, c->g, g->h
   * a -> b -> c -> e -> h (capacity 1), saturates b->c
   * a -> b -> f -> h (capacity 1), saturates b->f, f->h
   */
  c10::NetworkFlowGraph g;
  g.add_edge("a", "b", 3); // flow: 2
  g.add_edge("a", "c", 2); // flow: 2
  g.add_edge("a", "d", 3); // flow: 1
  g.add_edge("b", "f", 1); // flow: 1
  g.add_edge("c", "e", 2); // flow: 1
  g.add_edge("c", "g", 2); // flow: 2
  g.add_edge("d", "f", 1); // flow: 1
  g.add_edge("b", "c", 1); // flow: 1
  g.add_edge("e", "h", 2); // flow: 1
  g.add_edge("f", "h", 2); // flow: 2
  g.add_edge("g", "h", 2); // flow: 2
  auto res = g.minimum_cut("a", "h");
  EXPECT_EQ(res.status, c10::MinCutStatus::SUCCESS);
  EXPECT_EQ(res.max_flow, 5);

  // how we "reach" these vertices from "h":
  // h -> e: we see the e->h edge has residual capacity
  // e -> c: we see the c->e edge has residual capacity
  // c -> g: the c->g edge has flow, therefore the g->c edge has residual
  // capacity
  expect_vector_contains_subset(res.unreachable, {"h", "e", "c", "g"});
  expect_vector_contains_subset(res.reachable, {"a", "b", "d", "f"});
}

TEST(NetworkFlowTest, loop) {
  /*                         1
   *                 -------------------
   *                /                   \
   *       1       /    1          1     \    1
   *  a --------> b --------> c -------> d --------> e
   */
  c10::NetworkFlowGraph g;
  g.add_edge("a", "b", 1); // flow: 1
  g.add_edge("b", "c", 1); // flow: 1
  g.add_edge("c", "d", 1); // flow: 1
  g.add_edge("d", "e", 1); // flow: 1
  g.add_edge("d", "b", 1); // flow: 0
  auto res = g.minimum_cut("a", "e");
  EXPECT_EQ(res.status, c10::MinCutStatus::SUCCESS);
  EXPECT_EQ(res.max_flow, 1);

  expect_vector_contains_subset(res.unreachable, {"e"});
  expect_vector_contains_subset(res.reachable, {"a", "b", "c", "d"});
}

TEST(NetworkFlowTest, disconnected_vertices) {
  /*
   *        1
   *  c --------> d
   *
   *       1
   *  a --------> b
   */
  c10::NetworkFlowGraph g;
  g.add_edge("a", "b", 1); // flow: 1
  g.add_edge("c", "d", 1); // flow: 0
  auto res = g.minimum_cut("a", "b");
  EXPECT_EQ(res.status, c10::MinCutStatus::SUCCESS);
  EXPECT_EQ(res.max_flow, 1);

  expect_vector_contains_subset(res.unreachable, {"b"});
  // unintuitively, "c" and "d" get marked as reachable; this mirrors networkx
  // behavior.
  expect_vector_contains_subset(res.reachable, {"a", "c", "d"});
}

TEST(NetworkFlowTest, invalid_endpoints) {
  c10::NetworkFlowGraph g;
  g.add_edge("a", "b", 1);
  auto res = g.minimum_cut("a", "c");
  EXPECT_EQ(res.status, c10::MinCutStatus::INVALID);

  res = g.minimum_cut("c", "b");
  EXPECT_EQ(res.status, c10::MinCutStatus::INVALID);
}

TEST(NetworkFlowTest, unbounded) {
  c10::NetworkFlowGraph g;
  g.add_edge("a", "b", c10::NetworkFlowGraph::INF);
  auto res = g.minimum_cut("a", "b");
  EXPECT_EQ(res.status, c10::MinCutStatus::UNBOUNDED);
}

TEST(NetworkFlowTest, overflow) {
  c10::NetworkFlowGraph g;
  auto flow1 = c10::NetworkFlowGraph::INF / 2;
  auto flow2 = c10::NetworkFlowGraph::INF - flow1;
  g.add_edge("a", "b", flow1);
  g.add_edge("a", "b", flow2);
  auto res = g.minimum_cut("a", "b");
  EXPECT_EQ(res.status, c10::MinCutStatus::OVERFLOW_INF);
}

TEST(NetworkFlowTest, reverse_edge) {
  /*
   *                    100
   *                  --------
   *                 /        \
   *        1       <    1     \
   *  a ---------> b ---------> c
   *
   */
  c10::NetworkFlowGraph g;
  g.add_edge("a", "b", 1);
  g.add_edge("b", "c", 1);
  g.add_edge("c", "a", 100);
  auto res = g.minimum_cut("a", "c");
  EXPECT_EQ(res.status, c10::MinCutStatus::SUCCESS);
  EXPECT_EQ(res.max_flow, 1);

  expect_vector_contains_subset(res.unreachable, {"c"});
  expect_vector_contains_subset(res.reachable, {"a", "b"});
}

} // namespace test_network_flow

} // namespace
