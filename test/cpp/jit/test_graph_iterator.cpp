#include <iostream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/jit.h>
#include <torch/script.h>
#include <torch/torch.h>

namespace torch {
namespace jit {

/**
 * Inverts an unordered map.
 */
template <typename K, typename V>
std::unordered_map<V, K> invert_map(std::unordered_map<K, V>& map) {
  std::unordered_map<V, K> inverted;
  std::for_each(map.begin(), map.end(), [&inverted](const std::pair<K, V>& p) {
    inverted.insert(std::make_pair(p.second, p.first));
  });
  return inverted;
}

/**
 * Traverses the graph using the DepthFirstGraphNodeIterator and
 * returns an array containing the original names in the string
 * graph.
 */
std::vector<std::string> traverse_depth_first(
    std::string graph_string,
    int max_count = 100) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  torch::jit::parseIR(graph_string, graph.get(), vmap);
  auto get_name = invert_map(vmap);

  std::vector<std::string> result;
  DepthFirstGraphNodeIterator graph_it(graph);
  Node* node = graph_it.next();
  int count = 0;
  while (node && count < max_count) {
    std::stringstream buffer;
    std::vector<const torch::jit::Node*> vec;
    node->print(buffer, 0, &vec, false, true, true, false);
    result.push_back(buffer.str());
    node = graph_it.next();
    ++count;
  }
  return result;
}

/** Checks that the iteration order matches the expected/provided order. */
void assert_ordering(
    std::vector<std::string> actual,
    std::initializer_list<std::string> expected_list) {
  auto expected = std::vector<std::string>(expected_list);
  ASSERT_EQ(expected.size(), actual.size())
      << "Got " << actual.size() << " elements (" << actual << ")"
      << " expected " << expected.size() << " elements (" << expected << ")";
  for (unsigned i = 0; i < expected.size(); i++) {
    ASSERT_EQ(expected[i], actual[i])
        << "Difference at index " << i << " in " << actual << " (expected "
        << actual << ")";
  }
}

TEST(GraphIteratorTest, ConstantReturnGraph) {
  const auto graph_string = R"IR(
      graph():
        %1 : int = prim::Constant[value=0]()
        return (%1))IR";
  auto graph = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, graph.get());
  DepthFirstGraphNodeIterator graph_it(graph);
  ASSERT_EQ(graph_it.next()->kind(), prim::Constant);
  ASSERT_EQ(graph_it.next(), nullptr);
}

TEST(GraphIteratorTest, GraphWithParameters) {
  const auto graph_string = R"IR(
      graph(%0 : Double(2)):
        %1 : int = prim::Constant[value=0]()
        return (%0))IR";
  auto ordering = traverse_depth_first(graph_string);
  assert_ordering(ordering, {"%1 : int = prim::Constant[value=0]()"});
}

TEST(GraphIteratorTest, GraphWithIf) {
  const auto graph_string = R"IR(
graph(%a : Tensor):
  %a : int = prim::Constant[value=30]()
  %b : int = prim::Constant[value=10]()
  %c : bool = aten::Bool(%a)
  %d : int = prim::If(%c)
    block0():
      -> (%a)
    block1():
      -> (%b)
  %e : int = prim::Constant[value=20]()
  return (%d)
)IR";
  auto ordering = traverse_depth_first(graph_string);
  assert_ordering(
      ordering,
      {"%1 : int = prim::Constant[value=30]()",
       "%2 : int = prim::Constant[value=10]()",
       "%3 : bool = aten::Bool(%1)",
       "%4 : int = prim::If(%3)",
       "%5 : int = prim::Constant[value=20]()"});
}

TEST(GraphIteratorTest, GraphWithNestedIf) {
  const auto graph_string = R"IR(
graph(%a.1 : Tensor,
      %b.1 : Tensor):
  %2 : int = prim::Constant[value=10]()
  %3 : int = prim::Constant[value=20]()
  %4 : int = prim::Constant[value=30]()
  %5 : int = prim::Constant[value=40]()
  %6 : bool = aten::Bool(%a.1)
  %7 : int = prim::If(%6)
    block0():
      %8 : bool = aten::Bool(%b.1)
      %9 : int = prim::If(%8)
        block0():
          -> (%2)
        block1():
          -> (%3)
      -> (%9)
    block1():
      %10 : bool = aten::Bool(%b.1)
      %11 : int = prim::If(%10)
        block0():
          -> (%4)
        block1():
          -> (%5)
      -> (%11)
  %8 : bool = aten::Bool(%b.1)
  %9 : int = prim::If(%8)
    block0():
      -> (%2)
    block1():
      -> (%3)
  %10 : bool = aten::Bool(%b.1)
  %11 : int = prim::If(%10)
    block0():
      -> (%4)
    block1():
      -> (%5)
  return (%7)
)IR";
  auto ordering = traverse_depth_first(graph_string);
  assert_ordering(
      ordering,
      {"%2 : int = prim::Constant[value=10]()",
       "%3 : int = prim::Constant[value=20]()",
       "%4 : int = prim::Constant[value=30]()",
       "%5 : int = prim::Constant[value=40]()",
       "%6 : bool = aten::Bool(%a.1)",
       "%7 : int = prim::If(%6)",
       "%8 : bool = aten::Bool(%b.1)",
       "%9 : int = prim::If(%8)",
       "%10 : bool = aten::Bool(%b.1)",
       "%11 : int = prim::If(%10)",
       "%12 : bool = aten::Bool(%b.1)",
       "%13 : int = prim::If(%12)",
       "%14 : bool = aten::Bool(%b.1)",
       "%15 : int = prim::If(%14)"});
}

TEST(GraphIteratorTest, GraphWithLoop) {
  const auto graph_string = R"IR(
graph(%a.1 : Tensor):
  %1 : bool = prim::Constant[value=1]()
  %2 : int = prim::Constant[value=10]()
  %3 : int = prim::Constant[value=1]()
  %4 : Tensor = prim::Loop(%2, %1, %a.1)
    block0(%i : int, %b.9 : Tensor):
      %5 : Tensor = aten::add_(%b.9, %3, %3)
      -> (%1, %5)
  %6 : Tensor = prim::Loop(%2, %1, %a.1)
    block0(%i : int, %b.9 : Tensor):
      -> (%1, %4)
  return (%6)
)IR";
  auto ordering = traverse_depth_first(graph_string);
  assert_ordering(
      ordering,
      {"%1 : bool = prim::Constant[value=1]()",
       "%2 : int = prim::Constant[value=10]()",
       "%3 : int = prim::Constant[value=1]()",
       "%4 : Tensor = prim::Loop(%2, %1, %a.1)",
       "%7 : Tensor = aten::add_(%b.10, %3, %3)",
       "%8 : Tensor = prim::Loop(%2, %1, %a.1)"});
}

} // namespace jit
} // namespace torch
