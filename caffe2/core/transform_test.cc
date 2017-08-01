#include <google/protobuf/text_format.h>
#include <gtest/gtest.h>
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/transform.h"

namespace caffe2 {

namespace {

using transform::Graph;

/**
 * This dummy transform will find all subgraphs of shape (DummyOp1 ->
 * DummyOp2) and replaces them with (DummyOp3). Simple unit test.
 */
class DummyTransform : public Transform {
 public:
  // Finds all patterns of the form (DummyOp1 -> DummyOp2)
  bool PatternRule(const Graph& g, const std::vector<int>& subgraph, int idx)
      override {
    if (subgraph.size() >= pattern_chain.size()) {
      return false;
    }
    // which index are we trying to append the new node to?
    int pattern_idx = subgraph.size();
    // type doesn't match
    if (g.node(idx).op.type() != pattern_chain[pattern_idx]) {
      return false;
    }
    // not that head, and doesn't have exactly 1 parent
    if (pattern_idx > 0 && g.node(idx).parents.size() != 1) {
      return false;
    }
    // not that tail, and doesn't have exactly 1 child
    if (pattern_idx < pattern_chain.size() - 1 &&
        g.node(idx).children.size() != 1) {
      return false;
    }

    return true;
  }

  // Checks if the subgraph matched is (DummyOp1 -> DummyOp2)
  bool ValidatorRule(const Graph& g, const std::vector<int>& subgraph)
      override {
    if (subgraph.size() == 2) {
      if (g.node(subgraph[0]).op.type() == "DummyOp1" &&
          g.node(subgraph[1]).op.type() == "DummyOp2") {
        return true;
      }
    }
    return false;
  }

  // Replaces a match of (DummyOp1 -> DummyOp2) with (DummyOp3)
  bool ReplaceRule(const std::vector<int>& match, Graph* g_ptr) override {
    CHECK(g_ptr);
    auto& g = *g_ptr;
    OperatorDef new_op;
    new_op.set_type("DummyOp3");
    int new_idx = g.size();

    std::map<int, std::vector<string>> new_op_children;
    std::map<int, std::vector<string>> new_op_parents;

    // for each node parent in the head of the match, connect it to our new node
    for (const auto& edge : g.node(match[0]).parents) {
      int parent = edge.first;
      for (const auto& blob : edge.second) {
        g.node(parent).children[new_idx].push_back(blob);
        new_op_parents[parent].push_back(blob);
      }
    }
    for (const string& blob : g.node(match[0]).op.input()) {
      new_op.add_input(blob);
    }

    // for each child in the tail of the match, connect it to our new node
    for (const auto& edge : g.node(match[1]).children) {
      int child = edge.first;
      for (const auto& blob : edge.second) {
        g.node(child).parents[new_idx].push_back(blob);
        new_op_children[child].push_back(blob);
      }
    }
    for (const string& blob : g.node(match[1]).op.output()) {
      new_op.add_output(blob);
    }

    g.DeactivateSubgraph(match);

    g.push_node(transform::Node(new_op, true, new_op_parents, new_op_children));
    return true;
  }

 private:
  const std::vector<string> pattern_chain = {"DummyOp1", "DummyOp2"};
};

REGISTER_TRANSFORM(DummySwap, DummyTransform)

TEST(TransformTest, TestPatternMatch) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef netdef;

  AddOp(&netdef, "DummyOp1", {"in"}, {"mid1"});
  AddOp(&netdef, "DummyOp2", {"mid1"}, {"mid2"});
  AddOp(&netdef, "DummyOp1", {"mid2"}, {"mid3"});
  AddOp(&netdef, "DummyOp2", {"mid3"}, {"out"});

  auto t = CreateTransform("DummySwap");
  Graph g(netdef);
  auto matches = t->PatternMatch(g);

  EXPECT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0][0], 0);
  EXPECT_EQ(matches[0][1], 1);
  EXPECT_EQ(matches[1][0], 2);
  EXPECT_EQ(matches[1][1], 3);
}

TEST(TransformTest, TestReplacePattern) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef netdef;

  AddOp(&netdef, "DummyOp1", {"in"}, {"mid1"});
  AddOp(&netdef, "DummyOp2", {"mid1"}, {"mid2"});
  AddOp(&netdef, "DummyOp1", {"mid2"}, {"mid3"});
  AddOp(&netdef, "DummyOp2", {"mid3"}, {"out"});

  auto t = CreateTransform("DummySwap");
  Graph g(netdef);
  std::vector<std::vector<int>> matches = {{0, 1}, {2, 3}};
  t->ReplacePattern(matches, &g);

  EXPECT_EQ(g.size(), 6);
  EXPECT_FALSE(g.is_node_active(0));
  EXPECT_FALSE(g.is_node_active(1));
  EXPECT_FALSE(g.is_node_active(2));
  EXPECT_FALSE(g.is_node_active(3));
  EXPECT_TRUE(g.is_node_active(4));
  EXPECT_TRUE(g.is_node_active(5));

  EXPECT_EQ(g.node(4).children.size(), 1);
  EXPECT_EQ(g.node(4).parents.size(), 0);
  EXPECT_TRUE(g.node(4).children.count(5));

  NetDef replaced_netdef = g.GetNetDef();

  EXPECT_EQ(replaced_netdef.op().size(), 2);
  EXPECT_EQ(replaced_netdef.op(0).type(), "DummyOp3");
  EXPECT_EQ(replaced_netdef.op(0).input(0), "in");
  EXPECT_EQ(replaced_netdef.op(1).type(), "DummyOp3");
  EXPECT_EQ(replaced_netdef.op(1).output(0), "out");
}

TEST(TransformTest, TestTransformApply) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef netdef;

  AddOp(&netdef, "DummyOp1", {"in"}, {"mid1"});
  AddOp(&netdef, "DummyOp2", {"mid1"}, {"mid2"});
  AddOp(&netdef, "DummyOp1", {"mid2"}, {"mid3"});
  AddOp(&netdef, "DummyOp2", {"mid3"}, {"out"});

  auto t = CreateTransform("DummySwap");

  NetDef replaced_netdef = t->ApplyTo(netdef);

  EXPECT_EQ(replaced_netdef.op().size(), 2);
  EXPECT_EQ(replaced_netdef.op(0).type(), "DummyOp3");
  EXPECT_EQ(replaced_netdef.op(0).input(0), "in");
  EXPECT_EQ(replaced_netdef.op(1).type(), "DummyOp3");
  EXPECT_EQ(replaced_netdef.op(1).output(0), "out");
}

} // namespace

} // namespace Caffe2
