#include <gtest/gtest.h>
#include "caffe2/core/graph.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

namespace {

using transform::Graph;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::atomic<int> counter;

class GraphDummyOp final : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;
  bool Run(int /* unused */) override {
    counter.fetch_add(1);
    return true;
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(GraphDummyOp1, GraphDummyOp);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(GraphDummyOp1)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{0, 0}, {1, 1}});

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(GraphDummyOp2, GraphDummyOp);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(GraphDummyOp2)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{0, 0}, {1, 1}});

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(GraphDummyOp3, GraphDummyOp);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(GraphDummyOp3)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{0, 0}, {1, 1}});

// Checks if two netdefs are  in terms of type, input, and output.
void compare_netdefs(const NetDef& net_a, const NetDef& net_b) {
  EXPECT_EQ(net_a.op_size(), net_b.op_size());
  for (int i = 0; i < net_a.op_size(); i++) {
    EXPECT_EQ(net_a.op(i).type(), net_b.op(i).type());
    EXPECT_EQ(net_a.op(i).input_size(), net_b.op(i).input_size());
    for (int j = 0; j < net_a.op(i).input_size(); j++) {
      EXPECT_EQ(net_a.op(i).input(j), net_b.op(i).input(j));
    }
    EXPECT_EQ(net_a.op(i).output_size(), net_b.op(i).output_size());
    for (int j = 0; j < net_a.op(i).output_size(); j++) {
      EXPECT_EQ(net_a.op(i).output(j), net_b.op(i).output(j));
    }
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(GraphTest, TestGenerateGraphChain) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef netdef;
  AddOp(&netdef, "GraphDummyOp1", {"in"}, {"mid1"});
  AddOp(&netdef, "GraphDummyOp2", {"mid1"}, {"mid2"});
  AddOp(&netdef, "GraphDummyOp1", {"mid2"}, {"mid3"});
  AddOp(&netdef, "GraphDummyOp2", {"mid3"}, {"out"});
  Graph g(netdef);
  EXPECT_EQ(g.size(), 4);
  for (int i = 0; i < 4; i++) {
    if (i < 3) {
      EXPECT_EQ(g.node(i).children.size(), 1);
      EXPECT_TRUE(g.node(i).children.count(i + 1));
    }
    if (i > 0) {
      EXPECT_EQ(g.node(i).parents.size(), 1);
      EXPECT_TRUE(g.node(i).parents.count(i - 1));
    }
  }
  NetDef retrieved_net = g.GetNetDef();
  compare_netdefs(retrieved_net, netdef);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(GraphTest, TestGenerateGraphChainInPlace) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef netdef;
  AddOp(&netdef, "GraphDummyOp1", {"in"}, {"out"});
  AddOp(&netdef, "GraphDummyOp2", {"out"}, {"out"});
  AddOp(&netdef, "GraphDummyOp1", {"out"}, {"out"});
  AddOp(&netdef, "GraphDummyOp2", {"out"}, {"out"});
  Graph g(netdef);
  EXPECT_EQ(g.size(), 4);
  for (int i = 0; i < 4; i++) {
    if (i < 3) {
      EXPECT_EQ(g.node(i).children.size(), 1);
      EXPECT_TRUE(g.node(i).children.count(i + 1));
    }
    if (i > 0) {
      EXPECT_EQ(g.node(i).parents.size(), 1);
      EXPECT_TRUE(g.node(i).parents.count(i - 1));
    }
  }
  NetDef retrieved_net = g.GetNetDef();
  compare_netdefs(retrieved_net, netdef);
}

// Diamond Graph
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(GraphTest, TestGenerateGraphBranch) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef netdef;

  AddOp(&netdef, "GraphDummyOp1", {"in"}, {"mid1"});
  AddOp(&netdef, "GraphDummyOp2", {"mid1"}, {"mid2"});
  AddOp(&netdef, "GraphDummyOp2", {"mid1"}, {"mid3"});
  AddOp(&netdef, "GraphDummyOp3", {"mid2", "mid3"}, {"out"});

  Graph g(netdef);

  EXPECT_EQ(g.size(), 4);
  EXPECT_EQ(g.node(0).parents.size(), 0);
  EXPECT_EQ(g.node(0).children.size(), 2);
  EXPECT_EQ(g.node(1).parents.size(), 1);
  EXPECT_EQ(g.node(1).children.size(), 1);
  EXPECT_EQ(g.node(2).parents.size(), 1);
  EXPECT_EQ(g.node(2).children.size(), 1);
  EXPECT_EQ(g.node(3).parents.size(), 2);
  EXPECT_EQ(g.node(3).children.size(), 0);

  NetDef retrieved_net = g.GetNetDef();
  compare_netdefs(retrieved_net, netdef);
}

// Double Diamond Graph, reused names
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(GraphTest, TestReusedInputs) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef netdef;

  AddOp(&netdef, "GraphDummyOp1", {"in"}, {"in"});
  AddOp(&netdef, "GraphDummyOp2", {"in"}, {"mid1"});
  AddOp(&netdef, "GraphDummyOp2", {"in"}, {"mid2"});
  AddOp(&netdef, "GraphDummyOp3", {"mid1", "mid2"}, {"in"});
  AddOp(&netdef, "GraphDummyOp2", {"in"}, {"mid1"});
  AddOp(&netdef, "GraphDummyOp2", {"in"}, {"mid2"});
  AddOp(&netdef, "GraphDummyOp3", {"mid1", "mid2"}, {"in"});

  Graph g(netdef);

  EXPECT_EQ(g.size(), 7);
  EXPECT_EQ(g.node(0).parents.size(), 0);
  EXPECT_EQ(g.node(0).children.size(), 2);
  EXPECT_EQ(g.node(1).parents.size(), 1);
  EXPECT_EQ(g.node(1).children.size(), 1);
  EXPECT_EQ(g.node(2).parents.size(), 1);
  EXPECT_EQ(g.node(2).children.size(), 1);
  EXPECT_EQ(g.node(3).parents.size(), 2);
  EXPECT_EQ(g.node(3).children.size(), 2);
  EXPECT_EQ(g.node(4).parents.size(), 1);
  EXPECT_EQ(g.node(4).children.size(), 1);
  EXPECT_EQ(g.node(5).parents.size(), 1);
  EXPECT_EQ(g.node(5).children.size(), 1);
  EXPECT_EQ(g.node(6).parents.size(), 2);
  EXPECT_EQ(g.node(6).children.size(), 0);

  NetDef retrieved_net = g.GetNetDef();
  compare_netdefs(retrieved_net, netdef);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(GraphTest, TestGetPerimeter) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef netdef;

  AddOp(&netdef, "GraphDummyOp1", {"in"}, {"in"});
  AddOp(&netdef, "GraphDummyOp2", {"in"}, {"mid1"});
  AddOp(&netdef, "GraphDummyOp2", {"in"}, {"mid2"});
  AddOp(&netdef, "GraphDummyOp3", {"mid1", "mid2"}, {"in"});
  AddOp(&netdef, "GraphDummyOp2", {"in"}, {"mid1"});
  AddOp(&netdef, "GraphDummyOp2", {"in"}, {"mid2"});
  AddOp(&netdef, "GraphDummyOp1", {"mid1", "mid2"}, {"in"});

  Graph g(netdef);
  std::vector<int> subgraph = {3};

  auto subgraph_input = g.GetSubgraphInput(subgraph);
  EXPECT_EQ(subgraph_input.size(), 2);
  EXPECT_EQ(subgraph_input[0], std::make_pair(string("mid1"), 1));
  EXPECT_EQ(subgraph_input[1], std::make_pair(string("mid2"), 2));

  auto subgraph_output = g.GetSubgraphOutput(subgraph);
  EXPECT_EQ(subgraph_output.size(), 2);
  EXPECT_EQ(subgraph_output[0], std::make_pair(string("in"), 4));
  EXPECT_EQ(subgraph_output[1], std::make_pair(string("in"), 5));
}

} // namespace

} // namespace caffe2
