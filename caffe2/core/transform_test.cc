#include <gtest/gtest.h>
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/transform.h"

namespace caffe2 {

namespace {

using transform::Graph;

static std::atomic<int> counter;

class TransformDummyOp final : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;
  bool Run(int /* unused */) override {
    counter.fetch_add(1);
    return true;
  }
};

REGISTER_CPU_OPERATOR(TransformDummyOp1, TransformDummyOp);

OPERATOR_SCHEMA(TransformDummyOp1)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{0, 0}, {1, 1}});

REGISTER_CPU_OPERATOR(TransformDummyOp2, TransformDummyOp);

OPERATOR_SCHEMA(TransformDummyOp2)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{0, 0}, {1, 1}});

REGISTER_CPU_OPERATOR(TransformDummyOp3, TransformDummyOp);

OPERATOR_SCHEMA(TransformDummyOp3)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{0, 0}, {1, 1}});

/**
 * This TransformDummy transform will find all subgraphs of shape
 * (TransformDummyOp1 -> TransformDummyOp2) and replaces them with
 * (TransformDummyOp3). Simple unit test.
 */
class DummyTransform : public Transform {
 public:
  // Finds all patterns of the form (TransformDummyOp1 -> TransformDummyOp2)
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

  // Checks if the subgraph matched is (TransformDummyOp1 -> TransformDummyOp2)
  bool ValidatorRule(const Graph& g, const std::vector<int>& subgraph)
      override {
    if (subgraph.size() == 2) {
      if (g.node(subgraph[0]).op.type() == "TransformDummyOp1" &&
          g.node(subgraph[1]).op.type() == "TransformDummyOp2") {
        return true;
      }
    }
    return false;
  }

  // Replaces a match of (TransformDummyOp1 -> TransformDummyOp2) with
  // (TransformDummyOp3)
  bool ReplaceRule(const std::vector<int>& match, Graph* g_ptr) override {
    CHECK(g_ptr);
    auto& g = *g_ptr;
    OperatorDef new_op;
    new_op.set_type("TransformDummyOp3");
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
  const std::vector<string> pattern_chain = {"TransformDummyOp1",
                                             "TransformDummyOp2"};
};

REGISTER_TRANSFORM(TransformDummySwap, DummyTransform)

TEST(TransformTest, TestPatternMatch) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef netdef;

  AddOp(&netdef, "TransformDummyOp1", {"in"}, {"mid1"});
  AddOp(&netdef, "TransformDummyOp2", {"mid1"}, {"mid2"});
  AddOp(&netdef, "TransformDummyOp1", {"mid2"}, {"mid3"});
  AddOp(&netdef, "TransformDummyOp2", {"mid3"}, {"out"});

  auto t = CreateTransform("TransformDummySwap");
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

  AddOp(&netdef, "TransformDummyOp1", {"in"}, {"mid1"});
  AddOp(&netdef, "TransformDummyOp2", {"mid1"}, {"mid2"});
  AddOp(&netdef, "TransformDummyOp1", {"mid2"}, {"mid3"});
  AddOp(&netdef, "TransformDummyOp2", {"mid3"}, {"out"});

  auto t = CreateTransform("TransformDummySwap");
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
  EXPECT_EQ(replaced_netdef.op(0).type(), "TransformDummyOp3");
  EXPECT_EQ(replaced_netdef.op(0).input(0), "in");
  EXPECT_EQ(replaced_netdef.op(1).type(), "TransformDummyOp3");
  EXPECT_EQ(replaced_netdef.op(1).output(0), "out");
}

TEST(TransformTest, TestTransformApply) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef netdef;
  AddOp(&netdef, "TransformDummyOp1", {"in"}, {"mid1"});
  AddOp(&netdef, "TransformDummyOp2", {"mid1"}, {"mid2"});
  AddOp(&netdef, "TransformDummyOp1", {"mid2"}, {"mid3"});
  AddOp(&netdef, "TransformDummyOp2", {"mid3"}, {"out"});

  NetDef replaced_netdef = ApplyTransform("TransformDummySwap", netdef);

  EXPECT_EQ(replaced_netdef.op().size(), 2);
  EXPECT_EQ(replaced_netdef.op(0).type(), "TransformDummyOp3");
  EXPECT_EQ(replaced_netdef.op(0).input(0), "in");
  EXPECT_EQ(replaced_netdef.op(1).type(), "TransformDummyOp3");
  EXPECT_EQ(replaced_netdef.op(1).output(0), "out");
}

/**
 * Transform with Sorted Order matching.
 * Matches two operators of type TransformDummyOp1, even if disconnected.
 * These operators will be given in execution order,
 * but doesn't need connectivity.
 * Changes them to TransformDummyOp2.
 */
class SortedDummyTransform : public Transform {
 public:
  SortedDummyTransform() {
    SetPatternMatchType(SORTED_WRT_EXECUTION_ORDER);
  }
  bool PatternRule(const Graph& g, const std::vector<int>& subgraph, int idx)
      override {
    if (g.node(idx).op.type() != "TransformDummyOp1") {
      return false;
    }
    return true;
  }
  bool ValidatorRule(const Graph& g, const std::vector<int>& subgraph)
      override {
    if (subgraph.size() == 2) {
      if (g.node(subgraph[0]).op.type() == "TransformDummyOp1" &&
          g.node(subgraph[1]).op.type() == "TransformDummyOp1") {
        return true;
      }
    }
    return false;
  }
  bool ReplaceRule(const std::vector<int>& match, Graph* g_ptr) override {
    CHECK(g_ptr);
    for (const auto& x : match) {
      g_ptr->node(x).op.set_type("TransformDummyOp2");
    }
    return true;
  }
};

REGISTER_TRANSFORM(SortedTransformDummySwap, SortedDummyTransform)

TEST(TransformTest, TestPatternMatchTypeSortedOrder) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef netdef;

  AddOp(&netdef, "TransformDummyOp1", {"in"}, {"mid1"});
  AddOp(&netdef, "TransformDummyOp3", {"mid1"}, {"mid2"});
  AddOp(&netdef, "TransformDummyOp1", {"mid2"}, {"mid3"});
  AddOp(&netdef, "TransformDummyOp3", {"mid3"}, {"out"});

  auto t = CreateTransform("SortedTransformDummySwap");
  NetDef replaced_netdef = t->ApplyTo(netdef);

  EXPECT_EQ(replaced_netdef.op().size(), 4);
  EXPECT_EQ(replaced_netdef.op(0).type(), "TransformDummyOp2");
  EXPECT_EQ(replaced_netdef.op(2).type(), "TransformDummyOp2");
}

/**
 * General subgraph transform.
 * Matches a TransformDummyOp1, and a TransformDummyOp2.
 * Order doesn't matter. Connectedness doesn't matter.
 * Turns them into TransformDummyOp3.
 */
class GeneralDummyTransform : public Transform {
 public:
  GeneralDummyTransform() {
    SetPatternMatchType(GENERAL);
  }
  bool PatternRule(const Graph& g, const std::vector<int>& subgraph, int idx)
      override {
    if (subgraph.size() == 0 && g.node(idx).op.type() == "TransformDummyOp1") {
      return true;
    }
    if (subgraph.size() == 1 && g.node(idx).op.type() == "TransformDummyOp2") {
      return true;
    }
    return false;
  }
  bool ValidatorRule(const Graph& g, const std::vector<int>& subgraph)
      override {
    if (subgraph.size() == 2) {
      if (g.node(subgraph[0]).op.type() == "TransformDummyOp1" &&
          g.node(subgraph[1]).op.type() == "TransformDummyOp2") {
        return true;
      }
    }
    return false;
  }
  bool ReplaceRule(const std::vector<int>& match, Graph* g_ptr) override {
    CHECK(g_ptr);
    for (const auto& x : match) {
      g_ptr->node(x).op.set_type("TransformDummyOp3");
    }
    return true;
  }
};

REGISTER_TRANSFORM(GeneralTransformDummySwap, GeneralDummyTransform)

TEST(TransformTest, TestPatternMatchTypeGeneral) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef netdef;

  AddOp(&netdef, "TransformDummyOp2", {"in"}, {"mid1"});
  AddOp(&netdef, "TransformDummyOp3", {"mid1"}, {"mid2"});
  AddOp(&netdef, "TransformDummyOp1", {"mid2"}, {"mid3"});
  AddOp(&netdef, "TransformDummyOp3", {"mid3"}, {"out"});

  auto t = CreateTransform("GeneralTransformDummySwap");
  NetDef replaced_netdef = t->ApplyTo(netdef);

  EXPECT_EQ(replaced_netdef.op().size(), 4);
  EXPECT_EQ(replaced_netdef.op(0).type(), "TransformDummyOp3");
  EXPECT_EQ(replaced_netdef.op(2).type(), "TransformDummyOp3");
}

class TransformSleepFastOp final : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;
  bool Run(int /* unused */) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    return true;
  }
};

REGISTER_CPU_OPERATOR(TransformSleepFastOp, TransformSleepFastOp);

OPERATOR_SCHEMA(TransformSleepFastOp)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{0, 0}, {1, 1}});

class TransformSleepSlowOp final : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;
  bool Run(int /* unused */) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return true;
  }
};

REGISTER_CPU_OPERATOR(TransformSleepSlowOp, TransformSleepSlowOp);

OPERATOR_SCHEMA(TransformSleepSlowOp)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{0, 0}, {1, 1}});

/**
 * This TransformDummy transform will find all operators of type old_type,
 * and replace them with type new_type.
 */
class TypeSwapTransform : public Transform {
 public:
  // Determine the actual strings through inheriting from derived type.
  explicit TypeSwapTransform(string old_type, string new_type)
      : old_type(old_type), new_type(new_type) {}

  // Really simple, only accept if it's a FastSleepOp, and no match so far.
  bool PatternRule(const Graph& g, const std::vector<int>& subgraph, int idx)
      override {
    if (subgraph.size() == 0 && g.node(idx).op.type() == old_type) {
      return true;
    }
    return false;
  }
  // Checks if the subgraph matched is a FastSleepOp
  bool ValidatorRule(const Graph& g, const std::vector<int>& subgraph)
      override {
    if (subgraph.size() == 1) {
      if (g.node(subgraph[0]).op.type() == old_type) {
        return true;
      }
    }
    return false;
  }
  // Replaces op of original type to new type.
  bool ReplaceRule(const std::vector<int>& match, Graph* g_ptr) override {
    CHECK(g_ptr);
    auto& g = *g_ptr;
    g.node(match[0]).op.set_type(new_type);
    return true;
  }

 private:
  string old_type;
  string new_type;
};

class FastToSlowTransform : public TypeSwapTransform {
 public:
  explicit FastToSlowTransform()
      : TypeSwapTransform("TransformSleepFastOp", "TransformSleepSlowOp") {}
};

REGISTER_TRANSFORM(FastToSlow, FastToSlowTransform);

class SlowToFastTransform : public TypeSwapTransform {
 public:
  explicit SlowToFastTransform()
      : TypeSwapTransform("TransformSleepSlowOp", "TransformSleepFastOp") {}
};

REGISTER_TRANSFORM(SlowToFast, SlowToFastTransform);

TEST(TransformTest, TestApplyTransformIfFasterIsFaster) {
  NetDef init_netdef;
  auto* op = AddOp(&init_netdef, "ConstantFill", {}, {"in"});

  NetDef netdef;
  AddOp(&netdef, "TransformDummyOp1", {"in"}, {"mid"});
  AddOp(&netdef, "TransformSleepSlowOp", {"mid"}, {"out"});
  netdef.add_external_input("in"); // This is important for this function.

  // Make sure the transform would work normally.
  auto transformed_net = ApplyTransform("SlowToFast", netdef);
  EXPECT_EQ(transformed_net.op(1).type(), "TransformSleepFastOp");

  // Should be still transform normally.
  auto mystery_net =
      ApplyTransformIfFaster("SlowToFast", netdef, init_netdef, 5, 10, 1.01);
  EXPECT_EQ(mystery_net.op(1).type(), "TransformSleepFastOp");
}

TEST(TransformTest, TestApplyTransformIfFasterButSlower) {
  NetDef init_netdef;
  auto* op = AddOp(&init_netdef, "ConstantFill", {}, {"in"});

  NetDef netdef;
  AddOp(&netdef, "TransformDummyOp1", {"in"}, {"mid"});
  AddOp(&netdef, "TransformSleepFastOp", {"mid"}, {"out"});
  netdef.add_external_input("in"); // This is important for this function.

  // Make sure the transform would work normally.
  auto transformed_net = ApplyTransform("FastToSlow", netdef);
  EXPECT_EQ(transformed_net.op(1).type(), "TransformSleepSlowOp");

  // Should not actually change!
  auto mystery_net =
      ApplyTransformIfFaster("FastToSlow", netdef, init_netdef, 5, 10, 1.01);
  EXPECT_EQ(mystery_net.op(1).type(), "TransformSleepFastOp");
}

} // namespace

} // namespace Caffe2
