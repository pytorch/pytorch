#include <gtest/gtest.h>
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/transforms/pattern_net_transform.h"

namespace caffe2 {

namespace {

using transform::Graph;

static std::atomic<int> counter;

class DummyCounterOp final : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;
  bool Run(int /* unused */) override {
    counter.fetch_add(1);
    return true;
  }
};

REGISTER_CPU_OPERATOR(DummyCounterOp1, DummyCounterOp);
REGISTER_CUDA_OPERATOR(DummyCounterOp1, DummyCounterOp);

OPERATOR_SCHEMA(DummyCounterOp1)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{0, 0}, {1, 1}});

REGISTER_CPU_OPERATOR(DummyCounterOp2, DummyCounterOp);
REGISTER_CUDA_OPERATOR(DummyCounterOp2, DummyCounterOp);

OPERATOR_SCHEMA(DummyCounterOp2)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{0, 0}, {1, 1}});

REGISTER_CPU_OPERATOR(DummyCounterOp3, DummyCounterOp);
REGISTER_CUDA_OPERATOR(DummyCounterOp3, DummyCounterOp);

OPERATOR_SCHEMA(DummyCounterOp3)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{0, 0}, {1, 1}});

/**
 * P = ---> (Op1) ---> (Op2) --->
 *
 * R = ---> (Op3) ---> (Op3) --->
 */
TEST(PatternNetTransformTest, TestGenerateTransform) {
  Workspace ws;
  ws.CreateBlob("in");

  NetDef netdef;
  AddOp(&netdef, "DummyCounterOp1", {"in"}, {"mid1"});
  AddOp(&netdef, "DummyCounterOp2", {"mid1"}, {"mid2"});
  AddOp(&netdef, "DummyCounterOp1", {"mid2"}, {"mid3"});
  AddOp(&netdef, "DummyCounterOp2", {"mid3"}, {"out"});

  NetDef pdef;
  AddOp(&pdef, "DummyCounterOp1", {"in"}, {"mid"});
  AddOp(&pdef, "DummyCounterOp2", {"mid"}, {"out"});

  NetDef rdef;
  AddOp(&rdef, "DummyCounterOp3", {"in"}, {"new_mid"});
  AddOp(&rdef, "DummyCounterOp3", {"new_mid"}, {"out"});

  PatternNetTransform t(pdef, rdef);

  // test pattern match
  Graph g(netdef);

  auto matches = t.PatternMatch(g);
  EXPECT_EQ(matches.size(), 2);

  t.ReplacePattern(matches, &g);

  EXPECT_EQ(g.size(), 8);
  for (int i = 0; i < 4; i++) {
    EXPECT_FALSE(g.is_node_active(i));
  }
  for (int i = 4; i < 8; i++) {
    EXPECT_TRUE(g.is_node_active(i));
  }

  EXPECT_TRUE(g.node(4).children.count(5));
  EXPECT_TRUE(g.node(5).children.count(6));
  EXPECT_TRUE(g.node(6).children.count(7));

  for (int i = 4; i < 8; i++) {
    EXPECT_EQ(g.node(i).op.input().size(), 1);
    EXPECT_EQ(g.node(i).op.output().size(), 1);
  }

  NetDef replaced_netdef = g.GetNetDef();

  EXPECT_EQ(replaced_netdef.op().size(), 4);
  EXPECT_EQ(replaced_netdef.op(0).type(), "DummyCounterOp3");
  EXPECT_EQ(replaced_netdef.op(1).type(), "DummyCounterOp3");
  EXPECT_EQ(replaced_netdef.op(2).type(), "DummyCounterOp3");
  EXPECT_EQ(replaced_netdef.op(3).type(), "DummyCounterOp3");
}

/**
 * P = ---> (Op1) ---> (Op2) --->
 *
 * R = ---> (Op3) ---> (Op3) --->
 */
TEST(PatternNetTransformTest, TestRepeatedTransform) {
  Workspace ws;
  ws.CreateBlob("in");

  NetDef netdef;
  AddOp(&netdef, "DummyCounterOp1", {"in"}, {"out"});
  AddOp(&netdef, "DummyCounterOp2", {"out"}, {"out"});
  for (int i = 0; i < 99; i++) {
    AddOp(&netdef, "DummyCounterOp1", {"out"}, {"out"});
    AddOp(&netdef, "DummyCounterOp2", {"out"}, {"out"});
  }

  NetDef pdef;
  AddOp(&pdef, "DummyCounterOp1", {"in"}, {"mid"});
  AddOp(&pdef, "DummyCounterOp2", {"mid"}, {"out"});

  NetDef rdef;
  AddOp(&rdef, "DummyCounterOp3", {"in"}, {"new_mid"});
  AddOp(&rdef, "DummyCounterOp3", {"new_mid"}, {"out"});

  PatternNetTransform t(pdef, rdef);

  // test pattern match
  Graph g(netdef);

  auto matches = t.PatternMatch(g);
  EXPECT_EQ(matches.size(), 100);

  t.ReplacePattern(matches, &g);
  NetDef replaced_netdef = g.GetNetDef();

  EXPECT_EQ(replaced_netdef.op_size(), 200);
  for (int i = 0; i < 200; i++) {
    EXPECT_EQ(replaced_netdef.op(i).type(), "DummyCounterOp3");
  }

  unique_ptr<NetBase> net = CreateNet(replaced_netdef, &ws);
  counter.exchange(0);
  net.get()->Run();
  EXPECT_EQ(200, counter.load());
}

/**
 * P = ---> (Op1) ---> (Op3) ---> (Op2) --->
 *            |------> (Op3) -------|
 *
 * R = ---> (Op1) --------------> (Op3) --->
 *          |_(Op3)-->(Op3)-->(Op2)_|
 *
 */
TEST(PatternNetTransformTest, TestHardTransform) {
  Workspace ws;
  ws.CreateBlob("in");

  NetDef netdef;
  // Segment 1 (differs from P because of type)
  AddOp(&netdef, "DummyCounterOp1", {"in"}, {"mid1a_1", "mid1b_1"});
  AddOp(&netdef, "DummyCounterOp2", {"mid1a_1"}, {"mid2a_1"});
  AddOp(&netdef, "DummyCounterOp3", {"mid1b_1"}, {"mid2b_1"});
  AddOp(&netdef, "DummyCounterOp3", {"mid2a_1", "mid2b_1"}, {"out_1"});

  // Segment 2 (differs from P because of structure)
  AddOp(
      &netdef, "DummyCounterOp1", {"out_1"}, {"mid1a_2", "mid1b_2", "mid1c_2"});
  AddOp(&netdef, "DummyCounterOp3", {"mid1a_2"}, {"mid2a_2"});
  AddOp(&netdef, "DummyCounterOp3", {"mid1b_2"}, {"mid2b_2"});
  AddOp(&netdef, "DummyCounterOp3", {"mid1c_2"}, {"mid2c_2"});
  AddOp(
      &netdef, "DummyCounterOp2", {"mid2a_2", "mid2b_2", "mid2c_2"}, {"out_2"});

  // Segment 3
  AddOp(&netdef, "DummyCounterOp1", {"out_2"}, {"mid1a_3", "mid1b_3"});
  AddOp(&netdef, "DummyCounterOp3", {"mid1a_3"}, {"mid2a_3"});
  AddOp(&netdef, "DummyCounterOp3", {"mid1b_3"}, {"mid2b_3"});
  AddOp(&netdef, "DummyCounterOp2", {"mid2a_3", "mid2b_3"}, {"out"});

  NetDef pdef;
  // Should only match Segment 3
  AddOp(&pdef, "DummyCounterOp1", {"sub_in"}, {"mid1a", "mid1b"});
  AddOp(&pdef, "DummyCounterOp3", {"mid1a"}, {"mid2a"});
  AddOp(&pdef, "DummyCounterOp3", {"mid1b"}, {"mid2b"});
  AddOp(&pdef, "DummyCounterOp2", {"mid2a", "mid2b"}, {"sub_out"});

  NetDef rdef;
  AddOp(&rdef, "DummyCounterOp1", {"sub_in"}, {"mid1a", "mid1b"});
  AddOp(&rdef, "DummyCounterOp3", {"mid1b"}, {"mid2b"});
  AddOp(&rdef, "DummyCounterOp3", {"mid2b"}, {"mid3b"});
  AddOp(&rdef, "DummyCounterOp2", {"mid3b"}, {"mid4b"});
  AddOp(&rdef, "DummyCounterOp3", {"mid1a", "mid4b"}, {"sub_out"});

  PatternNetTransform t(pdef, rdef);
  Graph g(netdef);
  EXPECT_EQ(g.size(), 13);

  auto matches = t.PatternMatch(g);
  EXPECT_EQ(matches.size(), 1);

  t.ReplacePattern(matches, &g);
  EXPECT_EQ(g.size(), 18);

  NetDef replaced_netdef = g.GetNetDef();
  EXPECT_EQ(replaced_netdef.op_size(), 14);
  unique_ptr<NetBase> net = CreateNet(replaced_netdef, &ws);
  counter.exchange(0);
  net.get()->Run();
  EXPECT_EQ(14, counter.load());
}

TEST(PatternNetTransformTest, TestGeneralStringMatching) {
  Workspace ws;
  ws.CreateBlob("in");

  NetDef pdef;
  AddOp(&pdef, "*", {"in"}, {"mid"});
  AddOp(&pdef, "DummyOp1|DummyOp2", {"mid"}, {"mid2"});
  AddOp(&pdef, "DummyOp3", {"mid2"}, {"out"});

  NetDef rdef;
  AddOp(&rdef, "DummyOp1", {"in"}, {"out"});

  NetDef netdef;
  AddOp(&netdef, "DummyOp1", {"in"}, {"mid"});
  AddOp(&netdef, "DummyOp3", {"mid"}, {"mid"}); // start of match 1
  AddOp(&netdef, "DummyOp2", {"mid"}, {"mid"});
  AddOp(&netdef, "DummyOp3", {"mid"}, {"mid"}); // end of match 1
  AddOp(&netdef, "DummyOp1", {"mid"}, {"mid"}); // start of match 2
  AddOp(&netdef, "DummyOp1", {"mid"}, {"mid"});
  AddOp(&netdef, "DummyOp3", {"mid"}, {"mid"}); // end of match 2
  AddOp(&netdef, "DummyOp3", {"mid"}, {"out"});

  PatternNetTransform t(pdef, rdef);
  transform::Graph g(netdef);
  auto matches = t.PatternMatch(g);
  EXPECT_EQ(matches.size(), 2);
}

TEST(PatternNetTransformTest, TestDeviceOptionMatching) {
  Workspace ws;
  ws.CreateBlob("in");

  NetDef pdef;
  auto op = AddOp(&pdef, "DummyOp1", {"in"}, {"out"});
  op->mutable_device_option()->set_device_type(CPU);

  NetDef rdef;
  op = AddOp(&rdef, "DummyOp1", {"in"}, {"out"});
  op->mutable_device_option()->set_device_type(CUDA);

  NetDef netdef;
  op = AddOp(&netdef, "DummyOp1", {"in"}, {"mid"});
  op->mutable_device_option()->set_device_type(CPU);
  op = AddOp(&netdef, "DummyOp1", {"mid"}, {"mid"}); // should not match
  op->mutable_device_option()->set_device_type(CUDA);
  op = AddOp(&netdef, "DummyOp1", {"mid"}, {"out"});
  op->mutable_device_option()->set_device_type(CPU);

  PatternNetTransform t(pdef, rdef);
  transform::Graph g(netdef);
  auto matches = t.PatternMatch(g);
  EXPECT_EQ(matches.size(), 2);

  NetDef transformed_net = t.ApplyTo(netdef);
  for (const auto& opdef : transformed_net.op()) {
    EXPECT_TRUE(opdef.has_device_option());
    EXPECT_EQ(opdef.device_option().device_type(), CUDA);
  }
}

TEST(PatternNetTransformTest, TestEngineMatching) {
  Workspace ws;
  ws.CreateBlob("in");

  NetDef pdef;
  auto op = AddOp(&pdef, "DummyOp1", {"in"}, {"out"});
  op->set_engine("FakeEng1|FakeEng2");

  NetDef rdef;
  op = AddOp(&rdef, "DummyOp1", {"in"}, {"out"});
  op->set_engine("FakeEng3");

  NetDef netdef;
  op = AddOp(&netdef, "DummyOp1", {"in"}, {"mid"});
  op->set_engine("FakeEng1");
  op = AddOp(&netdef, "DummyOp1", {"mid"}, {"mid"});
  op->set_engine("FakeEng2");
  op = AddOp(&netdef, "DummyOp1", {"mid"}, {"out"}); // should not match
  op->set_engine("FakeEng3");

  PatternNetTransform t(pdef, rdef);
  transform::Graph g(netdef);
  auto matches = t.PatternMatch(g);
  EXPECT_EQ(matches.size(), 2);

  NetDef transformed_net = t.ApplyTo(netdef);
  for (const auto& opdef : transformed_net.op()) {
    EXPECT_EQ(opdef.engine(), "FakeEng3");
  }
}

TEST(PatternNetTransformTest, TestSingularArgumentMatching) {
  Workspace ws;
  ws.CreateBlob("in");

  NetDef pdef;
  auto op = AddOp(&pdef, "Conv", {"in"}, {"out"});
  {
    auto arg = op->add_arg();
    arg->set_name("stride_w");
    arg->set_i(3);
  }
  {
    auto arg = op->add_arg();
    arg->set_name("stride_h");
    arg->set_i(3);
  }

  NetDef rdef;
  op = AddOp(&rdef, "Conv", {"in"}, {"out"});
  {
    auto arg = op->add_arg();
    arg->set_name("stride_w");
    arg->set_i(5);
  }
  {
    auto arg = op->add_arg();
    arg->set_name("stride_h");
    arg->set_i(5);
  }

  NetDef netdef;
  op = AddOp(&netdef, "Conv", {"in"}, {"mid"}); // Will match
  {
    auto arg = op->add_arg();
    arg->set_name("stride_w");
    arg->set_i(3);
  }
  {
    auto arg = op->add_arg();
    arg->set_name("stride_h");
    arg->set_i(3);
  }
  op = AddOp(&netdef, "Conv", {"mid"}, {"mid"}); // Has bad args, will not match
  {
    auto arg = op->add_arg();
    arg->set_name("stride_w");
    arg->set_i(4);
  }
  {
    auto arg = op->add_arg();
    arg->set_name("stride_h");
    arg->set_i(4);
  }
  op = AddOp(&netdef, "Conv", {"mid"}, {"mid"}); // Has no args, will not match
  op = AddOp(&netdef, "Conv", {"mid"}, {"out"}); // Has different names
  {
    auto arg = op->add_arg();
    arg->set_name("yolo");
    arg->set_i(3);
  }
  {
    auto arg = op->add_arg();
    arg->set_name("swag");
    arg->set_i(3);
  }
  op = AddOp(&netdef, "Conv", {"in"}, {"mid"}); // Will match
  {
    auto arg = op->add_arg();
    arg->set_name("stride_w");
    arg->set_i(3);
  }
  {
    auto arg = op->add_arg();
    arg->set_name("stride_h");
    arg->set_i(3);
  }

  PatternNetTransform t(pdef, rdef);
  t.EnableArgumentMatching();
  transform::Graph g(netdef);
  auto matches = t.PatternMatch(g);
  EXPECT_EQ(matches.size(), 2);
  NetDef transformed_net = t.ApplyTo(netdef);
  EXPECT_EQ(transformed_net.op(0).arg(0).name(), "stride_w");
  EXPECT_EQ(transformed_net.op(0).arg(0).i(), 5);
  EXPECT_EQ(transformed_net.op(0).arg(1).name(), "stride_h");
  EXPECT_EQ(transformed_net.op(0).arg(1).i(), 5);

  EXPECT_EQ(transformed_net.op(4).arg(0).name(), "stride_w");
  EXPECT_EQ(transformed_net.op(4).arg(0).i(), 5);
  EXPECT_EQ(transformed_net.op(4).arg(1).name(), "stride_h");
  EXPECT_EQ(transformed_net.op(4).arg(1).i(), 5);
}

/**
 *           |--(Op2)--|
 * P = --->(Op1)----->(Op3)--->
 *           |--(Op2)--|
 *
 * R = ---> (Op2) --->
 *
 *                |--(Op2)--|
 *           -->(Op1)----->(Op3)---
 *           |    |--(Op2)--|     |
 * G = ---> (Op1)                (Op3) --->
 *           |    |--(Op2)--|     |
 *           -->(Op1)----->(Op3)--
 *                |--(Op2)--|
 *
 * In this test, the two "parallel" modules have intersecting execution orders.
 * We wish to test that the pattern match can still detect the two modules,
 * separately.
 *
 * Furthermore, we will apply the transform to G, TWICE.
 * It should reduce G to a single operator.
 */
TEST(PatternNetTransformTest, TestNonStrictTopographicTransform) {
  Workspace ws;
  ws.CreateBlob("in");

  NetDef netdef;
  // Head
  AddOp(&netdef, "DummyCounterOp1", {"in"}, {"in_1", "in_2"});

  // 2 intertwined segments, each matching P. No strict ordering.
  AddOp(&netdef, "DummyCounterOp1", {"in_1"}, {"m1_1", "m2_1"});
  AddOp(&netdef, "DummyCounterOp1", {"in_2"}, {"m1_2", "m2_2"});
  AddOp(&netdef, "DummyCounterOp2", {"m1_1"}, {"out1_1"});
  AddOp(&netdef, "DummyCounterOp2", {"m1_2"}, {"out1_2"});
  AddOp(&netdef, "DummyCounterOp2", {"m2_1"}, {"out2_1"});
  AddOp(&netdef, "DummyCounterOp2", {"m2_2"}, {"out2_2"});
  AddOp(&netdef, "DummyCounterOp3", {"out1_1", "out2_1"}, {"out1"});
  AddOp(&netdef, "DummyCounterOp3", {"out1_2", "out2_2"}, {"out2"});

  // Tail
  AddOp(&netdef, "DummyCounterOp3", {"out1", "out2"}, {"out"});

  NetDef pdef;
  AddOp(&pdef, "DummyCounterOp1", {"myin"}, {"mid1a", "mid1b"});
  AddOp(&pdef, "DummyCounterOp2", {"mid1a"}, {"mid2a"});
  AddOp(&pdef, "DummyCounterOp2", {"mid1b"}, {"mid2b"});
  AddOp(&pdef, "DummyCounterOp3", {"mid2a", "mid2b"}, {"myout"});

  NetDef rdef;
  AddOp(&rdef, "DummyCounterOp2", {"myin"}, {"myout"});

  PatternNetTransform t(pdef, rdef);

  NetDef replaced_netdef = t.ApplyTo(netdef);
  EXPECT_EQ(replaced_netdef.op_size(), 4);
  unique_ptr<NetBase> net = CreateNet(replaced_netdef, &ws);
  counter.exchange(0);
  net.get()->Run();
  EXPECT_EQ(4, counter.load());

  // apply the transform again
  // the entire net should get transformed this time
  NetDef double_transformed_net = t.ApplyTo(replaced_netdef);
  EXPECT_EQ(double_transformed_net.op_size(), 1);
}

/**
 *      --->(Op1)----->(Op2)--->
 *            |          ^
 * P =        |----------|
 *            |          v
 *      --->(Op1)----->(Op2)--->
 *
 * R =  ---> (Op3) --->
 *
 * G = P -> P
 *
 * In this test, we fuse a subgraph with two inputs and two outputs, into one
 * operator.
 *
 * This will ensure that we can allow a single edge to represent
 * multiple blob names (the input and output of R are both 2 blobs).
 *
 * This will also ensure that patternmatch can traverse "backwards", from a node
 * to its parent.
 *
 * Furthermore, this tests for repeat matches, since matching on either of the
 * first two Op1 nodes will produce a match, but they are identical.
 * So, the pattern should match 4 times, but only be replaced twice.
 */
TEST(PatternNetTransformTest, TestMultiInputOutputTransform) {
  Workspace ws;
  ws.CreateBlob("in1");
  ws.CreateBlob("in2");

  NetDef netdef;
  AddOp(&netdef, "DummyCounterOp1", {"in1"}, {"in1"}); // has 2 children
  AddOp(&netdef, "DummyCounterOp1", {"in2"}, {"in2"}); // has 2 children
  AddOp(&netdef, "DummyCounterOp2", {"in1", "in2"}, {"mid1"});
  AddOp(&netdef, "DummyCounterOp2", {"in1", "in2"}, {"mid2"});
  AddOp(&netdef, "DummyCounterOp1", {"mid1"}, {"mid1"}); // has 2 children
  AddOp(&netdef, "DummyCounterOp1", {"mid2"}, {"mid2"}); // has 2 children
  AddOp(&netdef, "DummyCounterOp2", {"mid1", "mid2"}, {"out1"});
  AddOp(&netdef, "DummyCounterOp2", {"mid1", "mid2"}, {"out2"});

  NetDef pdef;
  AddOp(&pdef, "DummyCounterOp1", {"subin1"}, {"subin1"}); // has 2 children
  AddOp(&pdef, "DummyCounterOp1", {"subin2"}, {"subin2"}); // has 2 children
  AddOp(&pdef, "DummyCounterOp2", {"subin1", "subin2"}, {"subout1"});
  AddOp(&pdef, "DummyCounterOp2", {"subin1", "subin2"}, {"subout2"});

  NetDef rdef;
  AddOp(&rdef, "DummyCounterOp3", {"subin1", "subin2"}, {"subout1", "subout2"});

  PatternNetTransform t(pdef, rdef);
  Graph g(netdef);

  NetDef replaced_netdef = t.ApplyTo(netdef);
  EXPECT_EQ(replaced_netdef.op_size(), 2);
  unique_ptr<NetBase> net = CreateNet(replaced_netdef, &ws);
  counter.exchange(0);
  net.get()->Run();
  EXPECT_EQ(2, counter.load());
}

} // namespace

} // namespace Caffe2
