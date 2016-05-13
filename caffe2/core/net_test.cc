#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

namespace caffe2 {

namespace {

// A net test dummy op that does nothing but scaffolding.
class NetTestDummyOp final : public OperatorBase {
 public:
  NetTestDummyOp(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws) {}
  bool Run() override { return true; }
};

REGISTER_CPU_OPERATOR(NetTestDummy, NetTestDummyOp);
REGISTER_CUDA_OPERATOR(NetTestDummy, NetTestDummyOp);

OPERATOR_SCHEMA(NetTestDummy).NumInputs(0, INT_MAX).NumOutputs(0, INT_MAX);

const char kExampleNetDefString[] =
"  name: \"example\""
"  op {"
"    input: \"in\""
"    output: \"hidden\""
"    type: \"NetTestDummy\""
"  }"
"  op {"
"    input: \"hidden\""
"    output: \"out\""
"    type: \"NetTestDummy\""
"  }";

NetBase* CreateNetTestHelper(
    Workspace* ws,
    const vector<string>& input,
    const vector<string>& output) {
  NetDef net_def;
  CAFFE_CHECK(google::protobuf::TextFormat::ParseFromString(
    kExampleNetDefString, &net_def));
  for (const auto& name : input) {
    net_def.add_external_input(name);
  }
  for (const auto& name : output) {
    net_def.add_external_output(name);
  }
  return CreateNet(net_def, ws);
}

}  // namespace

TEST(NetTest, ConstructionNoDeclaredInputOutput) {
  Workspace ws;
  ws.CreateBlob("in");
  unique_ptr<NetBase> net(
      CreateNetTestHelper(&ws, vector<string>(), vector<string>()));
  EXPECT_TRUE(net.get() != nullptr);
}

TEST(NetTest, ConstructionDeclaredInput) {
  Workspace ws;
  ws.CreateBlob("in");
  unique_ptr<NetBase> net(
      CreateNetTestHelper(&ws, vector<string>{"in"}, vector<string>()));
  EXPECT_TRUE(net.get() != nullptr);
}

TEST(NetTest, ConstructionDeclaredOutput) {
  Workspace ws;
  ws.CreateBlob("in");
  unique_ptr<NetBase> net(
      CreateNetTestHelper(&ws, vector<string>(), vector<string>{"out"}));
  EXPECT_TRUE(net.get() != nullptr);
}

TEST(NetDeathTest, DeclaredInputInsufficient) {
  Workspace ws;
  ws.CreateBlob("in");
  EXPECT_DEATH(
      CreateNetTestHelper(&ws, vector<string>{"unuseful_in"}, vector<string>()),
      "");
}

TEST(NetDeathTest, DeclaredOutputNotMet) {
  Workspace ws;
  ws.CreateBlob("in");
  EXPECT_DEATH(
      CreateNetTestHelper(&ws, vector<string>(),
                          vector<string>{"unproduced_out"}),
      "");
}

}  // namespace caffe2
