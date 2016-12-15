#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/scope_guard.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

CAFFE2_DECLARE_bool(caffe2_disable_chaining);

namespace caffe2 {

namespace {

// A net test dummy op that does nothing but scaffolding. Here, we
// inherit from OperatorBase because we instantiate on both CPU and
// GPU. In general, you want to only inherit from Operator<Context>.
class NetTestDummyOp final : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;
  bool Run() override {
    return true;
  }
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

unique_ptr<NetBase> CreateNetTestHelper(
    Workspace* ws,
    const vector<string>& input,
    const vector<string>& output) {
  NetDef net_def;
  CAFFE_ENFORCE(google::protobuf::TextFormat::ParseFromString(
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

TEST(NetTest, DeclaredInputInsufficient) {
  Workspace ws;
  ws.CreateBlob("in");
  ASSERT_THROW(
      CreateNetTestHelper(&ws, vector<string>{"unuseful_in"},
                          vector<string>()),
      EnforceNotMet);
}

TEST(NetDeathTest, DeclaredOutputNotMet) {
  Workspace ws;
  ws.CreateBlob("in");
  ASSERT_THROW(
      CreateNetTestHelper(&ws, vector<string>(),
                          vector<string>{"unproduced_out"}),
      EnforceNotMet);
}

void checkChaining(
    const char* spec,
    const DAGNetBase::ExecutionChains& expected) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef net_def;
  CAFFE_ENFORCE(google::protobuf::TextFormat::ParseFromString(spec, &net_def));
  {
    auto old = FLAGS_caffe2_disable_chaining;
    auto g = MakeGuard([&]() { FLAGS_caffe2_disable_chaining = old; });
    FLAGS_caffe2_disable_chaining = false;

    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    auto* dag = dynamic_cast_if_rtti<DAGNetBase*>(net.get());
    CHECK_NOTNULL(dag);
    const auto& chains = dag->TEST_execution_chains();
    EXPECT_TRUE(chains == expected);
  }
}

TEST(NetTest, ChainingForLinearModel) {
  const auto spec = R"DOC(
        name: "example"
        type: "dag"
        external_input: "in"
        op {
          input: "in"
          output: "hidden"
          type: "NetTestDummy"
        }
        op {
          input: "hidden"
          output: "out"
          type: "NetTestDummy"
        }
)DOC";
  checkChaining(spec, {{0, {0, 1}}});
}

TEST(NetTest, ChainingForDifferentDevices) {
  const auto spec = R"DOC(
        name: "example"
        type: "dag"
        external_input: "in"
        op {
          input: "in"
          output: "hidden"
          type: "NetTestDummy"
        }
        op {
          input: "hidden"
          output: "out"
          type: "NetTestDummy"
          device_option {
            device_type: 1
          }
        }
        op {
          input: "out"
          output: "out2"
          type: "NetTestDummy"
          device_option {
            device_type: 1
          }
        }
        op {
          input: "out2"
          output: "out3"
          type: "NetTestDummy"
          device_option {
            device_type: 1
            cuda_gpu_id: 1
          }
        }
)DOC";
  checkChaining(spec, {{0, {0}}, {1, {1, 2}}, {3, {3}}});
}

TEST(NetTest, ChainingForFork) {
  const auto spec = R"DOC(
        name: "example"
        type: "dag"
        external_input: "in"
        op {
          input: "in"
          output: "hidden"
          type: "NetTestDummy"
        }
        op {
          input: "hidden"
          output: "out1"
          type: "NetTestDummy"
        }
        op {
          input: "hidden"
          output: "out2"
          type: "NetTestDummy"
        }
)DOC";
  checkChaining(spec, {{0, {0}}, {1, {1}}, {2, {2}}});
}

// TEST(NetTest, ChainingForJoinWithAncestor) {
//   const auto spec = R"DOC(
//         name: "example"
//         type: "dag"
//         external_input: "in"
//         op {
//           input: "in"
//           output: "hidden"
//           type: "NetTestDummy"
//         }
//         op {
//           input: "hidden"
//           output: "out1"
//           type: "NetTestDummy"
//         }
//         op {
//           input: "hidden"
//           output: "out2"
//           type: "NetTestDummy"
//         }
//         op {
//           input: "hidden"
//           input: "out2"
//           type: "NetTestDummy"
//         }
// )DOC";
//   checkChaining(spec, {{0, {0}}, {1, {1}}, {2, {2, 3}}});
// }

TEST(NetTest, ChainingForForkJoin) {
  const auto spec = R"DOC(
        name: "example"
        type: "dag"
        external_input: "in"
        op {
          input: "in"
          output: "hidden1"
          type: "NetTestDummy"
        }
        op {
          input: "in"
          output: "hidden2"
          type: "NetTestDummy"
        }
        op {
          input: "hidden1"
          input: "hidden2"
          output: "out"
          type: "NetTestDummy"
        }
        op {
          input: "out"
          output: "out2"
          type: "NetTestDummy"
        }
)DOC";
  checkChaining(spec, {{0, {0}}, {1, {1}}, {2, {2, 3}}});
}

} // namespace caffe2
