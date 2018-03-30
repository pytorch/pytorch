#include <gtest/gtest.h>
#include "caffe2/core/net.h"
#include "caffe2/core/net_dag.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/scope_guard.h"

CAFFE2_DECLARE_bool(caffe2_disable_chaining);

namespace caffe2 {

namespace {

static std::atomic<int> counter;

// A net test dummy op that does nothing but scaffolding. Here, we
// inherit from OperatorBase because we instantiate on both CPU and
// GPU. In general, you want to only inherit from Operator<Context>.
class NetTestDummyOp final : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;

  NetTestDummyOp(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws),
        fail_(OperatorBase::GetSingleArgument<bool>("fail", false)) {}

  bool Run(int /* unused */ /*stream_id*/) override {
    if (fail_) {
      return false;
    }
    counter.fetch_add(1);
    return true;
  }

  // Simulate CUDA operator behavior
  bool HasAsyncPart() const override {
    return debug_def().device_option().device_type() == CUDA;
  }

  bool SupportsAsyncScheduling() const override {
    return debug_def().device_option().device_type() == CUDA;
  }

 protected:
  const bool fail_;
};

REGISTER_CPU_OPERATOR(NetTestDummy, NetTestDummyOp);
REGISTER_CUDA_OPERATOR(NetTestDummy, NetTestDummyOp);
REGISTER_CPU_OPERATOR(NetTestDummy2, NetTestDummyOp);
REGISTER_CUDA_OPERATOR(NetTestDummy2, NetTestDummyOp);

OPERATOR_SCHEMA(NetTestDummy)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{0, 0}, {1, 1}});
OPERATOR_SCHEMA(NetTestDummy2)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{1, 0}});

unique_ptr<NetBase> CreateNetTestHelper(
    Workspace* ws,
    const vector<string>& input,
    const vector<string>& output) {
  NetDef net_def;
  {
    auto& op = *(net_def.add_op());
    op.set_type("NetTestDummy");
    op.add_input("in");
    op.add_output("hidden");
  }
  {
    auto& op = *(net_def.add_op());
    op.set_type("NetTestDummy");
    op.add_input("hidden");
    op.add_output("out");
  }

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

void testExecution(std::unique_ptr<NetBase>& net, int num_ops) {
  // Run 100 times
  for (int i = 0; i < 100; i++) {
    counter.exchange(0);
    net.get()->Run();
    ASSERT_EQ(num_ops, counter.load());
  }
}

void checkChainingAndRun(
    const char* spec,
    const dag_utils::ExecutionChains& expected) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef net_def;
  CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));
  {
    net_def.set_num_workers(4);
    auto old = FLAGS_caffe2_disable_chaining;
    auto g = MakeGuard([&]() { FLAGS_caffe2_disable_chaining = old; });
    FLAGS_caffe2_disable_chaining = false;

    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    auto* dag = dynamic_cast_if_rtti<DAGNetBase*>(net.get());
    CHECK_NOTNULL(dag);
    const auto& chains = dag->TEST_execution_chains();
    EXPECT_TRUE(chains == expected);
    testExecution(net, net_def.op().size());
  }
}

void checkNumChainsAndRun(const char* spec, const int expected_num_chains) {
  Workspace ws;

  NetDef net_def;
  CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));
  net_def.set_num_workers(4);

  // Create all external inputs
  for (auto inp : net_def.external_input()) {
    ws.CreateBlob(inp);
  }

  {
    auto old = FLAGS_caffe2_disable_chaining;
    auto g = MakeGuard([&]() { FLAGS_caffe2_disable_chaining = old; });
    FLAGS_caffe2_disable_chaining = false;

    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    auto* dag = dynamic_cast_if_rtti<DAGNetBase*>(net.get());
    CHECK_NOTNULL(dag);
    const auto& chains = dag->TEST_execution_chains();
    EXPECT_EQ(expected_num_chains, chains.size());
    testExecution(net, net_def.op().size());
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
  checkChainingAndRun(spec, {{0, {0, 1}}});
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
  checkChainingAndRun(spec, {{0, {0}}, {1, {1}}, {2, {2}}});
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
//   checkChainingAndRun(spec, {{0, {0}}, {1, {1}}, {2, {2, 3}}});
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
  checkChainingAndRun(spec, {{0, {0}}, {1, {1}}, {2, {2, 3}}});
}

TEST(NetTest, ChainingForwardBackward) {
  const auto spec = R"DOC(
  name: "gpu_0"
  type: "dag"
  op {
    input: "in"
    input: "fc_0_w"
    input: "fc_0_b"
    output: "fc_0"
    name: "0"
    type: "NetTestDummy"
  }
  op {
    input: "fc_0"
    output: "fc_0"
    name: "1"
    type: "NetTestDummy"
  }
  op {
    input: "fc_0"
    input: "fc_1_w"
    input: "fc_1_b"
    output: "fc_1"
    name: "2"
    type: "NetTestDummy"
  }
  op {
    input: "fc_1"
    output: "fc_1"
    name: "3"
    type: "NetTestDummy"
  }
  op {
    input: "fc_1"
    input: "fc_2_w"
    input: "fc_2_b"
    output: "fc_2"
    name: "4"
    type: "NetTestDummy"
  }
  op {
    input: "fc_2"
    output: "fc_2"
    name: "5"
    type: "NetTestDummy"
  }
  op {
    input: "fc_2"
    input: "fc_3_w"
    input: "fc_3_b"
    output: "fc_3"
    name: "6"
    type: "NetTestDummy"
  }
  op {
    input: "fc_3"
    output: "fc_3"
    name: "7"
    type: "NetTestDummy"
  }
  op {
    input: "fc_3"
    input: "fc_4_w"
    input: "fc_4_b"
    output: "fc_4"
    name: "8"
    type: "NetTestDummy"
  }
  op {
    input: "fc_4"
    output: "fc_4"
    name: "9"
    type: "NetTestDummy"
  }
  op {
    input: "fc_4"
    input: "in2"
    output: "LabelCrossEntropy"
    name: "10"
    type: "NetTestDummy"
  }
  op {
    input: "LabelCrossEntropy"
    output: "AveragedLoss"
    name: "11"
    type: "NetTestDummy"
  }
  op {
    input: "AveragedLoss"
    output: "AveragedLoss_autogen_grad"
    name: "12"
    type: "NetTestDummy"
  }
  op {
    input: "LabelCrossEntropy"
    input: "AveragedLoss_autogen_grad"
    output: "LabelCrossEntropy_grad"
    name: "13"
    type: "NetTestDummy"
  }
  op {
    input: "fc_4"
    input: "label"
    input: "LabelCrossEntropy_grad"
    output: "fc_4_grad"
    name: "14"
    type: "NetTestDummy2"
  }
  op {
    input: "fc_4"
    input: "fc_4_grad"
    output: "fc_4_grad"
    name: "15"
    type: "NetTestDummy2"
  }
  op {
    input: "fc_3"
    input: "fc_4_w"
    input: "fc_4_grad"
    output: "fc_4_w_grad"
    output: "fc_4_b_grad"
    output: "fc_3_grad"
    name: "16"
    type: "NetTestDummy"
  }
  op {
    input: "fc_3"
    input: "fc_3_grad"
    output: "fc_3_grad"
    name: "17"
    type: "NetTestDummy2"
  }
  op {
    input: "fc_2"
    input: "fc_3_w"
    input: "fc_3_grad"
    output: "fc_3_w_grad"
    output: "fc_3_b_grad"
    output: "fc_2_grad"
    name: "18"
    type: "NetTestDummy"
  }
  op {
    input: "fc_2"
    input: "fc_2_grad"
    output: "fc_2_grad"
    name: "19"
    type: "NetTestDummy2"
  }
  op {
    input: "fc_1"
    input: "fc_2_w"
    input: "fc_2_grad"
    output: "fc_2_w_grad"
    output: "fc_2_b_grad"
    output: "fc_1_grad"
    name: "20"
    type: "NetTestDummy"
  }
  op {
    input: "fc_1"
    input: "fc_1_grad"
    output: "fc_1_grad"
    name: "21"
    type: "NetTestDummy2"
  }
  op {
    input: "fc_0"
    input: "fc_1_w"
    input: "fc_1_grad"
    output: "fc_1_w_grad"
    output: "fc_1_b_grad"
    output: "fc_0_grad"
    name: "22"
    type: "NetTestDummy"
  }
  op {
    input: "fc_0"
    input: "fc_0_grad"
    output: "fc_0_grad"
    name: "23"
    type: "NetTestDummy2"
  }
  op {
    input: "in"
    input: "fc_0_w"
    input: "fc_0_grad"
    output: "fc_0_w_grad"
    output: "fc_0_b_grad"
    output: "data_grad"
    name: "24"
    type: "NetTestDummy"
  }
  external_input: "in"
  external_input: "in2"
  external_input: "LR"
  external_input: "fc_0_w"
  external_input: "fc_0_b"
  external_input: "fc_1_w"
  external_input: "fc_1_b"
  external_input: "fc_2_w"
  external_input: "fc_2_b"
  external_input: "fc_3_w"
  external_input: "fc_3_b"
  external_input: "fc_4_w"
  external_input: "fc_4_b"
  external_input: "label"
  )DOC";
  checkNumChainsAndRun(spec, 1);
}

TEST(NetTest, ChainingForHogwildModel) {
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
          input: "hidden1"
          output: "mid1"
          type: "NetTestDummy"
        }
        op {
          input: "mid1"
          output: "out1"
          type: "NetTestDummy"
        }
        op {
          input: "in"
          output: "hidden2"
          type: "NetTestDummy"
        }
        op {
          input: "hidden2"
          output: "mid2"
          type: "NetTestDummy"
        }
        op {
          input: "mid2"
          output: "out2"
          type: "NetTestDummy"
        }
)DOC";
  checkNumChainsAndRun(spec, 2);
}

TEST(NetTest, FailingOperator) {
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
          arg {
            name: "fail"
            i: 1
          }
        }
)DOC";

  Workspace ws;
  ws.CreateBlob("in");

  NetDef net_def;
  CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));

  {
    net_def.set_num_workers(4);
    auto old = FLAGS_caffe2_disable_chaining;
    auto g = MakeGuard([&]() { FLAGS_caffe2_disable_chaining = old; });
    FLAGS_caffe2_disable_chaining = false;

    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    for (int i = 0; i < 10; i++) {
      counter.exchange(0);
      ASSERT_FALSE(net.get()->Run());
      ASSERT_EQ(1, counter.load());
    }
  }
}

} // namespace caffe2
