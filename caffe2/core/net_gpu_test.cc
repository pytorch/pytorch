#include <gtest/gtest.h>
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/net.h"
#include "caffe2/core/net_async_base.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/scope_guard.h"

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
    return debug_def().device_option().device_type() == PROTO_CUDA;
  }

  bool SupportsAsyncScheduling() const override {
    return debug_def().device_option().device_type() == PROTO_CUDA;
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

}  // namespace

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
    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    auto* dag = dynamic_cast_if_rtti<AsyncNetBase*>(net.get());
    CHECK_NOTNULL(dag);
    const auto& chains = dag->TEST_execution_chains();
    EXPECT_EQ(chains, expected);
    testExecution(net, net_def.op().size());
  }
}

TEST(NetTest, DISABLED_ChainingForDifferentDevices) {
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
            device_id: 1
          }
        }
)DOC";
  if (HasCudaGPU() && NumCudaDevices() >= 2) {
    checkChainingAndRun(spec, {{0, {0, 1, 2}}, {3, {3}}});
  }
}

} // namespace caffe2
