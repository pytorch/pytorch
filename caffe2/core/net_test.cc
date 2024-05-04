#include <gtest/gtest.h>
#include "c10/util/StringUtil.h"
#include "caffe2/core/net.h"
#include "caffe2/core/net_async_scheduling.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/scope_guard.h"

#include <google/protobuf/text_format.h>

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
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
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

} // namespace

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
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(
      CreateNetTestHelper(&ws, vector<string>{"unuseful_in"}, vector<string>()),
      EnforceNotMet);
}

TEST(NetDeathTest, DeclaredOutputNotMet) {
  Workspace ws;
  ws.CreateBlob("in");
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(
      CreateNetTestHelper(
          &ws, vector<string>(), vector<string>{"unproduced_out"}),
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
    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    auto* dag = dynamic_cast_if_rtti<AsyncNetBase*>(net.get());
    TORCH_CHECK_NOTNULL(dag);
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
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (auto inp : net_def.external_input()) {
    ws.CreateBlob(inp);
  }

  {
    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    auto* dag = dynamic_cast_if_rtti<AsyncNetBase*>(net.get());
    TORCH_CHECK_NOTNULL(dag);
    const auto& chains = dag->TEST_execution_chains();
    EXPECT_EQ(expected_num_chains, chains.size());
    testExecution(net, net_def.op().size());
  }
}

TEST(NetTest, DISABLED_ChainingForLinearModel) {
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

TEST(NetTest, DISABLED_ChainingForFork) {
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

TEST(NetTest, DISABLED_ChainingForForkJoin) {
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

TEST(NetTest, DISABLED_ChainingForwardBackward) {
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

TEST(NetTest, DISABLED_ChainingForHogwildModel) {
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

TEST(NetTest, DISABLED_FailingOperator) {
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
    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    for (int i = 0; i < 10; i++) {
      counter.exchange(0);
      bool run_result = false;
      try {
        run_result = net->Run();
      } catch (const std::exception&) {
        // async_scheduling would throw
      }
      ASSERT_FALSE(run_result);

      ASSERT_EQ(1, counter.load());
    }
  }
}

const int kTestPoolSize = 4;

class ExecutorHelperDummyOp final : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;

  ExecutorHelperDummyOp(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws) {}

  bool Run(int /* unused */ /*stream_id*/) override {
    auto helper = GetExecutorHelper();
    CAFFE_ENFORCE(helper);
    auto pool = helper->GetPool(device_option());
    CAFFE_ENFORCE(pool);
    auto pool_size = pool->size();
    CAFFE_ENFORCE_EQ(pool_size, kTestPoolSize);
    return true;
  }
};

REGISTER_CPU_OPERATOR(ExecutorHelperDummy, ExecutorHelperDummyOp);

OPERATOR_SCHEMA(ExecutorHelperDummy);

TEST(NetTest, OperatorWithExecutorHelper) {
  const auto spec = R"DOC(
        name: "example"
        type: "async_scheduling"
        op {
          type: "ExecutorHelperDummy"
        }
)DOC";

  NetDef net_def;
  CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));

  Workspace ws;
  net_def.set_num_workers(kTestPoolSize);
  std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
  ASSERT_TRUE(net->Run());
}

TEST(NetTest, DISABLED_OperatorWithDisabledEvent) {
  const auto spec = R"DOC(
        name: "example"
        type: "async_scheduling"
        external_input: "in"
        op {
          input: "in"
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
    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    net->GetOperators()[0]->DisableEvent();
    // async_scheduling propagates exception
    bool caught_exception = false;
    try {
      net->Run();
    } catch (const std::exception& e) {
      caught_exception = true;
    }
    ASSERT_TRUE(caught_exception);
  }
}

TEST(NetTest, ExecutorOverride) {
  const auto spec = R"DOC(
        name: "example"
        type: "dag"
  )DOC";

  NetDef net_def;
  CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));

  {
    Workspace ws;
    auto old = FLAGS_caffe2_override_executor;
    auto g = MakeGuard([&]() { FLAGS_caffe2_override_executor = old; });
    FLAGS_caffe2_override_executor = "dag,async_scheduling";

    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    auto async_net =
        caffe2::dynamic_cast_if_rtti<AsyncSchedulingNet*>(net.get());
    ASSERT_TRUE(async_net != nullptr);
  }
}

TEST(NetTest, AsyncEmptyNet) {
  const auto spec = R"DOC(
        name: "example"
        type: "async_scheduling"
  )DOC";

  Workspace ws;
  NetDef net_def;
  CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));

  {
    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    bool caught_exception = false;
    try {
      ASSERT_TRUE(net->Run());
    } catch (const std::exception& e) {
      caught_exception = true;
    }
    ASSERT_FALSE(caught_exception);
  }
}

TEST(NetTest, DISABLED_RunAsyncFailure) {
  const auto spec = R"DOC(
        name: "example"
        type: "async_scheduling"
        op {
          input: "in"
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
    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));

    bool caught_exception = false;
    try {
      ASSERT_FALSE(net->Run());
    } catch (const std::exception& e) {
      caught_exception = true;
    }
    ASSERT_TRUE(caught_exception);
  }
}

TEST(NetTest, NoTypeNet) {
  const auto spec = R"DOC(
        name: "no_type_net"
  )DOC";

  Workspace ws;
  NetDef net_def;
  CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));

  {
    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    ASSERT_TRUE(net);
  }
}

class NotFinishingOp final : public Operator<CPUContext> {
 public:
  NotFinishingOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    // never calls SetFinished
    return true;
  }

  bool HasAsyncPart() const override {
    return true;
  }
};

REGISTER_CPU_OPERATOR(NotFinishingOp, NotFinishingOp);

OPERATOR_SCHEMA(NotFinishingOp);

TEST(NetTest, PendingOpsAndNetFailure) {
  const auto spec = R"DOC(
        name: "example"
        type: "async_scheduling"
        op {
          type: "NotFinishingOp"
        }
        op {
          type: "NetTestDummy"
          arg {
            name: "fail"
            i: 1
          }
        }
)DOC";

  NetDef net_def;
  CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));

  Workspace ws;
  std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));

  try {
    // net is not stuck and returns false
    ASSERT_FALSE(net->Run());
  } catch (const caffe2::AsyncNetCancelled&) {
    // Cancellation exception is fine since if the ops run concurrently the
    // NotFinishingOp may be cancelled with an exception.
  }
}

class AsyncErrorOp final : public Operator<CPUContext> {
 public:
  AsyncErrorOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        throw_(OperatorBase::GetSingleArgument<bool>("throw", false)),
        fail_in_sync_(
            OperatorBase::GetSingleArgument<bool>("fail_in_sync", false)),
        sleep_time_s_(OperatorBase::GetSingleArgument<int>("sleep_time", 1)),
        error_msg_(OperatorBase::GetSingleArgument<std::string>(
            "error_msg",
            "Error")) {}

  bool RunOnDevice() override {
    if (fail_in_sync_) {
      if (throw_) {
        throw std::logic_error(error_msg_);
      } else {
        return false;
      }
    } else {
      if (thread_) {
        thread_->join();
      }
      thread_ = std::make_unique<std::thread>([this]() {
        try {
          std::this_thread::sleep_for(std::chrono::seconds(sleep_time_s_));
          if (throw_) {
            throw std::logic_error(error_msg_);
          } else {
            if (!cancel_.test_and_set()) {
              event().SetFinished(error_msg_.c_str());
            }
          }
        } catch (...) {
          if (!cancel_.test_and_set()) {
            event().SetFinishedWithException(error_msg_.c_str());
          }
        }
      });
      return true;
    }
  }

  bool HasAsyncPart() const override {
    return true;
  }

  void CancelAsyncCallback() override {
    cancel_.test_and_set();
  }

  ~AsyncErrorOp() override {
    if (thread_) {
      thread_->join();
    }
  }

 private:
  std::unique_ptr<std::thread> thread_;
  bool throw_;
  bool fail_in_sync_;
  int sleep_time_s_;
  std::string error_msg_;
  std::atomic_flag cancel_ = ATOMIC_FLAG_INIT;
};

REGISTER_CPU_OPERATOR(AsyncErrorOp, AsyncErrorOp);
OPERATOR_SCHEMA(AsyncErrorOp);

std::unique_ptr<NetBase> AsyncErrorNet(
    Workspace* ws,
    const std::string& net_name,
    bool throw_,
    bool fail_in_sync) {
  std::string spec_template = R"DOC(
        name: "<NET_NAME>"
        type: "async_scheduling"
        op {
          type: "AsyncErrorOp"
          arg {
            name: "throw"
            i: <THROW>
          }
          arg {
            name: "fail_in_sync"
            i: <FAIL_IN_SYNC>
          }
        }
  )DOC";

  std::string spec = spec_template;
  ReplaceAll(spec, "<NET_NAME>", net_name.c_str());
  ReplaceAll(spec, "<THROW>", throw_ ? "1" : "0");
  ReplaceAll(spec, "<FAIL_IN_SYNC>", fail_in_sync ? "1" : "0");

  NetDef net_def;
  CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));
  return CreateNet(net_def, ws);
}

TEST(NetTest, AsyncErrorOpTest) {
  Workspace ws;

  // Throw in sync part
  auto net = AsyncErrorNet(&ws, "net1", /*throw_*/ true, /*fail_in_sync*/ true);
#ifdef CAFFE2_USE_EXCEPTION_PTR
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(net->Run(), std::logic_error);
#endif

  // Return false in sync part
  net = AsyncErrorNet(&ws, "net2", /*throw_*/ false, /*fail_in_sync*/ true);
  ASSERT_FALSE(net->Run());

  // SetFinishedWithException in async part
  net = AsyncErrorNet(&ws, "net3", /*throw_*/ true, /*fail_in_sync*/ false);
#ifdef CAFFE2_USE_EXCEPTION_PTR
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(net->Run(), std::logic_error);
#endif

  // SetFinished(err) in async part
  net = AsyncErrorNet(&ws, "net4", /*throw_*/ false, /*fail_in_sync*/ false);
  ASSERT_FALSE(net->Run());
}

TEST(NetTest, AsyncErrorTimingsTest) {
  Workspace ws;
  std::string spec = R"DOC(
        name: "net"
        type: "async_scheduling"
        op {
          type: "AsyncErrorOp"
          arg {
            name: "throw"
            i: 1
          }
          arg {
            name: "fail_in_sync"
            i: 0
          }
          arg {
            name: "sleep_time"
            i: 2
          }
          arg {
            name: "error_msg"
            s: "Error1"
          }
        }
        op {
          type: "AsyncErrorOp"
          arg {
            name: "throw"
            i: 1
          }
          arg {
            name: "fail_in_sync"
            i: 0
          }
          arg {
            name: "sleep_time"
            i: 1
          }
          arg {
            name: "error_msg"
            s: "Error2"
          }
        }
  )DOC";

  NetDef net_def;
  CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));
  auto net = CreateNet(net_def, &ws);

  try {
    net->Run();
  } catch (const std::logic_error& e) {
    ASSERT_TRUE(std::string(e.what()) == "Error2");
  } catch (...) {
    FAIL() << "Expected std::logic_error thrown";
  }
}

class SyncErrorOp final : public Operator<CPUContext> {
 public:
  SyncErrorOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        fail_(OperatorBase::GetSingleArgument<bool>("fail", true)),
        throw_(OperatorBase::GetSingleArgument<bool>("throw", false)) {}

  bool RunOnDevice() override {
    if (fail_) {
      if (throw_) {
        throw std::logic_error("Error");
      } else {
        return false;
      }
    } else {
      return true;
    }
  }

  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~SyncErrorOp() override {}

 private:
  bool fail_;
  bool throw_;
};

REGISTER_CPU_OPERATOR(SyncErrorOp, SyncErrorOp);
OPERATOR_SCHEMA(SyncErrorOp);

std::unique_ptr<NetBase>
ChainErrorNet(Workspace* ws, const std::string& net_name, bool throw_) {
  std::string spec_template = R"DOC(
        name: "<NET_NAME>"
        type: "async_scheduling"
        op {
          type: "SyncErrorOp"
          arg {
            name: "fail"
            i: 1
          }
          arg {
            name: "throw"
            i: <THROW>
          }
        }
        op {
          type: "SyncErrorOp"
          arg {
            name: "fail"
            i: 0
          }
        }
  )DOC";

  std::string spec = spec_template;
  ReplaceAll(spec, "<NET_NAME>", net_name.c_str());
  ReplaceAll(spec, "<THROW>", throw_ ? "1" : "0");

  NetDef net_def;
  CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));
  return CreateNet(net_def, ws);
}

TEST(NetTest, ChainErrorTest) {
  Workspace ws;

  auto net = ChainErrorNet(&ws, "net1", /*throw_*/ true);
#ifdef CAFFE2_USE_EXCEPTION_PTR
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(net->Run(), std::logic_error);
#endif

  net = ChainErrorNet(&ws, "net2", /*throw_*/ false);
  ASSERT_FALSE(net->Run());
}

void testProfDAGNetErrorCase(bool test_error) {
  std::string spec_template = R"DOC(
        name: "prof_dag_error_test_net"
        type: "prof_dag"
        external_input: "in"
        op {
          input: "in"
          output: "hidden"
          type: "SyncErrorOp"
          arg {
            name: "fail"
            i: <FAIL>
          }
          arg {
            name: "throw"
            i: 0
          }
        }
        op {
          input: "hidden"
          output: "out"
          type: "SyncErrorOp"
          arg {
            name: "fail"
            i: 0
          }
        }
  )DOC";

  Workspace ws;
  ws.CreateBlob("in");

  NetDef net_def;
  std::string net_spec = spec_template;
  ReplaceAll(net_spec, "<FAIL>", test_error ? "1" : "0");
  CAFFE_ENFORCE(TextFormat::ParseFromString(net_spec, &net_def));
  auto net = CreateNet(net_def, &ws);

  // with failing op - net runs return false, without - true
  for (auto num_runs = 0; num_runs < 10; ++num_runs) {
    auto ret = net->Run();
    ASSERT_TRUE(test_error ? !ret : ret);
  }

  // with failing op - prof_dag handles invalid runs and returns empty stats,
  // without - returns stats for each op
  auto* prof_dag = dynamic_cast_if_rtti<AsyncNetBase*>(net.get());
  TORCH_CHECK_NOTNULL(prof_dag);
  auto stats_proto = prof_dag->GetPerOperatorCost();
  ASSERT_EQ(
      stats_proto.stats_size(), test_error ? 0 : net->GetOperators().size());
}

TEST(NetTest, ProfDAGNetErrorTest) {
  testProfDAGNetErrorCase(/*test_error=*/false);
  testProfDAGNetErrorCase(/*test_error=*/true);
}

} // namespace caffe2
