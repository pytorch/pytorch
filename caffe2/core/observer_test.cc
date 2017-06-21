#include <google/protobuf/text_format.h>
#include <gtest/gtest.h>
#include "caffe2/core/common.h"
#include "caffe2/core/net.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/registry.h"
#include "caffe2/core/scope_guard.h"

namespace caffe2 {

namespace {

static std::atomic<int> counter;

template <class T>
class DummyObserver final : public ObserverBase<T> {
 public:
  explicit DummyObserver<T>(T* subject_) : ObserverBase<T>(subject_) {}
  bool Start() override;
  bool Stop() override;

  ~DummyObserver() {}

 private:
  vector<unique_ptr<DummyObserver<OperatorBase>>> ops_obs;
};

template <>
bool DummyObserver<NetBase>::Start() {
  vector<OperatorBase*> operators = subject_->getOperators();
  for (auto& op : operators) {
    ops_obs.push_back(caffe2::make_unique<DummyObserver<OperatorBase>>(op));
  }
  counter.fetch_add(1000);
  return true;
}

template <>
bool DummyObserver<OperatorBase>::Start() {
  counter.fetch_add(100);
  return true;
}

template <>
bool DummyObserver<NetBase>::Stop() {
  counter.fetch_add(10);
  return true;
}

template <>
bool DummyObserver<OperatorBase>::Stop() {
  counter.fetch_add(1);
  return true;
}

class ObsTestDummyOp final : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;
  bool Run(int /* unused */) override {
    if (observer_)
      observer_->Start();
    if (observer_)
      observer_->Stop();
    return true;
  }
};

REGISTER_CPU_OPERATOR(ObsTestDummy, ObsTestDummyOp);
REGISTER_CUDA_OPERATOR(ObsTestDummy, ObsTestDummyOp);

OPERATOR_SCHEMA(ObsTestDummy)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{0, 0}, {1, 1}});

const std::basic_string<char> kExampleNetDefString = {
    "  name: \"example\""
    "  op {"
    "    input: \"in\""
    "    output: \"hidden\""
    "    type: \"ObsTestDummy\""
    "  }"
    "  op {"
    "    input: \"hidden\""
    "    output: \"out\""
    "    type: \"ObsTestDummy\""
    "  }"};

const std::basic_string<char> kExampleDAGNetDefString = {
    "  name: \"example\""
    "  type: \"dag\""
    "  op {"
    "    input: \"in\""
    "    output: \"hidden\""
    "    type: \"ObsTestDummy\""
    "  }"
    "  op {"
    "    input: \"hidden\""
    "    output: \"out\""
    "    type: \"ObsTestDummy\""
    "  }"};

unique_ptr<NetBase> CreateNetTestHelper(
    Workspace* ws,
    const vector<string>& input,
    const vector<string>& output,
    const std::basic_string<char>& net_def_string) {
  NetDef net_def;
  CAFFE_ENFORCE(
      google::protobuf::TextFormat::ParseFromString(net_def_string, &net_def));
  for (const auto& name : input) {
    net_def.add_external_input(name);
  }
  for (const auto& name : output) {
    net_def.add_external_output(name);
  }
  return CreateNet(net_def, ws);
}
}

TEST(ObserverTest, TestNotify) {
  auto count_before = counter.load();
  Workspace ws;
  ws.CreateBlob("in");
  NetDef net_def;
  unique_ptr<NetBase> net(
      CreateNetTestHelper(&ws, {"in"}, {"out"}, kExampleNetDefString));
  EXPECT_EQ(caffe2::dynamic_cast_if_rtti<SimpleNet*>(net.get()), net.get());
  unique_ptr<DummyObserver<NetBase>> net_ob =
      make_unique<DummyObserver<NetBase>>(net.get());
  net.get()->Run();
  auto count_after = counter.load();
  EXPECT_EQ(1212, count_after - count_before);
}

TEST(ObserverTest, TestNotifyAfterDetach) {
  auto count_before = counter.load();
  Workspace ws;
  ws.CreateBlob("in");
  NetDef net_def;
  unique_ptr<NetBase> net(
      CreateNetTestHelper(&ws, {"in"}, {"out"}, kExampleNetDefString));
  unique_ptr<DummyObserver<NetBase>> net_ob =
      make_unique<DummyObserver<NetBase>>(net.get());
  net.get()->RemoveObserver();
  net.get()->Run();
  auto count_after = counter.load();
  EXPECT_EQ(0, count_after - count_before);
}

TEST(ObserverTest, TestDAGNetBase) {
  auto count_before = counter.load();
  Workspace ws;
  ws.CreateBlob("in");
  NetDef net_def;
  unique_ptr<NetBase> net(
      CreateNetTestHelper(&ws, {"in"}, {"out"}, kExampleDAGNetDefString));
  EXPECT_EQ(caffe2::dynamic_cast_if_rtti<DAGNetBase*>(net.get()), net.get());
  unique_ptr<DummyObserver<NetBase>> net_ob =
      make_unique<DummyObserver<NetBase>>(net.get());
  net.get()->Run();
  auto count_after = counter.load();
  EXPECT_EQ(1212, count_after - count_before);
}
}
