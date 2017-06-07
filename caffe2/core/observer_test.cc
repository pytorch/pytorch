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
  explicit DummyObserver<T>(T& subject) : ObserverBase<T>(subject) {}
  bool Start() override;
  bool Stop() override;

  ~DummyObserver() {}

 private:
  vector<unique_ptr<DummyObserver<OperatorBase>>> ops_obs;
};

template <>
bool DummyObserver<SimpleNet>::Start() {
  vector<OperatorBase*> operators = subject.getOperators();
  for (auto& op : operators) {
    ops_obs.push_back(caffe2::make_unique<DummyObserver<OperatorBase>>(*op));
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
bool DummyObserver<SimpleNet>::Stop() {
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
}

TEST(ObserverTest, TestNotify) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef net_def;
  unique_ptr<NetBase> net(CreateNetTestHelper(&ws, {"in"}, {"out"}));
  EXPECT_EQ(caffe2::dynamic_cast_if_rtti<SimpleNet*>(net.get()), net.get());
  unique_ptr<DummyObserver<SimpleNet>> net_ob =
      make_unique<DummyObserver<SimpleNet>>(
          *(caffe2::dynamic_cast_if_rtti<SimpleNet*>(net.get())));
  net.get()->Run();
  EXPECT_EQ(1212, counter.load());
}

TEST(ObserverTest, TestNotifyAfterDetach) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef net_def;
  unique_ptr<NetBase> net(CreateNetTestHelper(&ws, {"in"}, {"out"}));
  unique_ptr<DummyObserver<SimpleNet>> net_ob =
      make_unique<DummyObserver<SimpleNet>>(
          *(caffe2::dynamic_cast_if_rtti<SimpleNet*>(net.get())));
  net_ob.get()->Deactivate();
  counter = 0;
  net.get()->Run();
  EXPECT_EQ(0, counter.load());
}
}
