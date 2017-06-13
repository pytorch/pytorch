#include "caffe2/contrib/observers/time_observer.h"
#include "caffe2/core/common.h"
#include "caffe2/core/net.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/operator.h"

#include <google/protobuf/text_format.h>
#include <gtest/gtest.h>
#include <chrono>
#include <thread>

namespace caffe2 {

namespace {

class SleepOp final : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;
  bool Run(int /* unused */) override {
    if (observer_) {
      observer_->Start();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    if (observer_) {
      observer_->Stop();
    }
    return true;
  }
};

REGISTER_CPU_OPERATOR(SleepOp, SleepOp);
REGISTER_CUDA_OPERATOR(SleepOp, SleepOp);

OPERATOR_SCHEMA(SleepOp)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{0, 0}, {1, 1}});

const std::basic_string<char> kExampleNetDefString = {
    "  name: \"example\""
    "  op {"
    "    input: \"in\""
    "    output: \"hidden\""
    "    type: \"SleepOp\""
    "  }"
    "  op {"
    "    input: \"hidden\""
    "    output: \"out\""
    "    type: \"SleepOp\""
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

TEST(TimeObserverTest, Test3Seconds) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef net_def;
  unique_ptr<NetBase> net(CreateNetTestHelper(&ws, {"in"}, {"out"}));
  unique_ptr<TimeObserver<SimpleNet>> net_ob =
      make_unique<TimeObserver<SimpleNet>>(
          *(caffe2::dynamic_cast_if_rtti<SimpleNet*>(net.get())));
  net.get()->Run();
  CAFFE_ENFORCE(net_ob.get()->average_time_children() > 3000);
  CAFFE_ENFORCE(net_ob.get()->average_time_children() < 3500);
  CAFFE_ENFORCE(net_ob.get()->average_time() > 6000);
  CAFFE_ENFORCE(net_ob.get()->average_time() < 6500);
}
}
