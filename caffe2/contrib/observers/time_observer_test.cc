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

unique_ptr<NetBase> CreateNetTestHelper(Workspace* ws) {
  NetDef net_def;
  {
    auto& op = *(net_def.add_op());
    op.set_type("SleepOp");
    op.add_input("in");
    op.add_output("hidden");
  }
  {
    auto& op = *(net_def.add_op());
    op.set_type("SleepOp");
    op.add_input("hidden");
    op.add_output("out");
  }
  net_def.add_external_input("in");
  net_def.add_external_output("out");

  return CreateNet(net_def, ws);
}
}

TEST(TimeObserverTest, Test3Seconds) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef net_def;
  unique_ptr<NetBase> net(CreateNetTestHelper(&ws));
  unique_ptr<TimeObserver<NetBase>> net_ob =
      make_unique<TimeObserver<NetBase>>(net.get());
  net->SetObserver(std::move(net_ob));
  net->Run();
  auto* ob = dynamic_cast_if_rtti<TimeObserver<NetBase>*>(net->GetObserver());
  CAFFE_ENFORCE(ob);
  LOG(INFO) << "av time children: " << ob->average_time_children();
  LOG(INFO) << "av time: " << ob->average_time();
  CAFFE_ENFORCE(ob->average_time_children() > 3000);
  CAFFE_ENFORCE(ob->average_time_children() < 3500);
  CAFFE_ENFORCE(ob->average_time() > 6000);
  CAFFE_ENFORCE(ob->average_time() < 6500);
}
}
