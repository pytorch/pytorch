/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/core/common.h"
#include "caffe2/core/net.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/operator.h"
#include "time_observer.h"

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
    StartAllObservers();
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    StopAllObservers();
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
  const auto* ob = dynamic_cast_if_rtti<const TimeObserver<NetBase>*>(
      net->AttachObserver(std::move(net_ob)));
  net->Run();
  CAFFE_ENFORCE(ob);
  LOG(INFO) << "av time children: " << ob->average_time_children();
  LOG(INFO) << "av time: " << ob->average_time();
  CAFFE_ENFORCE(ob->average_time_children() > 3000);
  CAFFE_ENFORCE(ob->average_time_children() < 3500);
  CAFFE_ENFORCE(ob->average_time() > 6000);
  CAFFE_ENFORCE(ob->average_time() < 6500);
}
}
