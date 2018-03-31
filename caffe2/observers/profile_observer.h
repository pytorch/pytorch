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

#pragma once

#include <unordered_map>

#include "caffe2/core/common.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/event.h"
#include "caffe2/core/net.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include "caffe2/observers/operator_attaching_net_observer.h"
#include "caffe2/operators/rnn/rnn_capable_operator_observer.h"

namespace caffe2 {

/**
 * This observer displays a description of each operator executed in a network.
 * This includes input and tensors (name, size, type), arguments, and execution
 * time. This can be used to analyze different performance characteristics.
 * NOTE: Currently this observer only supports synchronized computation
 **/

class ProfileObserver;
class ProfileCounter {
 public:
  explicit ProfileCounter() {}

 protected:
  Timer timer_;
  float start_time_ = 0.0f;
  float run_time_ = 0.0f;
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

class ProfileOperatorObserver : public ProfileCounter,
                                public RNNCapableOperatorObserver {
 public:
  explicit ProfileOperatorObserver(OperatorBase* subject) = delete;
  explicit ProfileOperatorObserver(
      OperatorBase* subject,
      ProfileObserver* netObserver)
      : RNNCapableOperatorObserver(subject), netObserver_(netObserver) {
    if (subject) {
      net_position_ = subject->net_position();
    }
  }
  explicit ProfileOperatorObserver(
      OperatorBase* subject,
      ProfileObserver* netObserver,
      int net_position,
      int rnn_order)
      : ProfileOperatorObserver(subject, netObserver) {
    net_position_ = net_position;
    rnn_order_ = rnn_order;
  }

  std::unique_ptr<ObserverBase<OperatorBase>> rnnCopy(
      OperatorBase* subject,
      int rnnOrder) const override;

  void Dump() const;

  virtual std::string getId() const {
    std::stringstream ss;
    ss << net_position_;
    if (rnn_order_ != OperatorBase::kNoNetPositionSet) {
      ss << "-" << rnn_order_;
    }
    return ss.str();
  }

 protected:
  ProfileObserver* netObserver_;
  int net_position_; // Needed because this is not visible in RNN Executor

 private:
  void Start() override;
  void Stop() override;
};

class ProfileObserver final : public OperatorAttachingNetObserver<
                                  ProfileOperatorObserver,
                                  ProfileObserver> {
 public:
  explicit ProfileObserver(NetBase* subject)
      : OperatorAttachingNetObserver<ProfileOperatorObserver, ProfileObserver>(
            subject,
            this) {}

  void Start() override{};
  void Stop() override{};

 private:
  vector<const ProfileOperatorObserver*> operator_observers_;
};

} // namespace caffe2
