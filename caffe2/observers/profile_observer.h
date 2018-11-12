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

namespace caffe2 {

/**
 * This observer displays a description of each operator executed in a network.
 * This includes input and tensors (name, size, type), arguments, analytical
 *cost and execution time. This can be used to analyze different performance
 *characteristics. NOTE: Currently this observer only supports synchronized
 *computation. And for RNN, --caffe2_rnn_executor=False need to be set if want
 *to get the cost summary at the net level.
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
                                public ObserverBase<OperatorBase> {
 public:
  struct DetailedStat {
    string opType;
    struct OpSchema::Cost c;
  };
  explicit ProfileOperatorObserver(OperatorBase* subject) = delete;
  explicit ProfileOperatorObserver(
      OperatorBase* subject,
      DetailedStat* stat,
      ProfileObserver* netObserver)
      : ObserverBase<OperatorBase>(subject),
        stat_(stat),
        netObserver_(netObserver) {
    stat->opType = subject->debug_def().type();
    if (subject) {
      net_position_ = subject->net_position();
    }
  }
  explicit ProfileOperatorObserver(
      OperatorBase* subject,
      DetailedStat* stat,
      ProfileObserver* netObserver,
      int net_position,
      int rnn_order)
      : ProfileOperatorObserver(subject, stat, netObserver) {
    net_position_ = net_position;
    rnn_order_ = rnn_order;
  }

  std::unique_ptr<ObserverBase<OperatorBase>> rnnCopy(
      OperatorBase* subject,
      int rnn_order) const override;

  void Dump() const;

  virtual std::string getId() const {
    std::stringstream ss;
    ss << net_position_;
    if (rnn_order_ != OperatorBase::kNoNetPositionSet) {
      ss << "-" << rnn_order_;
    }
    return ss.str();
  }

  OpSchema::Cost getOpCost() {
    const string& op_type = subject_->debug_def().type();
    auto* schema = OpSchemaRegistry::Schema(op_type);
    OpSchema::Cost cost;
    if (schema && schema->HasCostInferenceFunction()) {
      vector<TensorShape> shapes = subject_->InputTensorShapes();

      auto known_shapes = std::accumulate(
          shapes.begin(),
          shapes.end(),
          true,
          [](bool acc, const TensorShape& shape) {
            return acc && !shape.unknown_shape();
          });
      if (known_shapes) {
        cost = schema->InferCost(subject_->debug_def(), shapes);
      }
    }
    return cost;
  }

  void updateDetailedStat(const OpSchema::Cost cost) {
    stat_->c.flops += cost.flops;
    stat_->c.bytes_read += cost.bytes_read;
    stat_->c.bytes_written += cost.bytes_written;
    stat_->c.params_bytes += cost.params_bytes;
  }

 protected:
  DetailedStat* stat_;
  OpSchema::Cost cost_;
  ProfileObserver* netObserver_;
  int net_position_; // Needed because this is not visible in RNN Executor
  int rnn_order_ = OperatorBase::kNoNetPositionSet;

 private:
  void Start() override;
  void Stop() override;
};

class ProfileObserver final : public ObserverBase<NetBase> {
 public:
  explicit ProfileObserver(NetBase* subject)
      : ObserverBase<NetBase>(subject),
        detailedOpStats_(subject->GetOperators().size()),
        net_name_(subject->Name()) {
    const auto& ops = subject->GetOperators();
    for (int i = 0; i < ops.size(); i++) {
      ops[i]->AttachObserver(caffe2::make_unique<ProfileOperatorObserver>(
          ops[i], &detailedOpStats_[i], this));
    }
  }
  ~ProfileObserver();
  CaffeMap<string, OpSchema::Cost> getAggregatedOpTypeCost() const {
    CaffeMap<string, OpSchema::Cost> cost_per_op_type;
    for (int idx = 0; idx < detailedOpStats_.size(); ++idx) {
      const auto& stat = detailedOpStats_[idx];
      uint64_t flops = stat.c.flops;
      uint64_t bytes_read = stat.c.bytes_read;
      uint64_t bytes_written = stat.c.bytes_written;

      cost_per_op_type[stat.opType].flops += flops;
      cost_per_op_type[stat.opType].bytes_read += bytes_read;
      cost_per_op_type[stat.opType].bytes_written += bytes_written;
    }
    return cost_per_op_type;
  }

  void Start() override{};
  void Stop() override{};

 private:
  vector<const ProfileOperatorObserver*> operator_observers_;
  std::vector<ProfileOperatorObserver::DetailedStat> detailedOpStats_;
  std::string net_name_;
};

} // namespace caffe2
