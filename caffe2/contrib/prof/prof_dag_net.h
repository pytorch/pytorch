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

#include "caffe2/core/net_dag.h"
#include "caffe2/proto/prof_dag.pb.h"

namespace caffe2 {

struct Stats {
  float sum;
  float sqrsum;
  size_t cnt;
};

/**
 * This net type is identical to DAGNet, except that it
 * measures the time taken for each and every operator.
 *
 * To collect statistics from stable runs, this net ignores the first run.
 * Thus, at least two runs are required for this net to print operator metrics.
 */
class ProfDAGNet : public DAGNetBase {
 public:
  ProfDAGNet(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);
  ~ProfDAGNet();
  bool SupportsAsync() override {
    return false;
  }
  bool RunAsync() override;
  ProfDAGProtos GetOperatorStats();

 protected:
  bool RunAt(const std::vector<int>& chain) override;
  void PrintStats();
  void ValidateOpTensorDevices();
  ProfDAGProto ProtoMsg(std::pair<std::string, Stats> op_stat) const;
  std::vector<Stats> time_per_op_;
  CaffeMap<std::string, Stats> time_per_op_type_;
  int runs_ = 0;
};

} // namespace caffe2
