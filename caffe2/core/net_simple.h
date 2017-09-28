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

#ifndef CAFFE2_CORE_NET_SIMPLE_H_
#define CAFFE2_CORE_NET_SIMPLE_H_

#include <vector>

#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/net.h"
#include "caffe2/core/registry.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

// This is the very basic structure you need to run a network - all it
// does is simply to run everything in sequence. If you want more fancy control
// such as a DAG-like execution, check out other better net implementations.
class SimpleNet : public NetBase {
 public:
  SimpleNet(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);
  bool SupportsAsync() override {
    return false;
  }
  bool RunAsync() override;
  vector<float> TEST_Benchmark(
      const int warmup_runs,
      const int main_runs,
      const bool run_individual) override;

  /*
   * This returns a list of pointers to objects stored in unique_ptrs.
   * Used by Observers.
   *
   * Think carefully before using.
   */
  vector<OperatorBase*> GetOperators() const override {
    vector<OperatorBase*> op_list;
    for (auto& op : operators_) {
      op_list.push_back(op.get());
    }
    return op_list;
  }

 protected:
  vector<unique_ptr<OperatorBase>> operators_;

  DISABLE_COPY_AND_ASSIGN(SimpleNet);
};

} // namespace caffe2

#endif // CAFFE2_CORE_NET_SIMPLE_H_
