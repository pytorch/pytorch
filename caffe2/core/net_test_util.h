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

#include <google/protobuf/text_format.h>
#include <gtest/gtest.h>
#include "caffe2/core/net.h"
#include "caffe2/core/net_dag.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/scope_guard.h"

CAFFE2_DECLARE_bool(caffe2_disable_chaining);

namespace caffe2 {

template <class Context>
class NetTestDummyOp final : public Operator<Context> {
 public:
  NetTestDummyOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        fail_(OperatorBase::GetSingleArgument<bool>("fail", false)) {}

  bool RunOnDevice() override {
    if (fail_) {
      return false;
    }
    counter.fetch_add(1);
    return true;
  }

  static std::atomic<int> counter;

 protected:
  const bool fail_;
};

template <class Context>
std::atomic<int> NetTestDummyOp<Context>::counter;

unique_ptr<NetBase> CreateNetTestHelper(
    Workspace* ws,
    const vector<string>& input,
    const vector<string>& output);

void testExecution(std::unique_ptr<NetBase>& net);

void checkChainingAndRun(
    const char* spec,
    const dag_utils::ExecutionChains& expected);

void checkNumChainsAndRun(const char* spec, const int expected_num_chains);

} // namespace caffe2
