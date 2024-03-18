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

#include "caffe2/core/module.h"
#include "caffe2/core/operator.h"

// An explicitly defined module, testing correctness when we dynamically link a
// module
CAFFE2_MODULE(caffe2_module_test_dynamic, "Dynamic module only used for testing.");

namespace caffe2 {

class Caffe2ModuleTestDynamicDummyOp : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;
  bool Run(int /* unused */ /*stream_id*/) override {
    return true;
  }
  virtual string type() {
    return "base";
  }
};

REGISTER_CPU_OPERATOR(
  Caffe2ModuleTestDynamicDummy, Caffe2ModuleTestDynamicDummyOp);
OPERATOR_SCHEMA(Caffe2ModuleTestDynamicDummy);

} // namespace caffe2
