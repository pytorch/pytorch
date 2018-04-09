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

#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {

class IDEEPSumOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPSumOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws) {}
  virtual ~IDEEPSumOp() {}

  bool RunDirectCopy() {
    const auto &X = Input(INPUT0);
    auto* Y = Output(OUTPUT);

    ideep::direct_copy::compute(X, *Y);

    return true;
  }

  bool RunSumUp() {
    auto* Y = Output(OUTPUT);
    vector<itensor> inputs;
    vector<float> scales (InputSize(), 1.0);

    for (int i = 0; i < InputSize(); ++i) {
      inputs.emplace_back(Input(i));
    }

    ideep::sum::compute(scales, inputs, *Y);

    return true;
  }

  bool RunOnDevice() override {
    if (InputSize() == 1)
      return RunDirectCopy();
    else
      return RunSumUp();
  }

 private:

  INPUT_TAGS(INPUT0);
  OUTPUT_TAGS(OUTPUT);
};

REGISTER_IDEEP_OPERATOR(Sum, IDEEPSumOp);

} // namespace caffe2
