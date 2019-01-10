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

#include "caffe2/experiments/operators/tt_pad_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(TTPad, TTPadOp<float, CPUContext>);
OPERATOR_SCHEMA(TTPad).NumInputs(1).NumOutputs(2).EnforceInplace({{0, 0}});

REGISTER_CPU_OPERATOR(TTPadGradient, TTPadGradientOp<float, CPUContext>);
OPERATOR_SCHEMA(TTPadGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .EnforceInplace({{0, 0}});

class GetTTPadGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "TTPadGradient",
        "",
        vector<string>{GO(0), O(1)},
        vector<string>{GI(0)},
        Def().arg());
  }
};

REGISTER_GRADIENT(TTPad, GetTTPadGradient);

} // namespace
} // namespace caffe2
