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

#include "caffe2/operators/mean_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Mean, MeanOp<CPUContext>);
REGISTER_CPU_OPERATOR(MeanGradient, MeanGradientOp<CPUContext>);

OPERATOR_SCHEMA(Mean)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Element-wise mean of each of the input tensors. The first input tensor can be
used in-place as the output tensor, in which case the mean will be done in
place and results will be accumulated in input0. All inputs and outputs must
have the same shape and data type.
)DOC")
    .Input(0, "data_0", "First of the input tensors. Can be inplace.")
    .Output(0, "mean", "Output tensor. Same dimension as inputs.");

class GetMeanGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    auto outputs = std::vector<string>();
    for (int i = 0; i < def_.input_size(); i++) {
      outputs.push_back(GI(i));
    }
    return SingleGradientDef(
        "MeanGradient", "", std::vector<string>{GO(0)}, outputs);
  }
};

REGISTER_GRADIENT(Mean, GetMeanGradient);

OPERATOR_SCHEMA(MeanGradient)
    .NumInputs(1)
    .NumOutputs(1, INT_MAX)
    .AllowInplace({{0, 0}});

} // namespace caffe2
