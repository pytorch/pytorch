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

#include "caffe2/operators/loss_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(AveragedLoss, AveragedLoss<float, CPUContext>);
REGISTER_CPU_OPERATOR(AveragedLossGradient,
                      AveragedLossGradient<float, CPUContext>);

OPERATOR_SCHEMA(AveragedLoss)
  .NumInputs(1)
  .NumOutputs(1)
  .ScalarType(TensorProto::FLOAT)
  .SetDoc(R"DOC(
AveragedLoss takes in a 1-D tensor as input and returns a single output float
value which represents the average of input data (average of the losses).
)DOC")
  .Input(0, "input", "The input data as Tensor")
  .Output(0, "output", "The output tensor of size 1 containing the averaged "
          "value.");

OPERATOR_SCHEMA(AveragedLossGradient).NumInputs(2).NumOutputs(1);

class GetAveragedLossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "AveragedLossGradient", "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(AveragedLoss, GetAveragedLossGradient);

}  // namespace caffe2
