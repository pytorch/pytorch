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

#include "sample_as_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(SampleAs, SampleAsOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(SampleAsGradient, SampleAsGradientOp<float, CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(SampleAs)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Select the batch elements from input tensor X where the corresponding input
label value is > 0.
)DOC")
    .Input(
        0,
        "X",
        "Tensor of at least 1D shape (N, ...).")
    .Input(
        1,
        "labels",
        "Tensor of type int with 1D shape (N, ).")
    .Output(
        0,
        "Y",
        "Tensor with number of dims matching X, but with the length of dim 0 "
        "equal to the number of non-zero elements in labels. The batch items "
        "from X corresponding to the non-zero elements in labels are copied "
        "into Y.");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(SampleAsGradient)
    .NumInputs(3)
    .NumOutputs(1)
    .Input(
        0,
        "X",
        "See SampleAs.")
    .Input(
        1,
        "labels",
        "See SampleAs."
    )
    .Input(
        2,
        "dY",
        "Gradient of forward output 0 (Y).")
    .Output(
        0,
        "dX",
        "Gradient of forward input 0 (X).");

class GetSampleAsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SampleAsGradient",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0)});
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(SampleAs, GetSampleAsGradient);

} // namespace caffe2
