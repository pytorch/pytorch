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

#include "upsample_nearest_op.h"
#ifdef USE_MKLDNN
#include "caffe2/ideep/operators/operator_fallback_ideep.h"
#include "caffe2/ideep/utils/ideep_operator.h"
#endif

namespace caffe2 {
#ifdef USE_MKLDNN
REGISTER_IDEEP_OPERATOR(
    UpsampleNearest,
    IDEEPFallbackOp<UpsampleNearestOp<float, CPUContext>>);
#endif

REGISTER_CPU_OPERATOR(UpsampleNearest, UpsampleNearestOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    UpsampleNearestGradient,
    UpsampleNearestGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(UpsampleNearest)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Nearest neighbor upsampling operation. Implementation taken from THCUNN.
)DOC")
    .Arg(
        "scale",
        "(int) default 2; integer upsampling factor.")
    .Input(
        0,
        "X",
        "4D feature map input of shape (N, C, H, W).")
    .Output(
        0,
        "Y",
        "4D feature map of shape (N, C, scale * H, scale * W); Values are "
        "neareast neighbor samples from X.");

OPERATOR_SCHEMA(UpsampleNearestGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .Input(
        0,
        "X",
        "See UpsampleNearest.")
    .Input(
        1,
        "dY",
        "Gradient of forward output 0 (Y).")
    .Output(
        0,
        "dX",
        "Gradient of forward input 0 (X).");

class GetUpsampleNearestGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "UpsampleNearestGradient",
        "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(UpsampleNearest, GetUpsampleNearestGradient);

} // namespace caffe2
