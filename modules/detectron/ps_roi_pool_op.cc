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

#include "ps_roi_pool_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(PSRoIPool, PSRoIPoolOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    PSRoIPoolGradient,
    PSRoIPoolGradientOp<float, CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(PSRoIPool)
    .NumInputs(2)
    .NumOutputs(2)
    .SetDoc(R"DOC(
Position Sensitive Region of Interest Pooling as used in R-FCN.
)DOC")
    .Arg(
        "spatial_scale",
        "(float) default 1.0; Spatial scale of the input feature map X "
        "relative to the input image. E.g., 0.0625 if X has a stride of 16 "
        "w.r.t. the input image.")
    .Arg(
        "group_size",
        "(int) default 1; pooled_h = pooled_w = group_size where pooled_{h,w} "
        "is the pooled output Y's height and width, respectively.")
    .Arg(
        "output_dim",
        "(int) default 1; number of channels in the pooled output, which might "
        "be the number of classes is used for classification or 4 if used for "
        "class agnostic bounding box regression.")
    .Input(
        0,
        "X",
        "4D position sensitive feature map input of shape (N, C, H, W), where "
        "C = group_size**2 * output_dim.")
    .Input(
        1,
        "RoIs",
        "2D input of shape (R, 5) specifying R RoIs with five columns "
        "representing: batch index in [0, N - 1], x1, y1, x2, y2. The RoI "
        "coordinates are in the coordinate system of the input image.")
    .Output(
        0,
        "Y",
        "4D output of shape (R, output_dim, pooled_h, pooled_w). The r-th "
        "batch element is a pooled feature map cooresponding to the r-th RoI.")
    .Output(
        1,
        "argmaxes",
        "4D output of shape (R, output_dim, pooled_h, pooled_w). Same as Y, "
        "except it records the argmax indices rather than the max pooled "
        "values.");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(PSRoIPoolGradient)
    .NumInputs(4)
    .NumOutputs(1)
    .Input(
        0,
        "X",
        "See PSRoIPool.")
    .Input(
        1,
        "RoIs",
        "See PSRoIPool.")
    .Input(
        2,
        "argmaxes",
        "See PSRoIPool.")
    .Input(
        3,
        "dY",
        "Gradient of forward output 0 (Y)")
    .Output(
        0,
        "dX",
        "Gradient of forward input 0 (X)");

class GetPSRoIPoolGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "PSRoIPoolGradient",
        "",
        vector<string>{I(0), I(1), O(1), GO(0)},
        vector<string>{GI(0)});
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(PSRoIPool, GetPSRoIPoolGradient);

} // namespace caffe2
