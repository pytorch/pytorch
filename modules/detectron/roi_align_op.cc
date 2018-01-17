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

#include "roi_align_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(RoIAlign, RoIAlignOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(RoIAlignGradient, RoIAlignGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(RoIAlign)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Region of Interest (RoI) align operation as used in Mask R-CNN.
)DOC")
    .Arg(
        "spatial_scale",
        "(float) default 1.0; Spatial scale of the input feature map X "
        "relative to the input image. E.g., 0.0625 if X has a stride of 16 "
        "w.r.t. the input image.")
    .Arg(
        "pooled_h",
        "(int) default 1; Pooled output Y's height.")
    .Arg(
        "pooled_w",
        "(int) default 1; Pooled output Y's width.")
    .Arg(
        "sampling_ratio",
        "(int) default -1; number of sampling points in the interpolation grid "
        "used to compute the output value of each pooled output bin. If > 0, "
        "then exactly sampling_ratio x sampling_ratio grid points are used. If "
        "<= 0, then an adaptive number of grid points are used (computed as "
        "ceil(roi_width / pooled_w), and likewise for height)."
    )
    .Input(
        0,
        "X",
        "4D feature map input of shape (N, C, H, W).")
    .Input(
        1,
        "RoIs",
        "2D input of shape (R, 5) specifying R RoIs with five columns "
        "representing: batch index in [0, N - 1], x1, y1, x2, y2. The RoI "
        "coordinates are in the coordinate system of the input image.")
    .Output(
        0,
        "Y",
        "4D output of shape (R, C, pooled_h, pooled_w). The r-th batch element "
        "is a pooled feature map cooresponding to the r-th RoI.");

OPERATOR_SCHEMA(RoIAlignGradient)
    .NumInputs(3)
    .NumOutputs(1)
    .Input(
        0,
        "X",
        "See RoIPoolF.")
    .Input(
        1,
        "RoIs",
        "See RoIPoolF.")
    .Input(
        2,
        "dY",
        "Gradient of forward output 0 (Y)")
    .Output(
        0,
        "dX",
        "Gradient of forward input 0 (X)");

class GetRoIAlignGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "RoIAlignGradient",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(RoIAlign, GetRoIAlignGradient);

} // namespace caffe2
