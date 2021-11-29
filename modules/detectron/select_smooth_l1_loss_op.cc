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

#include "select_smooth_l1_loss_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    SelectSmoothL1Loss,
    SelectSmoothL1LossOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    SelectSmoothL1LossGradient,
    SelectSmoothL1LossGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(SelectSmoothL1Loss)
    .NumInputs(4)
    .NumOutputs(1)
    .SetDoc(R"DOC(
RetinaNet specific op for computing Smooth L1 Loss at select locations in a 4D
tensor that encodes bounding box regression predictions.
)DOC")
    .Arg(
        "beta",
        "(float) default 1.0; L2 to L1 transition point.")
    .Arg(
        "scale",
        "(float) default 1.0; multiply the loss by this scale factor.")
    .Input(
        0,
        "Y_hat",
        "4D tensor of bounding box regression predictions with shape "
        "(N, 4 * num_bbox_classes * num_anchors, H, W).")
    .Input(
        1,
        "Y",
        "2D tensor of labels shape (M, 4) for 4 contiguous channels starting "
        "at each of the M locations selected by the locations input.")
    .Input(
        2,
        "locations",
        "2D tensor of shape (M, 4) that identifies M 'select' locations "
        "encoded by the four columns: (n, c, y, x). The loss is computed on the "
        "four contiguous channel locations [c, c + 3] (inclusive).")
    .Input(
        3,
        "normalizer",
        "Scalar; the loss is divided by max(1, normalizer).")
    .Output(
        0,
        "loss",
        "Scalar loss.");

OPERATOR_SCHEMA(SelectSmoothL1LossGradient)
    .NumInputs(5)
    .NumOutputs(1)
    .Input(
        0,
        "Y_hat",
        "See SelectSmoothL1Loss.")
    .Input(
        1,
        "Y",
        "See SelectSmoothL1Loss.")
    .Input(
        2,
        "locations",
        "See SelectSmoothL1Loss.")
    .Input(
        3,
        "normalizer",
        "See SelectSmoothL1Loss.")
    .Input(
        4,
        "d_loss",
        "Gradient of forward output 0 (loss).")
    .Output(
        0,
        "d_Y_hat",
        "Gradient of forward input 0 (Y_hat).");

class GetSelectSmoothL1LossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SelectSmoothL1LossGradient",
        "",
        vector<string>{I(0), I(1), I(2), I(3), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(SelectSmoothL1Loss, GetSelectSmoothL1LossGradient);

} // namespace caffe2
