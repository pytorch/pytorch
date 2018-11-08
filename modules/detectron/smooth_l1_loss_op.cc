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

#include "smooth_l1_loss_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SmoothL1Loss, SmoothL1LossOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    SmoothL1LossGradient,
    SmoothL1LossGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(SmoothL1Loss)
    .NumInputs(4)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Smooth L1 Loss is a minor variation of Huber loss in which the point of
transition between L2 loss and L1 loss is adjustable by a hyper-parameter beta:

  SmoothL1(x) = 0.5 * x^2 / beta      if |x| < beta
                |x| - 0.5 * beta      otherwise.

SmoothL1 is used in Fast R-CNN and decendants as the loss function for bounding
box regression.

The loss computed by this op has a flexible form:

  scale / N * sum_i alpha_out[i] * SmoothL1(alpha_in[i] * (y_hat[i] - y[i])).

The weights alpha_in and alpha_out are called the "inside" and "outside"
weights, respectively. The inside weights are typically set to either 0 or 1 to
implement ignoring (when 0) certain samples. The outside weights can be used
to implement a per-sample loss weight. The overall loss is scaled by scale / N,
where N is the number of batch elements in the input predictions.
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
        "Tensor of predictions (at least 1D).")
    .Input(
        1,
        "Y",
        "Tensor of labels with the same shape as Y_hat.")
    .Input(
        2,
        "alpha_in",
        "Tensor of inside weights with the same shape as Y.")
    .Input(
        3,
        "alpha_out",
        "Tensor of outside weights with the same shape as Y.")
    .Output(
        0,
        "loss",
        "Scalar loss.");

OPERATOR_SCHEMA(SmoothL1LossGradient)
    .NumInputs(5)
    .NumOutputs(1)
    .Input(
        0,
        "Y_hat",
        "See SmoothL1Loss.")
    .Input(
        1,
        "Y",
        "See SmoothL1Loss.")
    .Input(
        2,
        "alpha_in",
        "See SmoothL1Loss.")
    .Input(
        3,
        "alpha_out",
        "See SmoothL1Loss.")
    .Input(
        4,
        "d_loss",
        "Gradient of forward output 0 (loss).")
    .Output(
        0,
        "d_Y_hat",
        "Gradient of forward input 0 (Y_hat).");

class GetSmoothL1LossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SmoothL1LossGradient",
        "",
        vector<string>{I(0), I(1), I(2), I(3), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(SmoothL1Loss, GetSmoothL1LossGradient);

} // namespace caffe2
