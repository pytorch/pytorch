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

#include "sigmoid_focal_loss_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(SigmoidFocalLoss, SigmoidFocalLossOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    SigmoidFocalLossGradient,
    SigmoidFocalLossGradientOp<float, CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(SigmoidFocalLoss)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
The binary form of Focal Loss designed for use in RetinaNet-like models.
The input is assumed to be unnormalized scores (sometimes called 'logits')
arranged in a 4D tensor with shape (N, C, H, W), where N is the number of
elements in the batch, H and W are the height and width, and C = num_anchors *
num_classes defines num_anchors 'groups' of logits, each of length
num_classes. For the binary form of Focal Loss, num_classes does not include
the background category. (So, for COCO, num_classes = 80, not 81.)

The binary form of focal loss is:

  FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t),

where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0,
respectively.

See: https://arxiv.org/abs/1708.02002 for details.
)DOC")
    .Arg(
       "scale",
       "(float) default 1.0; multiply the loss by this scale factor.")
    .Arg(
       "alpha",
       "(float) default 0.25; Focal Loss's alpha hyper-parameter.")
    .Arg(
       "gamma",
       "(float) default 1.0; Focal Loss's gamma hyper-parameter.")
    .Arg(
       "num_classes",
       "(int) default 80; number of classes (excluding background).")
    .Input(
       0,
       "logits",
       "4D tensor of sigmoid inputs (called 'scores' or 'logits') with shape "
       "(N, C, H, W), where C = num_anchors * num_classes.")
    .Input(
       1,
       "labels",
       "4D tensor of labels with shape (N, num_anchors, H, W). Each entry is "
       "a class label in [0, num_classes - 1] (inclusive). The label "
       "identifies the one class that should have a sigmoid target of 1.")
    .Input(
       2,
       "normalizer",
       "Scalar; the loss is normalized by 1 / max(1, normalizer)."
    )
    .Output(
       0,
       "loss",
       "Scalar loss.");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(SigmoidFocalLossGradient)
    .NumInputs(4)
    .NumOutputs(1)
    .Input(
        0,
        "logits",
        "See SigmoidFocalLoss.")
    .Input(
        1,
        "labels",
        "See SigmoidFocalLoss.")
    .Input(
        2,
        "normalizer",
        "See SigmoidFocalLoss.")
    .Input(
        3,
        "d_loss",
        "Gradient of forward output 0 (loss)")
    .Output(
        0,
        "d_logits",
        "Gradient of forward input 0 (logits)");

class GetSigmoidFocalLossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  vector<OperatorDef> GetGradientDefs() override {
    vector<string> blob_names{
        {I(0), I(1), I(2), GO(0)},
    };

    return SingleGradientDef(
        "SigmoidFocalLossGradient", "", blob_names, vector<string>{GI(0)});
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(SigmoidFocalLoss, GetSigmoidFocalLossGradient);

} // namespace caffe2
