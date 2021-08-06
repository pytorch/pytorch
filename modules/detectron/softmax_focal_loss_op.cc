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

#include "modules/detectron/softmax_focal_loss_op.h"

#include "caffe2/operators/softmax_utils.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SoftmaxFocalLoss, SoftmaxFocalLossOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    SoftmaxFocalLossGradient,
    SoftmaxFocalLossGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(SoftmaxFocalLoss)
    .NumInputs(3)
    .NumOutputs(2)
    .SetDoc(R"DOC(
A multiclass form of Focal Loss designed for use in RetinaNet-like models.
The input is assumed to be unnormalized scores (sometimes called 'logits')
arranged in a 4D tensor with shape (N, C, H, W), where N is the number of
elements in the batch, H and W are the height and width, and C = num_anchors *
num_classes. The softmax is applied num_anchors times along the C axis.

The softmax version of focal loss is:

  FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t),

where p_i = exp(s_i) / sum_j exp(s_j), t is the target (ground truth) class, and
s_j is the unnormalized score for class j.

See: https://arxiv.org/abs/1708.02002 for details.
)DOC")
    .Arg(
        "scale",
        "(float) default 1.0; multiply the loss by this scale factor.")
    .Arg("alpha", "(float) default 0.25; Focal Loss's alpha hyper-parameter.")
    .Arg("gamma", "(float) default 1.0; Focal Loss's gamma hyper-parameter.")
    .Arg(
        "num_classes",
        "(int) default 81; number of classes in each softmax group.")
    .Input(
        0,
        "scores",
        "4D tensor of softmax inputs (called 'scores' or 'logits') with shape "
        "(N, C, H, W), where C = num_anchors * num_classes defines num_anchors "
        "groups of contiguous num_classes softmax inputs.")
    .Input(
        1,
        "labels",
        "4D tensor of labels with shape (N, num_anchors, H, W). Each entry is "
        "a class label in [0, num_classes - 1] (inclusive).")
    .Input(
        2,
        "normalizer",
        "Scalar; the loss is normalized by 1 / max(1, normalizer).")
    .Output(0, "loss", "Scalar loss.")
    .Output(
        1,
        "probabilities",
        "4D tensor of softmax probabilities with shape (N, C, H, W), where "
        "C = num_anchors * num_classes, and softmax was applied to each of the "
        "num_anchors groups; within a group the num_classes values sum to 1.");

OPERATOR_SCHEMA(SoftmaxFocalLossGradient)
    .NumInputs(5)
    .NumOutputs(1)
    .Input(0, "scores", "See SoftmaxFocalLoss.")
    .Input(1, "labels", "See SoftmaxFocalLoss.")
    .Input(2, "normalizer", "See SoftmaxFocalLoss.")
    .Input(
        3,
        "probabilities",
        "Output 1 from SoftmaxFocalLoss; See SoftmaxFocalLoss.")
    .Input(4, "d_loss", "Gradient of forward output 0 (loss)")
    .Output(0, "d_scores", "Gradient of forward input 0 (scores)");

class GetSoftmaxFocalLossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SoftmaxFocalLossGradient",
        "",
        vector<string>{I(0), I(1), I(2), O(1), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(SoftmaxFocalLoss, GetSoftmaxFocalLossGradient);

} // namespace caffe2
