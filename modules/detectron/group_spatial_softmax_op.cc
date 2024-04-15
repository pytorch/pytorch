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

#include "modules/detectron/group_spatial_softmax_op.h"

#include "caffe2/operators/softmax_utils.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    GroupSpatialSoftmax,
    GroupSpatialSoftmaxOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    GroupSpatialSoftmaxGradient,
    GroupSpatialSoftmaxGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(GroupSpatialSoftmax)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
RetinaNet specific form of spatial softmax.

The input is assumed to be unnormalized scores (sometimes called 'logits')
arranged in a 4D tensor with shape (N, C, H, W), where N is the number of
elements in the batch, H and W are the height and width, and C = num_anchors *
num_classes defines num_anchors 'groups' of softmax inputs, each of length
num_classes. The softmax is applied to each group independently.

See: https://arxiv.org/abs/1708.02002 for details.
)DOC")
    .Arg(
        "num_classes",
        "(int) default 81; number of classes in each softmax group.")
    .Input(
        0,
        "scores",
        "4D tensor of softmax inputs (called 'scores' or 'logits') with shape "
        "(N, C, H, W), where C = num_anchors * num_classes defines num_anchors "
        "groups of contiguous num_classes softmax inputs.")
    .Output(
        0,
        "probabilities",
        "4D tensor of softmax probabilities with shape (N, C, H, W), where "
        "C = num_anchors * num_classes, and softmax was applied to each of the "
        "num_anchors groups; within a group the num_classes values sum to 1.");

OPERATOR_SCHEMA(GroupSpatialSoftmaxGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .Input(0, "scores", "See GroupSpatialSoftmax")
    .Input(
        1,
        "d_probabilities",
        "Gradient of forward output 0 (probabilities).")
    .Output(0, "d_scores", "Gradient of forward input 0 (scores).");

class GetGroupSpatialSoftmaxGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "GroupSpatialSoftmaxGradient",
        "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(GroupSpatialSoftmax, GetGroupSpatialSoftmaxGradient);

} // namespace caffe2
