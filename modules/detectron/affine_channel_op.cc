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

#include "affine_channel_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(AffineChannel,
                      AffineChannelOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(AffineChannelGradient,
                      AffineChannelGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(AffineChannel)
    .NumInputs(3)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Applies a separate affine transformation to each channel of the input. Useful
for replacing spatial batch norm with its equivalent fixed transformation.
)DOC")
    .Input(
        0,
        "X",
        "4D feature map input of shape (N, C, H, W).")
    .Input(
        1,
        "scale",
        "1D input of shape (C); the c-th element is the scale factor of the "
        "affine transformation for the c-th channel of the input.")
    .Input(
        2,
        "bias",
        "1D input of shape (C); the c-th element is the bias of the affine "
        "transformation for the c-th channel of the input.")
    .Output(
        0,
        "Y",
        "4D output of shape (N, C, H, W).");

OPERATOR_SCHEMA(AffineChannelGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}})
    .Input(
        0,
        "scale",
        "See AffineChannel.")
    .Input(
        1,
        "dY",
        "Gradient of forward output 0 (Y)")
    .Output(
        0,
        "dX",
        "Gradient of forward input 0 (X)");

class GetAffineChannelGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "AffineChannelGradient", "",
        vector<string>{I(1), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(AffineChannel, GetAffineChannelGradient);

} // namespace caffe2
