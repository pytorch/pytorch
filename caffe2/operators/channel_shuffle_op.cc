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

#include "channel_shuffle_op.h"

namespace caffe2 {

class GetChannelShuffleGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        vector<string>{GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_CPU_OPERATOR(ChannelShuffle, ChannelShuffleOp<CPUContext>);
REGISTER_CPU_OPERATOR(
    ChannelShuffleGradient,
    ChannelShuffleGradientOp<CPUContext>);
REGISTER_GRADIENT(ChannelShuffle, GetChannelShuffleGradient);
OPERATOR_SCHEMA(ChannelShuffle)
    .IdenticalTypeAndShape()
    .NumInputs(1)
    .NumOutputs(1)
    .InheritOnnxSchema("ChannelShuffle");
OPERATOR_SCHEMA(ChannelShuffleGradient)
    .IdenticalTypeAndShape()
    .NumInputs(1)
    .NumOutputs(1);
}
