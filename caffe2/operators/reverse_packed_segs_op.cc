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

#include "caffe2/operators/reverse_packed_segs_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(ReversePackedSegs, ReversePackedSegsOp<CPUContext>);

OPERATOR_SCHEMA(ReversePackedSegs)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Reverse segments in a 3-D tensor (lengths, segments, embeddings,), leaving
paddings unchanged. This operator is used to reverse input of a recurrent neural
network to make it a BRNN.
  )DOC")
    .Input(0, "data", "a 3-D (lengths, segments, embeddings,) tensor.")
    .Input(1, "lengths", "length of each segment.")
    .Output(
        0,
        "reversed data",
        "a (lengths, segments, embeddings,) tensor with each segment reversed"
        "and paddings unchanged.");

class GetReversePackedSegsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ReversePackedSegs",
        "",
        vector<string>{GO(0), I(1)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(ReversePackedSegs, GetReversePackedSegsGradient);
} // namespace caffe2
