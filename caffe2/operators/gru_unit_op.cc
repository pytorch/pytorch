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

#include "gru_unit_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(GRUUnit, GRUUnitOp<float, CPUContext>);
OPERATOR_SCHEMA(GRUUnit)
    .NumInputs(4)
    .NumOutputs(1)
    .SetDoc(R"DOC(
GRUUnit computes the activations of a standard GRU,
in a sequence-length aware fashion.

Concretely, given the (fused) inputs X (TxNxD), the previous hidden
state (NxD), and the sequence lengths (N), computes the GRU
activations, avoiding computation if the input is invalid (as in, the
value at X[t][n] >= seqLengths[n].

)DOC")
    .Arg(
        "drop_states",
        "Bool to determine if hidden state is zeroes or passed "
        "along for timesteps past the given sequence_length.")
    .Input(0, "hidden_prev", "The previous GRU hidden state.")
    .Input(
        1,
        "gates",
        "Unactivated gate outputs from forget, update, "
        "and output gates, pre-activation.")
    .Input(
        2,
        "seq_lengths",
        "Array of sequence lengths.  "
        "len(seq_lengths) should equal batch size N.")
    .Input(3, "t", "The timestep for this operation.")
    .Output(0, "hidden", "The new GRU hidden state calculated by this op.");
REGISTER_CPU_OPERATOR(GRUUnitGradient, GRUUnitGradientOp<float, CPUContext>);
OPERATOR_SCHEMA(GRUUnitGradient).NumInputs(6).NumOutputs(2);

class GetGRUUnitGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "GRUUnitGradient",
        "",
        vector<string>{I(0), I(1), I(2), I(3), O(0), GO(0)},
        vector<string>{GI(0), GI(1)});
  }
};
REGISTER_GRADIENT(GRUUnit, GetGRUUnitGradient);
} // namespace caffe2
