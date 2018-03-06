/**
 * Copyright (c) 2018-present, Facebook, Inc.
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

#include "variable_length_sequence_padding.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(
    VariableLengthSequencePadding,
    VariableLengthSequencePaddingOp<float, CPUContext>);
OPERATOR_SCHEMA(VariableLengthSequencePadding)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Super special-case operator. Used to pad a tensor to mimic pytorch's
pad_packed_sequence.

Given an input tensor INPUT of size NxBxM and an input tensor LENS
of size B, where

N = maximum sequence length
B = batch size
M = hidden size

set each element of INPUT to zero if it is is past the end of the
corresponding sequence (i.e. if LENS[j] > i for an index (i,j,k)).

)DOC");

} // namespace caffe2
