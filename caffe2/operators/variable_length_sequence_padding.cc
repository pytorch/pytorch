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

set each element of INPUT to zero if it is past the end of the
corresponding sequence (i.e. if LENS[j] > i for an index (i,j,k)).

)DOC");

} // namespace caffe2
