#include "variable_length_sequence_padding.h"

namespace caffe2 {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    VariableLengthSequencePadding,
    VariableLengthSequencePaddingOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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
