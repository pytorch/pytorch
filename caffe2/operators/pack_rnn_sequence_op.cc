#include "caffe2/operators/pack_rnn_sequence_op.h"

namespace caffe2 {
namespace {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(PackRNNSequence, PackRNNSequenceOpBase<CPUContext, true>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    UnpackRNNSequence,
    PackRNNSequenceOpBase<CPUContext, false>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(PackRNNSequence)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Pack values based on the length blob. Each number from length blob represents
the corresponding values that need to be packed. The dimension for each pack
is the same as the maximum number from the length blob (padding with zero is
implemented for smaller length value). The overall output dimension is:
T * N * D, where T is the max number of lengths, N is the size of lengths,
and D is the dimension of each feature value. The following example shows
the input and output of this operator:


Given:
  values = [v1, v2, v3, v4, v5, v6, v7, v8]
  lengths = [2, 3, 1, 2];


Output:
  output = [
    [v1, v3, v6, v7],
    [v2, v4, 0,  v8],
    [0,  v5, 0,  0 ],
  ]


One application for this operator is the transfer data into the format that is
used for RNN models. Note that the gradient operator of PackRNNSequence is
UnpackRNNSequence.
)DOC")
    .Input(0, "values", "Data tensor, contains a sequence of features")
    .Input(1, "lengths", "lengths with each number representing the pack size.")
    .Output(0, "output", "Output tensor after packing");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(UnpackRNNSequence)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
This is the reverse operator for PackRNNSequence. It maps the packed values
back to sequence values based on the length blob. Each number from length blob
represents the corresponding values that has been grouped. The dimension
for each pack is the same as the maximum number from the length blob (padding
with zero was implemented for smaller length value). The overall output
dimension is: M * D, where M is the sum of lengths, and D is the dimension of
each feature value. The following example shows the input and output of
this operator:


Given:
  values = [
    [v1, v3, v6, v7],
    [v2, v4, 0,  v8],
    [0,  v5, 0,  0 ],
  ]
  lengths = [2, 3, 1, 2]


Output:
  output = [v1, v2, v3, v4, v5, v6, v7, v8];


One application for this operator is the transfer data from the format of RNN
back to sequence values. Note that the gradient operator of
UnpackRNNSequence is PackRNNSequence.
)DOC")
    .Input(0, "values", "Data tensor, contains the packed features")
    .Input(1, "lengths", "lengths with each number representing the pack size.")
    .Output(0, "output", "Output tensor before packing");

class GetPackRNNSequenceGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE_EQ(def_.input_size(), 2);
    return SingleGradientDef(
        "UnpackRNNSequence",
        "",
        vector<string>{GO(0), I(1)},
        vector<string>{GI(0)});
  }
};

class GetUnpackRNNSequenceGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE_EQ(def_.input_size(), 2);
    return SingleGradientDef(
        "PackRNNSequence",
        "",
        vector<string>{GO(0), I(1)},
        vector<string>{GI(0)});
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(PackRNNSequence, GetPackRNNSequenceGradient);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(UnpackRNNSequence, GetUnpackRNNSequenceGradient);
} // namespace
} // namespace caffe2
