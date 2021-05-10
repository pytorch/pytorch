#include "gru_unit_op.h"

namespace caffe2 {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(GRUUnit, GRUUnitOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(GRUUnit)
    .NumInputs(3, 4)
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
    .Arg(
        "sequence_lengths",
        "When false, the sequence lengths input is left out, "
        "and all following inputs are shifted left by one.")
    .Output(0, "hidden", "The new GRU hidden state calculated by this op.");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(GRUUnitGradient, GRUUnitGradientOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(GRUUnitGradient)
    .NumInputs(5, 6)
    .NumOutputs(2)
    .Arg(
        "sequence_lengths",
        "When false, the sequence lengths input is left out, "
        "and all following inputs are shifted left by one.");

class GetGRUUnitGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    if (GetFlagArgument(def_, "sequence_lengths", true)) {
      return SingleGradientDef(
          "GRUUnitGradient",
          "",
          vector<string>{I(0), I(1), I(2), I(3), O(0), GO(0)},
          vector<string>{GI(0), GI(1)});
    } else {
      return SingleGradientDef(
          "GRUUnitGradient",
          "",
          vector<string>{I(0), I(1), I(2), O(0), GO(0)},
          vector<string>{GI(0), GI(1)});
    }
  }
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(GRUUnit, GetGRUUnitGradient);
} // namespace caffe2
