#include "caffe2/operators/partition_ops.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(Partition, PartitionOp);
REGISTER_CPU_OPERATOR(LengthsPartition, LengthsPartitionOp);
REGISTER_CPU_OPERATOR(GatherByKey, GatherByKeyOp);

OPERATOR_SCHEMA(GatherByKey)
    .NumInputs(2, INT_MAX)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Inverse operation of Partition.

Takes the original, full 'keys' tensor followed by sharded value tensors,
and returns the full value tensor, combined using the same hash used in
Partition.
)DOC")
    .Input(
        0,
        "keys",
        "The first input is the full keys tensor"
        " (same as the first input of Partition).")
    .Input(
        1,
        "sharded_values",
        "Subsequented inputs are sharded values tensors.")
    .Output(0, "values", "Reconstructed values tensor.");

OPERATOR_SCHEMA(Partition)
    .NumInputsOutputs([](int in, int out) {
      return in > 0 && out > 0 && out % in == 0;
    })
    .SetDoc(R"DOC(
Splits the input int tensor into multiple ones according to the first tensor.

Takes the first input and partitions it to shards according to the remainder of
values modulo the number of partitions. It requires that the first tensor is of
integral type. The number of partitions is derived as (num_output / num_input).

If additional inputs are present they must have the same shape as the first
input, optionally with extra trailing dimensions. They will be partitioned
accordingly to the first input.

Optional arg 'pack_first_input' transforms the first tensor values as
X_ij / num_partitions.

Outputs are ordered as
X_0_part_0, X_1_part_0, ..., X_N-1_part_0, X_0_part_1, ..., X_N-1_part_K-1
)DOC")
    .Arg(
        "pack_first_input",
        "(int, default 0) If set, the operator transforms "
        "the first tensor values as floor(X_ij / num_partitions)")
    .Input(
        0,
        "input",
        "Input tensor containing data to be partitioned. The "
        "number of input tensors might be greater than 1 but must have the "
        "same shape as the previous tensors.")
    .Output(
        0,
        "partitions",
        "Output Partitions. The number of output tensors has to be a "
        "multiple of the number of input tensors.");

OPERATOR_SCHEMA(LengthsPartition)
    .NumInputsOutputs([](int in, int out) {
      return in >= 2 && out > 0 && out % in == 0;
    })
    .SetDoc(R"DOC(
LengthsPartition splits the input int tensor into multiple ones according to the
second tensor. The first dimension is expected to be the tensor that describes
lengths of the elements.

Takes the second input and partitions it to shards according to the remainder of
values modulo the number of partitions. It requires the second tensor to be
a 1D-tensor of the integral type. The first tensor should be 1D-tensor of int32
that would represent the lengths of the elements in the input. The number of
partitions is derived as (num_output / num_input).

If additional inputs are present they must have the same shape as the first
input, optionally with extra trailing dimensions. They will be partitioned
accordingly to the first input.

Optional arg 'pack_first_input' transforms the first tensor values as
X_ij / num_partitions.

Outputs are ordered as
X_0_part_0, X_1_part_0, ..., X_N-1_part_0, X_0_part_1, ..., X_N-1_part_K-1
)DOC")
    .Arg(
        "pack_first_input",
        "(int, default 0) If set, the operator transforms "
        "the first tensor values as floor(X_ij / num_partitions)")
    .Input(
        0,
        "input",
        "Input tensor containing data to be partitioned. The "
        "number of input tensors might be greater than 1 but must have the "
        "same shape as the previous tensors.")
    .Output(
        0,
        "partitions",
        "Output Partitions. The number of output tensors has to be a "
        "multiple of the number of input tensors.");

namespace {

class GetGatherByKeyGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    ArgumentHelper argsHelper(def_);
    auto pack_first_input =
        argsHelper.GetSingleArgument<int>("pack_first_input", 0);

    Argument packArg = MakeArgument<int>("pack_first_input", pack_first_input);
    if (g_output_[0].IsDense()) {
      std::vector<std::string> inputs;
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      for (int i = 1; i < g_input_.size(); ++i) {
        inputs.push_back("_" + GI(i) + "_keys");
        inputs.push_back(GI(i));
      }
      return SingleGradientDef(
          "Partition",
          "",
          std::vector<std::string>{I(0), GO(0)},
          inputs,
          std::vector<Argument>{packArg});
    } else {
      std::vector<std::string> inputs;
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      for (int i = 1; i < g_input_.size(); ++i) {
        inputs.push_back("_" + GI_I(i) + "_keys");
        inputs.push_back(GI_I(i));
        inputs.push_back(GI_V(i));
      }
      return SingleGradientDef(
          "Partition",
          "",
          std::vector<std::string>{I(0), GO_I(0), GO_V(0)},
          inputs,
          std::vector<Argument>{packArg});
    }
  }
};

} // namespace

// This should actually have gradient, but for now nothing uses it.
// Because gradient computation right now is not input/output aware it can't be
// GRADIENT_NOT_IMPLEMENTEDYET
NO_GRADIENT(Partition);
NO_GRADIENT(LengthsPartition);
REGISTER_GRADIENT(GatherByKey, GetGatherByKeyGradient);
} // namespace
} // namespace caffe2
