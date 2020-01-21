#include "rowwise_prune_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    RowwisePruneFloatUInt8,
    RowwisePruneOp<float, std::uint8_t, CPUContext>);
OPERATOR_SCHEMA(RowwisePruneFloatUInt8)
    .NumInputs(3)
    .NumOutputs(2)
    .EnforceInplace({{0, 0}})
    .SetDoc(R"DOC(
    Prune the rows of the input tensor (inplace), based on the provided indicator and threshold;
    Assume tensor is of type Uint8 and the indicator of type float;
)DOC")
    .Input(0, "X", "2-D tensor to be pruned")
    .Input(
        1,
        "indicator",
        "1-D tensor with same number of element as the number of rows of X")
    .Input(
        2,
        "threshold",
        "1-D tensor with one element: threshold, to be used together with indictor for pruning:"
        " rows with indicator value lessEq to the threshold will be pruned.")
    .Output(0, "X_output", "pruned 2-D tensor")
    .Output(
        1,
        "COMPRESSED_INDICES_MAPPING",
        "Integer vector mapping uncompressed indices to compressed indices")
    .Arg("abs", "If true, apply abs() on the indicator values.");

SHOULD_NOT_DO_GRADIENT(RowwisePruneFloatUInt8);
} // namespace caffe2
