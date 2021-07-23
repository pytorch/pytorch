#include "caffe2/operators/length_split_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(LengthsSplit, LengthsSplitOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(LengthsSplit)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .ScalarType(TensorProto::INT32)
    .SetDoc(R"DOC(
Given input vector LENGTHS, and input n_split, LengthsSplit returns
a single output vector. It "splits" each length into n_split values which add
up to the original length. It will attempt to do equal splits, and if not possible,
it orders larger values first. If the n_split is larger than the length, zero
padding will be applied.

e.g. LENGTHS = [9 4 5]
     n_split = 3
     Y = [3 3 3 2 1 1 2 2 1]

e.g. LENGTHS = [2, 1, 2]
     n_split = 3
     Y = [1 1 0 1 0 0 1 1 0]
)DOC")
    .Arg("n_split", "Number of splits for each element in LENGTHS")
    .Input(0, "LENGTHS", "Mx1 Input tensor denoting INT32 lengths")
    .Input(
        1,
        "n_split",
        "(Optional) Number of splits for each element in LENGTHS (overrides argument)")
    .Output(0, "Y", "(M*n_split)x1 Output vector denoting split lengths");

// TODO: Write gradient for this when needed
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
GRADIENT_NOT_IMPLEMENTED_YET(LengthsSplit);

} // namespace caffe2
