#include "caffe2/operators/length_split_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(LengthsSplit, LengthsSplitOp<CPUContext>);

OPERATOR_SCHEMA(LengthsSplit)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .ScalarType(TensorProto::INT32)
    .SetDoc(R"DOC(
Given input vector LENGTHS, and input n_splits, LengthsSplit returns
a single output vector. It "splits" each length into n_splits values which add
up to the original length. It will attempt to do equal splits, and if not possible,
it orders larger values first.

e.g. LENGTHS = [9 4 5]
     n_splits = 3
     Y = [3 3 3 2 1 1 2 2 1]
)DOC")
    .Arg("n_splits", "Number of splits for each element in LENGTHS")
    .Input(0, "LENGTHS", "Mx1 Input tensor denoting INT32 lengths")
    .Input(
        1,
        "n_splits",
        "(Optional) Number of splits for each element in LENGTHS (overrides argument)")
    .Output(0, "Y", "(M*n_splits)x1 Output vector denoting split lengths");

// TODO: Write gradient for this when needed
GRADIENT_NOT_IMPLEMENTED_YET(LengthsSplit);

} // namespace caffe2
