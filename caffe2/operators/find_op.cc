#include "caffe2/operators/find_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Find)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(1)
    .Input(0, "index", "Index (integers)")
    .Input(1, "query", "Needles / query")
    .Output(
        0,
        "query_indices",
        "Indices of the needles in index or 'missing value'")
    .Arg("missing_value", "Placeholder for items that are not found")
    .SetDoc(R"DOC(
Finds elements of second input from first input,
outputting the last (max) index for each query.
If query not find, inserts missing_value.
See IndexGet() for a version that modifies the index when
values are not found.
)DOC");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Find, FindOp<CPUContext>)

} // namespace caffe2
