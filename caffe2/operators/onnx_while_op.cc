#include "caffe2/operators/onnx_while_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ONNXWhile, ONNXWhileOp<CPUContext>);

OPERATOR_SCHEMA(ONNXWhile)
    .NumInputs(2, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .SetDoc(R"DOC(
*** EXPERIMENTAL. This operator is a work-in-progress. No assumption should be
made about the stability or correctness of this op. ***

Generic Looping construct confirming to the ONNX Loop operator spec. This loop
has multiple termination conditions:

1. Trip count. Iteration count specified at runtime. Set by specifying the
    input M. Optional. Set to empty string to omit. Note that a static trip
    count (specified at graph construction time) can be specified by passing
    in a constant node for input M.
2. Loop termination condition. This is an input to the op that determines
    whether to run the first interation and also a loop-carried dependency for
    the body graph. The body graph must yield a value for the condition
    variable, whether this input is provided or not.

This table summarizes the operating modes of this operator with equivalent
C-style code:

Operator inputs defined as (max_trip_count, condition_var). Omitted optional
inputs are represented as empty string. Concretely, in this caffe2 op an input
is marked as omitted by setting its 'has_{name}' argument to False.

    input ("", ""):
        for (int i=0; ; ++i) {
          cond = ... // Note this value is ignored, but is required in the body
        }

    input ("", cond) // Note this is analogous to a while loop
        bool cond = ...;
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input ("", 1) // Note this is analogous to a do-while loop
        bool cond = true
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input (trip_count, "") // Note this is analogous to a for loop
        int trip_count = ...
        for (int i=0; i < trip_count; ++i) {
          cond = ...; // ignored
        }

    input (trip_count, cond)
        int trip_count = ...;
        bool cond = ...;
        for (int i=0; i < trip_count && cond; ++i) {
          cond = ...;
        }
    )DOC")
    .Arg("loop_net", "Net executed on each iteration")
    .Input(0, "condition", "Scalar boolean condition")
    .AllowInplace([](int in, int out) -> bool { return true; });

} // namespace caffe2
