#include "counter_ops.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(CreateCounter, CreateCounterOp<int32_t>);
REGISTER_CPU_OPERATOR(ResetCounter, ResetCounterOp<int32_t>);
REGISTER_CPU_OPERATOR(CountDown, CountDownOp<int32_t>);

OPERATOR_SCHEMA(CreateCounter)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Creates a count-down counter with initial value specified by the 'init_count'
argument.
)DOC")
    .Output(0, "counter", "A blob pointing to an instance of a new counter.")
    .Arg("init_count", "Initial count for the counter, must be >= 0.");

OPERATOR_SCHEMA(ResetCounter)
    .NumInputs(1)
    .NumOutputs(0)
    .SetDoc(R"DOC(
Resets a count-down counter with initial value specified by the 'init_count'
argument.
)DOC")
    .Input(0, "counter", "A blob pointing to an instance of a new counter.")
    .Arg("init_count", "Resets counter to this value, must be >= 0.");

OPERATOR_SCHEMA(CountDown)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
If the internal count value > 0, decreases count value by 1 and outputs false,
otherwise outputs true.
)DOC")
    .Input(0, "counter", "A blob pointing to an instance of a counter.")
    .Output(0, "should_stop", "false unless the internal count is zero.");

SHOULD_NOT_DO_GRADIENT(CreateCounter);
SHOULD_NOT_DO_GRADIENT(ResetCounter);
SHOULD_NOT_DO_GRADIENT(CountDown);

} // namespace

} // namespace caffe2
