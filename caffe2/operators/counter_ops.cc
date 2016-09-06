#include "counter_ops.h"

namespace caffe2 {
namespace {

// TODO(jiayq): deprecate these ops & consolidate them with IterOp/AtomicIterOp

REGISTER_CPU_OPERATOR(CreateCounter, CreateCounterOp<int64_t, CPUContext>);
REGISTER_CPU_OPERATOR(ResetCounter, ResetCounterOp<int64_t, CPUContext>);
REGISTER_CPU_OPERATOR(CountDown, CountDownOp<int64_t, CPUContext>);
REGISTER_CPU_OPERATOR(CountUp, CountUpOp<int64_t, CPUContext>);
REGISTER_CPU_OPERATOR(RetrieveCount, RetrieveCountOp<int64_t, CPUContext>);

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
    .Output(0, "done", "false unless the internal count is zero.");

OPERATOR_SCHEMA(CountUp)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Increases count value by 1 and outputs the previous value atomically
)DOC")
    .Input(0, "counter", "A blob pointing to an instance of a counter.")
    .Output(0, "previous_count", "count value BEFORE this operation");

OPERATOR_SCHEMA(RetrieveCount)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Retrieve the current value from the counter.
)DOC")
    .Input(0, "counter", "A blob pointing to an instance of a counter.")
    .Output(0, "count", "current count value.");

SHOULD_NOT_DO_GRADIENT(CreateCounter);
SHOULD_NOT_DO_GRADIENT(ResetCounter);
SHOULD_NOT_DO_GRADIENT(CountDown);
SHOULD_NOT_DO_GRADIENT(CountUp);
SHOULD_NOT_DO_GRADIENT(RetrieveCount);

} // namespace

} // namespace caffe2
