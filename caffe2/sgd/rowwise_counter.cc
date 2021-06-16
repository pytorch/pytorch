#include "rowwise_counter.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(RowWiseCounter, RowWiseCounterOp);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(RowWiseCounter)
    .NumInputs(4)
    .NumOutputs(2)
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(
    Count the number recent update on rows. Exponential decay is
    applied on the counter with decay rate r, such that
    r^{counter_halflife} = 0.5; If counter_halflife is nonpositive,
    this operator is turned off.
)DOC")
    .Input(0, "prev_iter", "Iter at last update")
    .Input(1, "update_counter", "update counter")
    .Input(2, "indices", "Sparse indices")
    .Input(3, "iter", "current iteration")
    .Output(0, "output_prev_iter", "Updated iter at last update")
    .Output(1, "output_update_counter", "Output update counter")
    .Arg("counter_halflife", "Default -1: off");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(RowWiseCounter);
} // namespace caffe2
