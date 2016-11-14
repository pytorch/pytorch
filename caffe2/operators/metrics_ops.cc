#include "caffe2/operators/metrics_ops.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(CreateQPSMetric, CreateQPSMetricOp);
REGISTER_CPU_OPERATOR(QPSMetric, QPSMetricOp);
REGISTER_CPU_OPERATOR(QPSMetricReport, QPSMetricReportOp);

OPERATOR_SCHEMA(CreateQPSMetric)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc(R"DOC(
CreateQPSMetric operator create a blob that will store state that is required
for computing QPSMetric. The only output of the operator will have blob with
QPSMetricState as an output.
)DOC")
    .Output(0, "output", "Blob with QPSMetricState");

OPERATOR_SCHEMA(QPSMetric)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
QPSMetric operator syncronously updates metric storedcreate a blob that will
store state that is required for computing QPSMetric. The only output of the
operator will have blob with QPSMetricState as an output.
)DOC")
    .Input(
        0,
        "QPS_METRIC_STATE",
        "Input Blob QPSMetricState, that needs to be updated")
    .Input(
        1,
        "INPUT_BATCH",
        "Input Blob containing a tensor with batch of the examples."
        " First dimension of the batch will be used to get the number of"
        " examples in the batch.")
    .Output(0, "output", "Blob with QPSMetricState")
    .EnforceInplace({{0, 0}});

OPERATOR_SCHEMA(QPSMetricReport)
    .NumInputs(1)
    .NumOutputs(0)
    .SetDoc(R"DOC(
QPSMetricReport operator that syncronously consumes the QPSMetricState blob and
reports the information about QPS.
)DOC")
    .Output(0, "output", "Blob with QPSMetricState");

SHOULD_NOT_DO_GRADIENT(CreateQPSMetric);
SHOULD_NOT_DO_GRADIENT(QPSMetric);
SHOULD_NOT_DO_GRADIENT(QPSMetricReport);
} // namespace
} // namespace caffe2
