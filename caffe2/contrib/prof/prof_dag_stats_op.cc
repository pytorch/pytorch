#include "prof_dag_stats_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(GetProfDagStats, GetProfDagStatsOp<float, CPUContext>);

OPERATOR_SCHEMA(GetProfDagStats)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Gets the profiling statistics.
)DOC")
    .Arg(
        "per_op",
        "(bool) default to false; False: calculate per-op-type cost."
        "True: calculate per-op cost, the cost of multiple instances of the same "
        "op will be calculated separately")
    .Arg(
        "partial_net_name",
        "(string) default to empty; describes the partial name of the ProfDAGNet")
    .Arg(
        "net_name",
        "(string) default to empty; describes the name of the ProfDAGNet");
} // namespace
} // namespace caffe2
