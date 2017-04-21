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
        "net_name",
        "(string) default to empty; describes the name of the ProfDAGNet");
} // namespace
} // namespace caffe2
