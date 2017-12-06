/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
