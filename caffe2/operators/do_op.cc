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

#include "caffe2/operators/do_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Do, DoOp<CPUContext>);

OPERATOR_SCHEMA(Do)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1, INT_MAX)
    .SetDoc(R"DOC(
'Do' control operator, executes a subnet in a separate workspace.
Last blobs in the input and output lists should be the same blob created with
CreateScope op. Arguments 'inner_blobs' and 'outer_blobs_idx' provide a mapping
between selected inner blob names and corresponding outer blob indices.
    )DOC")
    .Arg("net", "Subnet with blob bindings")
    .Arg(
        "inner_blobs",
        "List of inner net blob names to bind to outer workspace")
    .Arg(
        "outer_blobs_idx",
        "Indices of corresponding outer workspace blobs, "
        "in order: operator inputs, operator outputs (skipping workspace blobs)")
    .Arg(
        "saved_fwd_blobs",
        "List of blobs from the forward Do operator workspace needed "
        "in backward pass, used in gradient Do operator")
    .Arg(
        "reuse_workspace",
        "Whether to reuse workspace or create a new one in a given scope")
    .AllowInplace([](int /*in*/, int /*out*/) -> bool { return true; });

} // namespace caffe2
