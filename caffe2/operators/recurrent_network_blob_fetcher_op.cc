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

#include "caffe2/operators/recurrent_network_blob_fetcher_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    RecurrentNetworkBlobFetcher,
    RecurrentNetworkBlobFetcherOp<CPUContext>);

OPERATOR_SCHEMA(RecurrentNetworkBlobFetcher)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Retrieves blobs from scratch workspaces (which contain intermediate recurrent
network computation for each timestep) and puts them in the global
workspace under CPUContext.
)DOC")
    .Arg("prefix", "Prefix string to prepend extracted blobs.")
    .Input(
        0,
        "ScratchWorkspaceBlob",
        "Name of scratch workspace blob returned by recurrent network.")
    .Output(
        0,
        "blob_names",
        "1D tensor of strings containing extracted blob names.");

SHOULD_NOT_DO_GRADIENT(RecurrentNetworkBlobFetcher);
} // namespace caffe2
