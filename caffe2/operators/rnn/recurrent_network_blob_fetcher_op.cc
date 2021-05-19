#include "caffe2/operators/rnn/recurrent_network_blob_fetcher_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    RecurrentNetworkBlobFetcher,
    RecurrentNetworkBlobFetcherOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(RecurrentNetworkBlobFetcher);
} // namespace caffe2
