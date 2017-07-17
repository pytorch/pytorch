#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/recurrent_network_blob_fetcher_op.h"

namespace caffe2 {
namespace {
REGISTER_CUDA_OPERATOR(
    RecurrentNetworkBlobFetcher,
    RecurrentNetworkBlobFetcherOp<CUDAContext>);
} // namespace
} // namespace caffe2
