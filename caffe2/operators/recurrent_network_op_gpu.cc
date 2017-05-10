#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/recurrent_network_op.h"

namespace caffe2 {
namespace {
REGISTER_CUDA_OPERATOR(
    RecurrentNetwork,
    RecurrentNetworkOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    RecurrentNetworkGradient,
    RecurrentNetworkGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(
    rnn_internal_accumulate_gradient_input,
    RNNAccumulateInputGradientOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    rnn_internal_apply_link,
    RNNApplyLinkOp<float, CUDAContext>);
}
}
