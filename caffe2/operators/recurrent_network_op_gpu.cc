#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/recurrent_network_op.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(
    RecurrentNetwork,
    RecurrentNetworkOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    RecurrentNetworkGradient,
    RecurrentNetworkGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(
    rnn_internal_accumulate_gradient_input,
    AccumulateInputGradientOp<float, CUDAContext>);
}
