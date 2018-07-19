#include "caffe2/operators/relu_op.h"

#include "caffe2/operators/activation_ops_cudnn.h"

namespace caffe2 {

REGISTER_CUDNN_OPERATOR(Relu, CuDNNActivationOp<CUDNN_ACTIVATION_RELU>);
REGISTER_CUDNN_OPERATOR(
    ReluGradient,
    CuDNNActivationGradientOp<CUDNN_ACTIVATION_RELU>);

} // namespace caffe2
