#include "caffe2/operators/sigmoid_op.h"

#include "caffe2/operators/activation_ops_cudnn.h"

namespace caffe2 {

REGISTER_CUDNN_OPERATOR(Sigmoid, CuDNNActivationOp<CUDNN_ACTIVATION_SIGMOID>);
REGISTER_CUDNN_OPERATOR(
    SigmoidGradient,
    CuDNNActivationGradientOp<CUDNN_ACTIVATION_SIGMOID>);

} // namespace caffe2
