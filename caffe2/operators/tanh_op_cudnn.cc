#include "caffe2/operators/tanh_op.h"

#include "caffe2/operators/activation_ops_cudnn.h"

namespace caffe2 {

REGISTER_CUDNN_OPERATOR(Tanh, CuDNNActivationOp<CUDNN_ACTIVATION_TANH>);
REGISTER_CUDNN_OPERATOR(
    TanhGradient,
    CuDNNActivationGradientOp<CUDNN_ACTIVATION_TANH>);

} // namespace caffe2
