#include "caffe2/operators/relu_op.h"

#include "caffe2/operators/hip/activation_ops_miopen.h"

namespace caffe2 {

REGISTER_MIOPEN_OPERATOR(Relu, MIOPENActivationOp<miopenActivationRELU>);
REGISTER_MIOPEN_OPERATOR(
    ReluGradient,
    MIOPENActivationGradientOp<miopenActivationRELU>);

} // namespace caffe2
