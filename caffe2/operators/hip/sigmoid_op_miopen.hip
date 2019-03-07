#include "caffe2/operators/sigmoid_op.h"

#include "caffe2/operators/hip/activation_ops_miopen.h"

namespace caffe2 {

REGISTER_MIOPEN_OPERATOR(Sigmoid, MIOPENActivationOp<miopenActivationLOGISTIC>);
REGISTER_MIOPEN_OPERATOR(
    SigmoidGradient,
    MIOPENActivationGradientOp<miopenActivationLOGISTIC>);

} // namespace caffe2
