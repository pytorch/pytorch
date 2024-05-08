#include "caffe2/operators/tanh_op.h"

#include "caffe2/operators/hip/activation_ops_miopen.h"

namespace caffe2 {

REGISTER_MIOPEN_OPERATOR(Tanh, MIOPENActivationOp<miopenActivationTANH>);
REGISTER_MIOPEN_OPERATOR(
    TanhGradient,
    MIOPENActivationGradientOp<miopenActivationTANH>);

} // namespace caffe2
