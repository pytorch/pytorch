#include "c10_sigmoid_op.h"
#include "caffe2/core/dispatch/OpSchemaRegistration.h"
#include "caffe2/core/operator_c10wrapper.h"

using caffe2::CPUContext;
using caffe2::Tensor;

C10_DEFINE_OP_SCHEMA(caffe2::SigmoidOp);

namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(
    SigmoidOp,
    C10Sigmoid_DontUseThisOpYet)
}
