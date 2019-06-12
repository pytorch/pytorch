#include "caffe2/operators/experimental/c10/schemas/averaged_loss.h"
#include "caffe2/core/dispatch/OpSchemaRegistration.h"
#include "caffe2/core/operator_c10wrapper.h"

using caffe2::CPUContext;
using caffe2::Tensor;

C10_DEFINE_OP_SCHEMA(caffe2::ops::AveragedLoss);

namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_PARAMETERS(
    ops::AveragedLoss,
    ops::AveragedLoss::State,
    C10AveragedLoss_DontUseThisOpYet)
}
