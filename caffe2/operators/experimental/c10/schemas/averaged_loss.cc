#include "caffe2/operators/experimental/c10/schemas/averaged_loss.h"
#include <ATen/core/dispatch/OpSchemaRegistration.h>
#include "caffe2/core/operator_c10wrapper.h"

using caffe2::CPUContext;

C10_DEFINE_OP_SCHEMA(caffe2::ops::AveragedLoss);

namespace caffe2 {

CAFFE_KNOWN_TYPE(ops::AveragedLoss::State);

REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_PARAMETERS(
    ops::AveragedLoss,
    ops::AveragedLoss::State,
    C10AveragedLoss_DontUseThisOpYet)
}
