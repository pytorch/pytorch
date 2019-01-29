#include "caffe2/operators/experimental/c10/schemas/batch_gather.h"
#include <ATen/core/dispatch/OpSchemaRegistration.h>
#include "caffe2/core/operator_c10wrapper.h"

using caffe2::CPUContext;

C10_DEFINE_OP_SCHEMA(caffe2::ops::BatchGather);

namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(
    ops::BatchGather,
    C10BatchGather_DontUseThisOpYet)
}
