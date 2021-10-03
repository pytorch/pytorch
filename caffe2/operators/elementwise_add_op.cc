#include "caffe2/operators/elementwise_add_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    Add,
    BinaryElementwiseBroadcastOp<NumericTypes, CPUContext, AddFunctor<CPUContext>>);

} // namespace caffe2
