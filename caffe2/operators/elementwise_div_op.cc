#include "caffe2/operators/elementwise_div_op.h"


namespace caffe2 {

REGISTER_CPU_OPERATOR(
    Div,
    BinaryElementwiseBroadcastOp<NumericTypes, CPUContext, DivFunctor<CPUContext>>);

} // namespace caffe2
