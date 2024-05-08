#include "caffe2/operators/elementwise_sub_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    Sub,
    BinaryElementwiseOp<NumericTypes, CPUContext, SubFunctor<CPUContext>>);

} // namespace caffe2
