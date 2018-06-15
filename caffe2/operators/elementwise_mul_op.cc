#include "caffe2/operators/elementwise_mul_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    Mul,
    BinaryElementwiseOp<NumericTypes, CPUContext, MulFunctor<CPUContext>>);

} // namespace caffe2
