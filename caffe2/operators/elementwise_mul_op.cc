#include "caffe2/operators/elementwise_mul_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    Mul,
    BinaryElementwiseOp<NumericTypes, CPUContext, MulFunctor<CPUContext>>);

} // namespace caffe2
