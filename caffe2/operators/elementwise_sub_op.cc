#include "caffe2/operators/elementwise_sub_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    Sub,
    BinaryElementwiseOp<NumericTypes, CPUContext, SubFunctor<CPUContext>>);

} // namespace caffe2
