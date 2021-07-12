#include "caffe2/operators/byte_weight_dequant_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(ByteWeightDequant, ByteWeightDequantOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(ByteWeightDequant).NumInputs(1).NumOutputs(1);

} // namespace caffe2
