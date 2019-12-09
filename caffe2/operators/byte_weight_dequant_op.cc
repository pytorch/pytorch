#include "caffe2/operators/byte_weight_dequant_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ByteWeightDequant, ByteWeightDequantOp<CPUContext>);

OPERATOR_SCHEMA(ByteWeightDequant).NumInputs(1).NumOutputs(1);

} // namespace caffe2
