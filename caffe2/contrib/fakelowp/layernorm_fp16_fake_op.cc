#include "layernorm_fp16_fake_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(LayerNormFakeFP16, LayerNormFakeFp16Op<CPUContext>);
OPERATOR_SCHEMA(LayerNormFakeFP16).NumInputs(1).NumOutputs(3);

} // namespace caffe2
