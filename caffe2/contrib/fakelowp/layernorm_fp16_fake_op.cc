#include "layernorm_fp16_fake_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(LayerNormFakeFP16, LayerNormFakeFp16Op);

}
