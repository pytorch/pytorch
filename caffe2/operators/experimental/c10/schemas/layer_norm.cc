#include "caffe2/operators/layer_norm_op.h"
#include "caffe2/core/operator_c10wrapper.h"

namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(
    caffe2::_c10_ops::LayerNorm(),
    C10LayerNorm_DontUseThisOpYet)
}
