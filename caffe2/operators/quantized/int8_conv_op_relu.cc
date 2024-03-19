#include "caffe2/operators/quantized/int8_conv_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Int8ConvRelu, int8::Int8ConvOp<int8::Activation::RELU>);

} // namespace caffe2
