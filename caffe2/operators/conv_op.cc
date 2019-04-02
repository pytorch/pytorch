#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_op_impl.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(Conv, ConvOp<float, CPUContext>)
REGISTER_CPU_OPERATOR(ConvGradient, ConvGradientOp<float, CPUContext>)

}  // namespace
}  // namespace caffe2
