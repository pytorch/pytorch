#include "caffe2/operators/fully_connected_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(FC, FullyConnectedOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(FCGradient, FullyConnectedGradientOp<float, CPUContext>);

}  // namespace
}  // namespace caffe2
