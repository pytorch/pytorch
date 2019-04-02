#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(Add, AddOp<float, CPUContext>)
REGISTER_CPU_OPERATOR(Sub, SubOp<float, CPUContext>)
REGISTER_CPU_OPERATOR(Mul, MulOp<float, CPUContext>)
REGISTER_CPU_OPERATOR(Div, DivOp<float, CPUContext>)

}  // namespace
}  // namespace caffe2
