#include "caffe2/operators/im2col_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(Im2Col, Im2ColOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(Col2Im, Col2ImOp<float, CPUContext>);

OPERATOR_SCHEMA(Im2Col)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc("The Im2Col operator from Matlab.")
    // TODO: add Shape inference function (bootcamp)
    .Input(0, "X", "4-tensor in NCHW or NHWC.")
    .Output(
        0,
        "Y",
        "4-tensor. For NCHW: N x (C x kH x kW) x outH x outW."
        "For NHWC: N x outH x outW x (kH x kW x C");

OPERATOR_SCHEMA(Col2Im).NumInputs(2).NumOutputs(1);

} // namespace
} // namespace caffe2
