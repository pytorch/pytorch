#include "caffe2/operators/piecewise_linear_transform_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(
    PiecewiseLinearTransform,
    PiecewiseLinearTransformOp<float, CPUContext>);

OPERATOR_SCHEMA(PiecewiseLinearTransform)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
PiecewiseLinearTransform takes one inputs- predictions, a 2-D tensor
(Tensor<float>) of size (batch_size x prediction_dimensions), and three args -
upper bounds, slopes and intercepts of piecewise functions. The output tensor
has the same shape of input tensor and contains the piecewise linear
transformation. Each feature dimension has its own piecewise linear
transformation function. Therefore the size of piecewise function parameters are
all (pieces x prediction_dimensions). Note that in each piece, low bound is
excluded while high bound is included. Also the piecewise linear function
must be continuous. If the input is binary predictions (Nx2 tensor), set
the binary arg to true (see details below).
)DOC")
    .Input(
        0,
        "predictions",
        "2-D tensor (Tensor<float>) of size "
        "(num_batches x num_classes) containing scores")
    .Arg(
        "bounds",
        "1-D vector of size (prediction_dimensions x (pieces+1)) contain the "
        "upper bounds of each piece of linear function. One special case is "
        "the first bound is the lower bound of whole piecewise function and we "
        "treat it the same as the left most functions")
    .Arg(
        "slopes",
        "1-D vector of size (prediction_dimensions x pieces) containing the "
        "slopes of linear function")
    .Arg(
        "intercepts",
        "1-D vector of size (prediction_dimensions x pieces) containing the "
        "intercepts of linear function")
    .Arg(
        "pieces",
        "int value for the number of pieces for the piecewise linear function")
    .Arg(
        "binary",
        "If set true, we assume the input is a Nx2 tensor. Its first column is "
        "negative predictions and second column is positive and "
        "negative + positive = 1. We just need one set of transforms for the "
        "positive column.")
    .Output(
        0,
        "transforms",
        "2-D tensor (Tensor<float>) of size (num_batches x num_classes) "
        "containing transformed predictions");

SHOULD_NOT_DO_GRADIENT(PiecewiseLinearTransform);
} // namespace
} // namespace caffe2
