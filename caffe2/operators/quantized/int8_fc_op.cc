#include "caffe2/operators/quantized/int8_fc_op.h"

#include <functional>

#include "caffe2/operators/fc_inference.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Int8FC, int8::Int8FCOp);

using namespace std::placeholders;
OPERATOR_SCHEMA(Int8FC)
    .NumInputs(3)
    .NumOutputs(1, 4)
    .TensorInferenceFunction(std::bind(FCShapeInference, _1, _2, false))
    .CostInferenceFunction(std::bind(CostInferenceForFC, _1, _2, false))
    .SetDoc(R"DOC(
Computes the result of passing an input vector X into a fully
connected layer with 2D weight matrix W and 1D bias vector b. That is,
the layer computes Y = X * W^T + b, where X has size (M x K),
W has size (N x K), b has size (N), and Y has size (M x N),
where M is often the batch size.


NOTE: X does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
X \in [a_0, a_1 * ... * a_{n-1}]. Only this case is supported!
Lastly, even though b is a 1D vector of size N, it is copied/resized to
be size (M x N) implicitly and added to each vector in the batch.
Each of these dimensions must be matched correctly, or else the operator
will throw errors.
)DOC")
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .Input(
        0,
        "X",
        "input tensor that's coerced into a 2D matrix of size (MxK) "
        "as described above")
    .Input(
        1,
        "W",
        "A tensor that is coerced into a 2D blob of size (KxN) "
        "containing fully connected weight matrix")
    .Input(2, "b", "1D blob containing bias vector")
    .Output(0, "Y", "2D output tensor");

} // namespace caffe2
