#include "caffe2/operators/tt_linear_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(TT, TTLinearOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(TTLinearGradient, TTLinearGradientOp<float, CPUContext>);

// The TT-layer serves as a low-rank decomposition of a fully connected layer.
// The inputs are the same as to an FC layer, but the number of the parameters
// are greatly reduced.
OPERATOR_SCHEMA(TT)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
The TT-layer serves as a low-rank decomposition of a fully connected layer. The
inputs are the same as to a fully connected layer, but the number of parameters
are greatly reduced and forward computation time can be drastically reduced
especially for layers with large weight matrices. The multiplication is computed
as a product of the input vector with each of the cores that make up the TT
layer. Given the input sizes (inp_sizes), output sizes(out_sizes), and the ranks
of each of the cores (tt_ranks), the ith core will have size:

    inp_sizes[i] * tt_ranks[i] * tt_ranks[i + 1] * out_sizes[i].

The complexity of the computation is dictated by the sizes of inp_sizes,
out_sizes, and tt_ranks, where there is the trade off between accuracy of the
low-rank decomposition and the speed of the computation.
)DOC")
    .Arg(
        "inp_sizes",
        "(int[]) Input sizes of cores. Indicates the input size of "
        "the individual cores; the size of the input vector X must match the "
        "product of the inp_sizes array.")
    .Arg(
        "out_sizes",
        "(int[]) Output sizes of cores. Indicates the output size "
        "of the individual cores; the size of the output vector Y must match "
        "the product of the out_sizes array.")
    .Arg(
        "tt_ranks",
        "(int[]) Ranks of cores. Indicates the ranks of the "
        "individual cores; lower rank means larger compression, faster "
        "computation but reduce accuracy.")
    .Input(
        0,
        "X",
        "Input tensor from previous layer with size (M x K), where "
        "M is the batch size and K is the input size.")
    .Input(1, "b", "1D blob containing the bias vector")
    .Input(
        2,
        "cores",
        "1D blob containing each individual cores with sizes "
        "specified above.")
    .Output(
        0,
        "Y",
        "Output tensor from previous layer with size (M x N), "
        "where M is the batch size and N is the output size.");

OPERATOR_SCHEMA(TTLinearGradient);

GRADIENT_NOT_IMPLEMENTED_YET(TT);
} // namespace
} // namespace caffe2
