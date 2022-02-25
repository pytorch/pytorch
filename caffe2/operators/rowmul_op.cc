#include "caffe2/operators/rowmul_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(ReduceTailSum, ReduceTailSumOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(RowMul, RowMulOp<float, CPUContext>);

OPERATOR_SCHEMA(ReduceTailSum)
    .NumInputs(1, 1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Reduce the tailing dimensions
)DOC")
    .Input(0, "mat", "The matrix")
    .Output(0, "output", "Output");

OPERATOR_SCHEMA(RowMul)
    .NumInputs(2, 2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given a matrix A and column vector w, the output is the multiplication of row i
of A and element i of w, e.g. C[i][j] = A[i][j] * w[i]. This operator should be
deprecated when the gradient operator of Mul with broadcast is implemented.
)DOC")
    .Input(0, "mat", "The matrix")
    .Input(1, "w", "The column vector")
    .Output(0, "output", "Output");

class GetRowMulGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return vector<OperatorDef>{
        CreateOperatorDef(
            "RowMul", "", vector<string>{GO(0), I(1)}, vector<string>{GI(0)}),
        CreateOperatorDef(
            "Mul",
            "",
            vector<string>{GO(0), I(0)},
            vector<string>{GI(1) + "before_aggregate"}),
        CreateOperatorDef(
            "ReduceTailSum",
            "",
            vector<string>{GI(1) + "before_aggregate"},
            vector<string>{GI(1)})};
  }
};
REGISTER_GRADIENT(RowMul, GetRowMulGradient);

} // namespace

} // namespace caffe2
