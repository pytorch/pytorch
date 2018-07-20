#include "caffe2/operators/lengths_matmul_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(LengthsMatMul, LengthsMatMulOp<float, CPUContext>);

// Input: X, X_LEN, Y, Y_LEN
OPERATOR_SCHEMA(LengthsMatMul)
    .NumInputs(4)
    .NumOutputs(2)
    .SetDoc(R"DOC(
Apply matrix multiplication from the matrices on a per segment basis specified by the lengths tensor. )DOC")
    .Input(0, "X", "Tensor of rank 2. Second dimension must be same as Y's")
    .Input(1, "X_LENGTHS", "Tensor of int32 lengths of rank 1")
    .Input(2, "Y", "Tensor of rank 2. Second dimensino must be same as X's")
    .Input(3, "Y_LENGTHS", "Tensor of int32 lengths of rank 1")
    .Output(
        0,
        "MatMul",
        "Per batch matrix multiplication flattened to tensor of rank 1")
    .Output(1, "MatMulLengths", "Per batch matrix multiplication lengths")
    .Arg(
        "trans_a",
        "whether to transpose first matrix, on default is not transposed")
    .Arg(
        "trans_b",
        "Whether to transpose second matrix, on default is transposed");

namespace {
class GetLengthsMatMulGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE_EQ(def_.input_size(), 4);

    auto no_trans_arg = vector<Argument>{MakeArgument<int>("trans_a", 0),
                                         MakeArgument<int>("trans_b", 0)};
    auto trans_a_arg = vector<Argument>{MakeArgument<int>("trans_a", 1),
                                        MakeArgument<int>("trans_b", 0)};
    // AB'
    // dA = GB, dB = G'A
    return vector<OperatorDef>{CreateOperatorDef(
                                   "LengthsMatMul",
                                   "",
                                   vector<string>{GO(0), O(1), I(2), I(3)},
                                   vector<string>{GI(0), ""},
                                   no_trans_arg),
                               CreateOperatorDef(
                                   "LengthsMatMul",
                                   "",
                                   vector<string>{GO(0), O(1), I(0), I(1)},
                                   vector<string>{GI(2), ""},
                                   trans_a_arg)};
  }
  bool CopyArguments() const override {
    return false;
  }
};

REGISTER_GRADIENT(LengthsMatMul, GetLengthsMatMulGradient);
} // namespace
} // namespace caffe2
