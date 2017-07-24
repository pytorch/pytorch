#include "caffe2/operators/matmul_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(MatMul, MatMulOp<float, CPUContext>);

OPERATOR_SCHEMA(MatMul)
    .NumInputs(2)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /*def*/,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out[0].set_data_type(in[0].data_type());

      int M = in[0].dims().Get(0);
      int N = in[1].dims().Get(1);

      out[0].add_dims(M);
      out[0].add_dims(N);

      return out;
    })
    .SetDoc(R"DOC(
Matrix multiplication Y = A * B, where A has size (M x K), B has size (K x N),
and Y will have a size (M x N).
)DOC")
    .Input(0, "A", "2D matrix of size (M x K)")
    .Input(1, "B", "2D matrix of size (K x N)")
    .Output(0, "Y", "2D matrix of size (M x N)")
    .Arg("trans_a", "Pass 1 to transpose A before multiplication")
    .Arg("trans_b", "Pass 1 to transpose B before multiplication");

class GetMatMulGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE_EQ(def_.input_size(), 2);

    bool trans_a = 0;
    bool trans_b = 0;

    if (ArgumentHelper::HasArgument(Def(), "trans_a")) {
      trans_a = GetArgument(Def(), "trans_a").i();
    }
    if (ArgumentHelper::HasArgument(Def(), "trans_b")) {
      trans_b = GetArgument(Def(), "trans_b").i();
    }

    const auto no_trans_arg = vector<Argument>();
    const auto trans_a_arg = vector<Argument>{
        MakeArgument<int>("trans_a", 1)};
    const auto trans_b_arg = vector<Argument>{
        MakeArgument<int>("trans_b", 1)};
    const auto trans_both_arg = vector<Argument>{
        MakeArgument<int>("trans_a", 1),
        MakeArgument<int>("trans_b", 1)};

    if (trans_a) {
      if (trans_b) {
        // A'B':
        // dA = B'G', dB = G'A'
        return vector<OperatorDef>{
            CreateOperatorDef(
                "MatMul",
                "",
                vector<string>{I(1), GO(0)},
                vector<string>{GI(0)},
                trans_both_arg),
            CreateOperatorDef(
                "MatMul",
                "",
                vector<string>{GO(0), I(0)},
                vector<string>{GI(1)},
                trans_both_arg)};
      } else {
        // A'B:
        // dA = BG', dB = AG
        return vector<OperatorDef>{
            CreateOperatorDef(
                "MatMul",
                "",
                vector<string>{I(1), GO(0)},
                vector<string>{GI(0)},
                trans_b_arg),
            CreateOperatorDef(
                "MatMul",
                "",
                vector<string>{I(0), GO(0)},
                vector<string>{GI(1)},
                no_trans_arg)};
      }
    } else {
      if (trans_b) {
        // AB':
        // dA = GB, dB = G'A
        return vector<OperatorDef>{
            CreateOperatorDef(
                "MatMul",
                "",
                vector<string>{GO(0), I(1)},
                vector<string>{GI(0)},
                no_trans_arg),
            CreateOperatorDef(
                "MatMul",
                "",
                vector<string>{GO(0), I(0)},
                vector<string>{GI(1)},
                trans_a_arg)};
      } else {
        // AB:
        // dA = GB', dB = A'G
        return vector<OperatorDef>{
            CreateOperatorDef(
                "MatMul",
                "",
                vector<string>{GO(0), I(1)},
                vector<string>{GI(0)},
                trans_b_arg),
            CreateOperatorDef(
                "MatMul",
                "",
                vector<string>{I(0), GO(0)},
                vector<string>{GI(1)},
                trans_a_arg)};
      }
    }
  }

  bool CopyArguments() const override {
    return false;
  }
};

REGISTER_GRADIENT(MatMul, GetMatMulGradient);

} // namespace caffe2
