#include "caffe2/experiments/operators/fully_connected_op_decomposition.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(FC_Decomp, FullyConnectedOpDecomp<float, CPUContext>);
REGISTER_CPU_OPERATOR(FCGradient_Decomp,
                      FullyConnectedDecompGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(FC_Decomp).NumInputs(4).NumOutputs(1);
OPERATOR_SCHEMA(FCGradient_Decomp).NumInputs(4).NumOutputs(3, 4);

class GetFCDecompGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE_EQ(def_.input_size(), 4);
    // TODO(wyiming): Check whether it is right? Let's move fast first.
    return SingleGradientDef(
        "FCGradient_Decomp", "",
        vector<string>{I(0), I(1), I(2), GO(0)},
        vector<string>{GI(1), GI(2), GI(3), GI(0)});
  }
};
REGISTER_GRADIENT(FC_Decomp, GetFCDecompGradient);
}  // namespace caffe2
