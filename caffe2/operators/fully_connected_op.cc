#include "caffe2/operators/fully_connected_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(FC, FullyConnectedOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(FCGradient, FullyConnectedGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(FC)
  .NumInputs(3)
  .NumOutputs(1)
  .SetDoc(R"DOC(
Computes the result of passing an input vector X into a fully connected
layer with 2D weight matrix W and 1D bias vector b.

The layer computes Y = X * W + b, where X has size (M x K), W has size (K x N),
b has size (N), and Y has size (M x N), where M is the batch size. Even though b
is 1D, it is resized to size (M x N) implicitly and added to each vector in the
batch. These dimensions must be matched correctly, or else the operator will
throw errors.
)DOC")
  .Arg("axis", "(int32_t) default to 1; describes the axis of the inputs; "
  "defaults to one because the 0th axis most likely describes the batch_size")
  .Input(0, "X", "2D input of size (MxK) data")
  .Input(1, "W", "2D blob of size (KxN) containing fully connected weight "
  "matrix")
  .Input(2, "b", "1D blob containing bias vector")
  .Output(0, "Y", "1D output tensor");

OPERATOR_SCHEMA(FCGradient).NumInputs(3).NumOutputs(2, 3);

class GetFCGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE_EQ(def_.input_size(), 3);
    return SingleGradientDef(
        "FCGradient", "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(1), GI(2), GI(0)});
  }
};
REGISTER_GRADIENT(FC, GetFCGradient);
}  // namespace
}  // namespace caffe2
