#include "caffe2/operators/fully_connected_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(FC, FullyConnectedOp<CPUContext>);
REGISTER_CPU_OPERATOR(FCGradient, FullyConnectedGradientOp<CPUContext>);

OPERATOR_SCHEMA(FC)
    .NumInputs(3)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      ArgumentHelper helper(def);

      auto axis = helper.GetSingleArgument<int32_t>("axis", 1);
      const auto canonical_axis =
          canonical_axis_index_(axis, in[0].dims().size());
      const int M = size_to_dim_(canonical_axis, GetDimsVector(in[0]));
      const int N = in[1].dims(0);
      out[0] = CreateTensorShape(vector<int>{M, N}, TensorProto::FLOAT);
      return out;
    })
    .SetDoc(R"DOC(
    Computes the result of passing an input vector X into a fully
    connected layer with 2D weight matrix W and 1D bias vector b. That is,
    the layer computes Y = X * W^T + b, where X has size (M x K),
    W has size (N x K), b has size (N), and Y has size (M x N),
    where M is often the batch size.


    NOTE: X does not need to explicitly be a 2D vector; rather, it will be
    coerced into one. For an arbitrary n-dimensional tensor
    X \in [a_0, a_1, ...,a_{k-1}, a_k, ..., a_{n-1}] where a_i \in N+ and k is
    the axis provided, then X will be coerced into a 2-dimensional tensor with
    dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
    case where axis=1, this means the X tensor will be coerced into a 2D tensor
    of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
    In this situation, we must have a_0 = M and a_1 * ... * a_{n-1} = K.
    Lastly, even though b is a 1D vector of size N, it is copied/resized to
    be size (M x N) implicitly and added to each vector in the batch.
    Each of these dimensions must be matched correctly, or else the operator
    will throw errors.

)DOC")
    .Arg(
        "axis",
        "(int32_t) default to 1; describes the axis of the inputs; "
        "defaults to one because the 0th axis most likely describes "
        "the batch_size")
    .Input(
        0,
        "X",
        "input tensor that's coerced into a 2D matrix of size (MxK) "
        "as described above")
    .Input(
        1,
        "W",
        "2D blob of size (KxN) containing fully connected weight "
        "matrix")
    .Input(2, "b", "1D blob containing bias vector")
    .Output(0, "Y", "2D output tensor");

OPERATOR_SCHEMA(FCGradient).NumInputs(3).NumOutputs(2, 3);

class GetFCGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE_EQ(def_.input_size(), 3);
    return SingleGradientDef(
        "FCGradient",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(1), GI(2), GI(0)});
  }
};
REGISTER_GRADIENT(FC, GetFCGradient);
}  // namespace caffe2
