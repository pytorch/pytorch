#include "caffe2/operators/top_k.h"

#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(TopK, TopKOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(TopKGradient, TopKGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(TopK)
    .NumInputs(1)
    .NumOutputs(2)
    .TensorInferenceFunction(
        [](const OperatorDef& def, const vector<TensorShape>& in) {
          vector<TensorShape> out = {in[0], in[0]};
          ArgumentHelper helper(def);
          auto k = helper.GetSingleArgument("k", -1);
          auto dims_size = in[0].dims_size();
          out[0].set_dims(dims_size - 1, k);
          out[1].set_dims(dims_size - 1, k);
          out[1].set_data_type(TensorProto_DataType_INT32);
          return out;
        })
    .SetDoc(R"DOC(
Retrieve the top-K elements for the last dimension. Given an input tensor of
shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:
-Value tensor of shape [a_1, a_2, ..., a_n, k] which contains the values of
 the top k elements along the last dimension
-Index tensor of shape [a_1, a_2, ..., a_n, k] which contains the indices
 of the top k elements (original indices from the input tensor).

Given two equivalent values, this operator uses the indices along the last dim-
ension as a tiebreaker. That is, the element with the lower index will appear
first.
    )DOC")
    .Input(0, "X", "Tensor of shape [a_1, a_2, ..., a_n, r]")
    .Output(
        0,
        "Values",
        "Tensor of shape [a_1, a_2, ..., a_n, k] containing"
        " top K values from the input tensor")
    .Output(
        1,
        "Indices",
        "Tensor of shape [a_1, a_2, ..., a_n, k] containing"
        " the corresponding input tensor indices for the top K values.")
    .Arg("k", "Number of top elements to retrieve");

OPERATOR_SCHEMA(TopKGradient).NumInputs(3).NumOutputs(1);

class GetTopKGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "TopKGradient",
        "",
        vector<string>{GO(0), O(1), I(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(TopK, GetTopKGradient);

} // namespace
} // namespace caffe2
