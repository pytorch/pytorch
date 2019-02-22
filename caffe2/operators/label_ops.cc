#include "caffe2/operators/label_ops.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SparseLabelSplit, SparseLabelSplitOp<float, CPUContext>);

OPERATOR_SCHEMA(SparseLabelSplit)
    .NumInputs(3)
    .NumOutputs(2, INT_MAX)
    .ValueKeyLengthInputFillers(2, 1, 0)
    .SetDoc(R"DOC(
Suppose the maximum of label index is r. This operator has 2r + 1 1-D outputs.
0<= i < r, output[i] contains the label_values of labels with label_index=i
(original order is kept).
r<= i < 2r, output[i] contains the corresponding example_ids for output[i-r].
output[2r] (optional) keeps an offset map that is useful for the gradient computation.
Specifically, this map keeps track of the ordering of examples in the expert inputs.
)DOC")
    .Arg("num_labels", "Optional; Number of label tasks")
    .Input(
        0,
        "length",
        "A Nx1 int32 tensor. Sum of its values needs to be"
        "the same as the size of label_index and label_value")
    .Input(1, "label_index.", "A Mx1 int64 tensor.")
    .Input(2, "label_value.", "A Mx1 float tensor.")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      auto num_labels = helper.GetSingleArgument<int>("num_labels", -1);
      if (num_labels == -1) {
        num_labels = def.output_size() / 2;
      }
      vector<TensorShape> out(2 * num_labels);
      for (int i = 0; i < 2 * num_labels; i++) {
        for (auto d : in[0].dims()) {
          out[i].add_dims(d);
        }
        if (i < num_labels) {
          out[i].set_data_type(in[2].data_type());
        } else {
          out[i].set_data_type(in[0].data_type());
        }
      }
      if (def.output_size() > 2 * num_labels) {
        out.push_back(in[1]);
        out.back().set_data_type(TensorProto::INT32);
      }
      return out;
    });

REGISTER_CPU_OPERATOR(
    SparseLabelSplitGradient,
    SparseLabelSplitGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(SparseLabelSplitGradient)
    .NumInputs(2, INT_MAX)
    .NumOutputs(1)
    .DisallowInputFillers();

class GetSparseLabelSplitGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    const auto& arg = GetArgument(def_, "num_labels");
    CAFFE_ENFORCE(arg.has_i());
    int num_labels = arg.i();

    vector<std::string> grad_inputs{I(0), I(1)};
    for (int i = 0; i < num_labels; i++) {
      grad_inputs.push_back(GO(i));
    }
    if (def_.output_size() > 2 * num_labels) {
      grad_inputs.push_back(O(2 * num_labels));
    }
    return SingleGradientDef(
        "SparseLabelSplitGradient", "", grad_inputs, vector<std::string>{GI(2)});
  }
};
REGISTER_GRADIENT(SparseLabelSplit, GetSparseLabelSplitGradient);

} // namespace caffe2
