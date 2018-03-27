#include "caffe2/operators/flatten_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Flatten, FlattenOp<CPUContext>);

OPERATOR_SCHEMA(Flatten)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      const int axis = helper.GetSingleArgument<int>("axis", 1);
      vector<TensorShape> out(1);
      TIndex outer = 1;
      TIndex inner = 1;
      std::size_t index = 0;
      for (auto d : in[0].dims()) {
        if (index < axis) {
          outer *= d;
        } else {
          inner *= d;
        }
        ++index;
      }
      out[0].set_data_type(in[0].data_type());
      out[0].add_dims(outer);
      out[0].add_dims(inner);
      return out;
    })
    .SetDoc(R"DOC(
Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn)
)DOC")
    .Input(0, "input", "A tensor of rank >= axis.")
    .Output(
        0,
        "output",
        "A 2D tensor with the contents of the input tensor, "
        "with input dimensions up to axis flattened to the outer dimension "
        "of the output and remaining input dimensions flattened into the inner "
        "dimension of the output.")
    .Arg(
        "axis",
        "(Default to 1) Indicate up to which input dimensions "
        "(exclusive) should be flattened to the outer dimension of the output")
    .InheritOnnxSchema("Flatten");

class GetFlattenGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ResizeLike", "", vector<string>{GO(0), I(0)}, vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(Flatten, GetFlattenGradient);

} // namespace caffe2
