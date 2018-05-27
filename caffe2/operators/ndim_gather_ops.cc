#include "caffe2/operators/ndim_gather_ops.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(NdimGather, NdimGatherOp<CPUContext>);
REGISTER_CPU_OPERATOR(NdimGatherGradient, NdimGatherGradientOp<CPUContext>);

OPERATOR_SCHEMA(NdimGather)
    .NumInputs(2)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      ArgumentHelper helper(def);

      vector<int> output_dims;
      const auto& data_dims = GetDimsVector(in[0]);
      const auto& indices_dims = GetDimsVector(in[1]);
      const int axis = helper.GetSingleArgument<int>("axis", 0);
      if (axis > 0) {
        output_dims.insert(
            output_dims.end(), data_dims.begin(), data_dims.begin() + axis);
      }
      output_dims.insert(
          output_dims.end(), indices_dims.begin(), indices_dims.end());
      if (axis < data_dims.size() - 1) {
        output_dims.insert(
            output_dims.end(), data_dims.begin() + axis + 1, data_dims.end());
      }

      out[0] = CreateTensorShape(output_dims, TensorProto::FLOAT);
      return out;
    })
    .SetDoc(R"DOC(
Given DATA tensor of rank r >= 1, INDICES tensor of rank q >= 1, and axis, gather
entries of DATA along dimension "axis" indexed by INDICES, and concatenate
them in an output tensor of rank q + (r - 1).

Example:
  DATA  = [
      [1.0, 1.2, 2.4, 4.5],
      [2.3, 3.4, 3.6, 2.3],
      [4.5, 5.7, 1.2, 4.5],
  ]
  INDICES = [
      [0, 2],
  ]
  axis = 1
  OUTPUT = [
      [1.0, 2.4],
      [2.3, 3.6],
      [4.5, 1.2],
  ]
)DOC")
    .Arg("axis", "The dimension in which we index, default to 0")
    .Input(0, "DATA", "Tensor of rank r >= 1.")
    .Input(1, "INDICES", "Tensor of int32/int64 indices, of any rank q.")
    .Output(0, "OUTPUT", "Tensor of rank q + (r - 1).");

OPERATOR_SCHEMA(NdimGatherGradient).NumInputs(3).NumOutputs(1);

class GetNdimGatherGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    using Op = NdimGatherOp<CPUContext>;
    return SingleGradientDef(
        "NdimGatherGradient",
        "",
        vector<string>{I(Op::DATA), I(Op::INDICES), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(NdimGather, GetNdimGatherGradient);

} // namespace caffe2
