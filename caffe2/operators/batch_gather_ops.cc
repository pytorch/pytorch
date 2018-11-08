#include "caffe2/operators/batch_gather_ops.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(BatchGather, BatchGatherOp<CPUContext>);
REGISTER_CPU_OPERATOR(BatchGatherGradient, BatchGatherGradientOp<CPUContext>);

OPERATOR_SCHEMA(BatchGather)
    .NumInputs(2)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      ArgumentHelper helper(def);

      vector<int> output_dims;
      const auto& data_dims = GetDimsVector(in[0]);
      const auto& indices_dims = GetDimsVector(in[1]);
      output_dims.push_back(data_dims[0]);
      output_dims.insert(
          output_dims.end(), indices_dims.begin(), indices_dims.end());
      output_dims.insert(
          output_dims.end(), data_dims.begin() + 2, data_dims.end());

      out[0] = CreateTensorShape(output_dims, TensorProto::FLOAT);
      return out;
    })
    .SetDoc(R"DOC(
Batch gather operation, first dimension in DATA is the batch size.
Given DATA tensor of rank r >= 2, and INDICES tensor of rank q >= 1, gather
entries of the outer-most dimension of DATA indexed by INDICES, and concatenate
them in an output tensor of rank (q - 1) + (r - 1).

Example:
  DATA  = [
      [1.0, 1.2, 2.4, 4.5],
      [2.3, 3.4, 3.6, 2.3],
      [4.5, 5.7, 1.2, 4.5],
  ]
  INDICES = [
      [0, 2],
  ]
  OUTPUT = [
      [1.0, 2.4],
      [2.3, 3.6],
      [4.5, 1.2],
  ]
)DOC")
    .Input(0, "DATA", "Tensor of rank r >= 2.")
    .Input(1, "INDICES", "Tensor of int32/int64 indices, of any rank q.")
    .Output(0, "OUTPUT", "Tensor of rank (q - 1) + (r - 1).");

OPERATOR_SCHEMA(BatchGatherGradient).NumInputs(3).NumOutputs(1);

class GetBatchGatherGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    using Op = BatchGatherOp<CPUContext>;
    return SingleGradientDef(
        "BatchGatherGradient",
        "",
        vector<string>{I(Op::DATA), I(Op::INDICES), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(BatchGather, GetBatchGatherGradient);

} // namespace caffe2
