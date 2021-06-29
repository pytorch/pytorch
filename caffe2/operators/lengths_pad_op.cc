#include "caffe2/operators/lengths_pad_op.h"

namespace caffe2 {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(LengthsPad, LengthsPadOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(LengthsPad)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given DATA tensor of rank r >= 1, and LENGTHS tensor of rank 1, pad each
segment in DATA with `value`, so that each segment's length is `target_length`.
If will throw, if there is segment of length larger than `target_length`.

Example:
  DATA  = [
      [2.3, 3.4],
      [4.5, 5.7],
      [6.8, 7.9],
  ]
  LENGTHS = [0, 1, 1, 1]
  and target_length = 2, padding value = -1.0
  OUTPUT = [
    [-1.0, -1.0],
    [-1.0, -1.0],
    [2.3, 3.4],
    [-1.0, -1.0],
    [4.5, 5.7],
    [-1.0, -1.0],
    [6.8, 7.9],
    [-1.0, -1.0],
  ]
)DOC")
    .Input(
        0,
        "DATA",
        "Tensor of rank r >= 1. First dimension must be equal to the size of "
        "lengths")
    .Input(1, "LENGTHS", "Tensor of int32 lengths of rank 1")
    .Output(0, "OUTPUT", "Padded DATA tensor")
    .Arg("padding_value", "The value to pad the data")
    .Arg("target_length", "The target length of each segment")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      ArgumentHelper helper(def);
      int target_length = helper.GetSingleArgument<int>("target_length", -1);
      CAFFE_ENFORCE_GE(target_length, 1);
      vector<int> output_dims;
      const auto& data_dims = GetDimsVector(in[0]);
      const auto& lengths_dims = GetDimsVector(in[1]);
      output_dims.push_back(lengths_dims[0] * target_length);
      output_dims.insert(
          output_dims.end(), data_dims.begin() + 1, data_dims.end());

      out[0] = CreateTensorShape(output_dims, in[0].data_type());
      return out;
    });

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(LengthsPad);
} // namespace caffe2
