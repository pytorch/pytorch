#include "copy_rows_to_tensor_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(CopyRowsToTensor, CopyRowsToTensorOp<CPUContext>);
REGISTER_CPU_GRADIENT_OPERATOR(
    CopyRowsToTensorGradient,
    CopyRowsToTensorGradientOp<CPUContext>);

OPERATOR_SCHEMA(CopyRowsToTensor)
    .NumInputs(3)
    .NumOutputs(1)
    .EnforceInplace({{0, 0}})
    .SetDoc(R"DOC(
      This operator takes in a 2d tensor, a list of indices, and a 1d tensor
      with the same width of the 2d tensor. It will replace the rows in 2d
      tensor specified in indices with the 2d tensor. The operator does an
      in-place change to the input tensor.
      Example:
        INPUT_TENSOR = [[1, 2], [3, 4], [5, 6]]
        INDICES = [1]
        ROW = [9, 0]
        OUTPUT_TENSOR = [[1, 2], [9, 0], [5, 6]]
      )DOC")
    .Input(0, "input_tensor", "Input tensor needs to be modified.")
    .Input(1, "indices", "Indices of rows need to be copied")
    .Input(2, "row", "1-d tensor that is going to replace the rows")
    .Output(0, "output_tensor", "updated tensor")
    .TensorInferenceFunction([](const OperatorDef& /*def*/,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out[0] = in[0];
      return out;
    });

GRADIENT_OPERATOR_SCHEMA(CopyRowsToTensorGradient)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}});

class GetCopyRowsToTensorGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    if (g_output_[0].IsDense()) {
      return SingleGradientDef(
          "CopyRowsToTensorGradient",
          "",
          vector<string>{GO(0)},
          vector<string>{GI(0)});
    } else {
      return vector<OperatorDef>{CreateOperatorDef(
                                     "CopyRowsToTensorGradient",
                                     "",
                                     std::vector<string>{GO_I(0)},
                                     std::vector<string>{GI_I(0)}),
                                 CreateOperatorDef(
                                     "CopyRowsToTensorGradient",
                                     "",
                                     std::vector<string>{GO_V(0)},
                                     std::vector<string>{GI_V(0)})};
    }
  }
};

REGISTER_GRADIENT(CopyRowsToTensor, GetCopyRowsToTensorGradient);

} // namespace
} // namespace caffe2
