#include "caffe2/operators/expand_squeeze_dims_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(ExpandDims, ExpandDimsOp<CPUContext>);
REGISTER_CPU_OPERATOR(Squeeze, SqueezeOp<CPUContext>);

OPERATOR_SCHEMA(ExpandDims)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      auto dims = helper.template GetRepeatedArgument<int>("dims");
      auto originalSize = dims.size();
      CAFFE_ENFORCE(originalSize > 0, "Parameter `dims` must be provided.");

      std::sort(dims.begin(), dims.end());
      dims.erase(std::unique(dims.begin(), dims.end()), dims.end());
      if (dims.size() < originalSize) {
        LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
      }

      CAFFE_ENFORCE(dims.front() >= 0, "Dimension ids must be non-negative.");
      CAFFE_ENFORCE_GE(
          in[0].dims_size() + dims.size(),
          dims.back() + 1,
          "Input needs at least ",
          (1 + dims.back() - dims.size()),
          " dimensions given `dims`.");

      vector<TensorShape> out(1);

      int cur_pos = 0;
      int idx = 0;
      for (const auto new_dim : dims) {
        for (int i = cur_pos; i < new_dim; i++) {
          out[0].add_dims(in[0].dims(idx++));
        }
        out[0].add_dims(1);
        cur_pos = new_dim + 1;
      }
      for (; idx < in[0].dims_size(); idx++) {
        out[0].add_dims(in[0].dims(idx));
      }
      out[0].set_data_type(in[0].data_type());
      return out;
    })
    .SetDoc(R"DOC(
Insert single-dimensional entries to the shape of a tensor.
Takes one required argument `dims`, a list of dimensions that will be inserted.
Dimension indices in `dims` are as seen in the output tensor. For example:

  Given a tensor such that tensor.Shape() = [3, 4, 5], then
  ExpandDims(tensor, dims=[0, 4]).Shape() == [1, 3, 4, 5, 1])

If the same blob is provided in input and output, the operation is copy-free.
)DOC")
    .Input(0, "data", "Original tensor")
    .Output(0, "expanded", "Reshaped tensor with same data as input.");

OPERATOR_SCHEMA(Squeeze)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Remove single-dimensional entries from the shape of a tensor.
Takes a parameter `dims` with a list of dimension to squeeze.
If the same blob is provided in input and output, the operation is copy-free.
This is the exact inverse operation of ExpandDims given the same `dims` arg.
)DOC")
    .Input(0, "data", "Tensors with at least max(dims) dimensions.")
    .Output(0, "squeezed", "Reshaped tensor with same data as input.")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      auto dims = helper.template GetRepeatedArgument<int>("dims");
      auto originalSize = dims.size();
      std::sort(dims.begin(), dims.end());
      dims.erase(std::unique(dims.begin(), dims.end()), dims.end());
      if (dims.size() < originalSize) {
        LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
      }
      CAFFE_ENFORCE(dims.front() >= 0, "Dimension ids must be non-negative.");

      vector<TensorShape> out(1);
      std::vector<int> newDims =
          SqueezeOp<CPUContext>::ComputeDims(GetDimsVector(in[0]), dims);
      out[0] = CreateTensorShape(newDims, in[0].data_type());
      return out;
    })
    .InheritOnnxSchema("Squeeze");

class GetSqueezeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ExpandDims", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Squeeze, GetSqueezeGradient);

class GetExpandDimsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Squeeze", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(ExpandDims, GetExpandDimsGradient);
}
