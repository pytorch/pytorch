#include "caffe2/operators/slice_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Slice, SliceOp<int, CPUContext>);
REGISTER_CPU_OPERATOR(SliceGradient, SliceGradientOp<int, CPUContext>);

OPERATOR_SCHEMA(Slice)
    .NumInputs(1, 3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Produces a slice of the input tensor. Currently, only slicing in a single
dimension is supported.
Slices are passed as 2 1D vectors or as two keyword argument lists with starting
and end indices for each dimension of the input `data` tensor. If a negative
value is passed for any of the start or end indices, it represents the number of
elements before the end of that dimension. End indices are non-inclusive unless
negative (end index -1 means up to and including the last element).


Example:

  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 3]

  result = [
      [2, 3],
      [6, 7],
  ]
)DOC")
    .Input(0, "data", "Tensor of data to extract slices from.")
    .Input(1, "starts", "1D tensor: start-indices for each dimension of data.")
    .Input(2, "ends", "1D tensor: end-indices for each dimension of data.")
    .Arg("starts", "List of starting indices")
    .Arg("ends", "List of ending indices")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      if (in.size() > 1) {
        // Cannot compute shape inference when the splits are defined
        // in data.
        return vector<TensorShape>();
      }
      auto const& data = in[0];

      ArgumentHelper helper(def);
      auto starts = helper.GetRepeatedArgument<int>("starts", vector<int>());
      auto ends = helper.GetRepeatedArgument<int>("ends", vector<int>());
      vector<int> dst_sizes(data.dims_size());

      for (int i = 0; i < data.dims_size(); ++i) {
        if (i >= starts.size()) {
          continue;
        }
        if (data.dims_size() > 0) {
          auto start = starts[i];
          auto end = ends[i];
          if (start < 0) {
            start = data.dims(i) + 1 + start;
          }
          if (end < 0) {
            end = data.dims(i) + 1 + end;
          }
          dst_sizes[i] = end - start;
        } else {
          dst_sizes[i] = 0;
        }
      }
      return vector<TensorShape>{
          CreateTensorShape(dst_sizes, data.data_type())};
    })
    .Output(0, "output", "Sliced data tensor.")
    .InheritOnnxSchema("Slice");

OPERATOR_SCHEMA(SliceGradient);

namespace {
struct GetSliceGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    if (def_.input_size() > 1) {
      return vector<OperatorDef>{CreateOperatorDef(
          "SliceGradient",
          "",
          std::vector<string>{I(0), I(1), I(2), GO(0)},
          std::vector<string>{GI(0)})};
    } else {
      return vector<OperatorDef>{CreateOperatorDef(
          "SliceGradient",
          "",
          std::vector<string>{I(0), GO(0)},
          std::vector<string>{GI(0)})};
    }
  }
};
}
REGISTER_GRADIENT(Slice, GetSliceGradient);
} // namespace caffe2
