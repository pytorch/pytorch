#include "caffe2/operators/slice_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Slice, SliceOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_GRADIENT_OPERATOR(SliceGradient, SliceGradientOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Slice)
    .NumInputs(1, 3)
    .NumOutputs(1)
    .DisallowInputFillers() // the filler cannot be enabled without output dims
    .SetDoc(R"DOC(
Produces a slice of the input tensor.

- Currently, only slicing in a single dimension is supported.

- Start and end indices are either passed as two 1D input tensors or using the `starts` and `ends` arguments.

- If a negative value is passed for any of the start or end indices, it represents |value| - 1 elements before the end of that dimension. End indices are non-inclusive unless negative (end index -1 means up to and including the last element).

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/slice_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Slice",
    ["X"],
    ["Y"],
    starts=(0,1),
    ends=(-1,3)
)

workspace.FeedBlob("X", np.array([[1,2,3,4],[5,6,7,8]]))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))

```

**Result**

```

X:
[[1 2 3 4]
 [5 6 7 8]]
Y:
[[2 3]
 [6 7]]

```

</details>

)DOC")
    .Input(0, "X", "(*Tensor*): tensor to extract slices from")
    .Input(
        1,
        "starts",
        "(*Tensor`<int>`*): 1D tensor of start-indices for each dimension of data (dimensions following the sliced one might be omitted)")
    .Input(
        2,
        "ends",
        "(*Tensor`<int>`*): 1D tensor of end-indices for each dimension of data (dimensions following the sliced one might be omitted)")
    .Arg("starts", "(*Tuple(int)*): list of starting indices")
    .Arg("ends", "(*Tuple(int)*): list of ending indices")
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
        // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
        if (i >= starts.size()) {
          dst_sizes[i] = data.dims(i);
          continue;
        }
        if (data.dims(i) > 0) {
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
    .Output(0, "Y", "(*Tensor*): sliced output tensor")
    .InheritOnnxSchema();

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
GRADIENT_OPERATOR_SCHEMA(SliceGradient)
    .TensorInferenceFunction([](const OperatorDef& /*def*/,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out.at(0) = in.at(0);
      return out;
    });

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
} // namespace
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(Slice, GetSliceGradient);
} // namespace caffe2
