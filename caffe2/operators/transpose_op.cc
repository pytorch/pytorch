#include "caffe2/operators/transpose_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Transpose, TransposeOp<CPUContext>);

OPERATOR_SCHEMA(Transpose)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      vector<int> axes = helper.GetRepeatedArgument<int>("axes");
      vector<TensorShape> out(1);
      out[0].set_data_type(in[0].data_type());

      if (axes.empty()) {
        for (auto axis = in [0].dims().rbegin(); axis != in[0].dims().rend();
             ++axis) {
          out[0].add_dims(*axis);
        }
      } else {
        auto tensor_size = in[0].dims().size();
        auto valid_axes =
            std::all_of(axes.begin(), axes.end(), [&tensor_size](int& axis) {
              return axis >= 0 && axis < tensor_size;
            });

        CAFFE_ENFORCE(valid_axes, "Axes argument passed in had invalid values");
        CAFFE_ENFORCE(
            // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
            axes.size() == tensor_size,
            "Axes argument passed in had the incorrect size");

        // NOLINTNEXTLINE(modernize-loop-convert)
        for (auto axis = axes.begin(); axis != axes.end(); ++axis) {
          out[0].add_dims(in[0].dims().Get(*axis));
        }
      }

      return out;
    })
    .SetDoc(R"DOC(
Transpose the input tensor by permuting the axes of the input according
to the `axes` argument. Similar to numpy's
[transpose](https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html)
function.

For example, when axes=(1, 0, 2), given an input tensor of shape
(1, 2, 3), the output shape will be (2, 1, 3).

Github Links:

- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/transpose_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "Transpose",
    ["X"],
    ["Y"],
    axes=(0,3,1,2)
)

x = np.random.rand(1,32,32,3)
workspace.FeedBlob("X", x)
print("X.shape (NHWC order):", workspace.FetchBlob("X").shape)
workspace.RunOperatorOnce(op)
print("Y.shape (NCHW order):", workspace.FetchBlob("Y").shape)
```

**Result**

```
X.shape (NHWC order): (1, 32, 32, 3)
Y.shape (NCHW order): (1, 3, 32, 32)
```

</details>

)DOC")
    .Arg(
        "axes",
        "*(type: Tuple(int))* Order to permute axes of input tensor. Reverses "
        "the dimensions by default.")
    .Input(0, "X", "*(type: Tensor)* Input tensor.")
    .Output(0, "Y", "*(type: Tensor)* Transposed output.")
    .InheritOnnxSchema();

class GetTransposeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  // We will create our own arguments.
  bool CopyArguments() const override {
    return false;
  }
  vector<OperatorDef> GetGradientDefs() override {
    auto ops = SingleGradientDef(
        "Transpose", "", vector<string>{GO(0)}, vector<string>{GI(0)});
    ops[0].mutable_arg()->CopyFrom(Def().arg());
    if (ArgumentHelper::HasArgument(Def(), "axes")) {
      // If axes is specified, we will need to figure out the inverse index.
      const Argument& old_axes = GetArgument(Def(), "axes");
      const int axes_size = old_axes.ints_size();
      Argument* new_arg = GetMutableArgument("axes", false, &ops[0]);
      for (int i = 0; i < axes_size; ++i) {
        new_arg->set_ints(old_axes.ints(i), i);
      }
    }
    return ops;
  }
};

REGISTER_GRADIENT(Transpose, GetTransposeGradient);

} // namespace caffe2
