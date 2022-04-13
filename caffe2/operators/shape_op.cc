#include "caffe2/operators/shape_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Shape, ShapeOp<CPUContext>);

OPERATOR_SCHEMA(Shape)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg(
        "axes",
        "*(type: int[])* Array of interested axes."
        "If given, this operator only returns the dimensions of the given axes."
        "Otherwise, the operator returns the dimensions of all axes.")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper args(def);
      const vector<int>& axes = args.GetRepeatedArgument<int>("axes");
      vector<TensorShape> out(1);
      if (axes.empty()) {
        out[0].add_dims(in[0].dims().size());
      } else {
        out[0].add_dims(axes.size());
      }
      out[0].set_data_type(TensorProto::INT64);
      return out;
    })
    .SetDoc(R"DOC(
Produce a 1D int64 tensor with the shape of the input tensor.
If called with an optional argument `axes`, the result will only
contain the dimensions of specified axes.

Github Link:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/shape_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Shape",
    ["X"],
    ["shape"],
)

workspace.FeedBlob("X", (np.random.randint(10, size=(2,3))))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("shape:", workspace.FetchBlob("shape"))

```

**Result**

```

X:
[[3 2 5]
 [5 7 3]]
shape: [2 3]

```

</details>

      )DOC")
    .Input(0,"X", "*(type: Tensor)* Input tensor.")
    .Output(0,"shape", "*(type: Tensor)* Output tensor containing shape of input tensor.");

SHOULD_NOT_DO_GRADIENT(Shape);

} // namespace caffe2
