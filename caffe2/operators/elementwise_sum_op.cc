#include "caffe2/operators/utility_ops.h"

namespace caffe2 {

namespace {
OpSchema::Cost CostInferenceForSum(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  struct OpSchema::Cost cost = PointwiseCostInference<1>(def, in);
  cost.flops *= (in.size() - 1);
  cost.params_bytes = 0;
  return cost;
}

std::vector<TensorShape> SumOpShapeInference(
    const OperatorDef& def,
    const std::vector<TensorShape>& in) {
  std::vector<TensorShape> out(1);
  out[0].set_data_type(in[0].data_type());
  ArgumentHelper helper(def);
  const bool broadcast = helper.GetSingleArgument<bool>("broadcast", false);
  if (broadcast) {
    out[0].mutable_dims()->CopyFrom(in[0].dims());
  } else {
    const std::vector<int> in0_dims(in[0].dims().begin(), in[0].dims().end());
    const std::vector<int> in1_dims(in[1].dims().begin(), in[1].dims().end());
    std::vector<int> out_dims =
        elementwise_ops_utils::ComputeBinaryBroadcastForwardDims(
            in0_dims, in1_dims);
    for (auto i = 2; i < in.size(); i++) {
      const std::vector<int> ini_dims(in[i].dims().begin(), in[0].dims().end());
      out_dims = elementwise_ops_utils::ComputeBinaryBroadcastForwardDims(
          out_dims, ini_dims);
    }
    for (const int dim : out_dims) {
      out[0].add_dims(dim);
    }
  }
  return out;
}

} // namespace

REGISTER_CPU_OPERATOR(Sum, SumOp<CPUContext>);

OPERATOR_SCHEMA(Sum)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(CostInferenceForSum)
    .TensorInferenceFunction(SumOpShapeInference)
    .InputsCanCrossDevices()
    .SetDoc(R"DOC(
Element-wise sum of each of the input tensors. The first input tensor can be used
in-place as the output tensor, in which case the sum will be done in place and
results will be accumulated the first input tensor.
If necessary the inputs arguments will be broadcasted to match the
shape of the first argument. When broadcasting is specified, the input
tensors can either be of size 1 (a scalar value), or having theirs shapes as a
contiguous subset of the first tensor's shape. The starting of the mutually
equal shape is specified by the argument "axis", and if it is not set, suffix
matching is assumed. 1-dim expansion doesn't work yet.

For example, the following tensor shapes are supported (with broadcast=1):
```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
```
Argument `broadcast=1` needs to be passed to enable broadcasting.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_sum_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Sum",
    ["A",  "B"],
    ["C"],
)

workspace.FeedBlob("A", np.array([[1,2],[3,4]]).astype(np.float32))
workspace.FeedBlob("B", np.array([[5,6],[7,8]]).astype(np.float32))
print("A:", workspace.FetchBlob("A"))
print("B:", workspace.FetchBlob("B"))
workspace.RunOperatorOnce(op)
print("C:", workspace.FetchBlob("A"))

```

**Result**

```

A: [[1. 2.]
 [3. 4.]]
B: [[5. 6.]
 [7. 8.]]
C: [[1. 2.]
 [3. 4.]]

```

</details>

<details>

<summary> <b>Example 2</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Sum",
    ["A",  "B"],
    ["A"],  // inplace
)

workspace.FeedBlob("A", np.array([[1,2,5],[8,3,4]]).astype(np.float32))
workspace.FeedBlob("B", np.array([[9,5,6],[6,7,8]]).astype(np.float32))
print("A:", workspace.FetchBlob("A"))
print("B:", workspace.FetchBlob("B"))
workspace.RunOperatorOnce(op)
print("A after Sum:", workspace.FetchBlob("A"))

```

**Result**

```

A: [[1. 2. 5.]
 [8. 3. 4.]]
B: [[9. 5. 6.]
 [6. 7. 8.]]
A after Sum: [[10.  7. 11.]
 [14. 10. 12.]]

```

</details>

)DOC")
    .Arg("broadcast", "*(type: int; default: 0)* Pass 1 to enable broadcasting")
    .Arg("axis", "*(type: int; default: -1)* Axis to concatenate on.")
    .Input(
        0,
        "A",
        "*(type: Tensor`<float>`)* First tensor to be added element-wise.")
    .Input(
        1,
        "B",
        "*(type: Tensor`<float>`)* Second tensor to be added element-wise.")
    .Output(0, "C", "*(type: Tensor`<float>`)* Sum of A and B.")
    .InheritOnnxSchema();
} // namespace caffe2
