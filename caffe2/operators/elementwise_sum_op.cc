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

class GetSumGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    auto inputs = vector<string>{GO(0)};
    auto outputs = vector<string>();

    for (auto i = 0; i < def_.input_size(); i++) {
      inputs.push_back(I(i));
      outputs.push_back(GI(i));
    }

    return SingleGradientDef("SumGradient", "", inputs, outputs);
  }
};

} // namespace

REGISTER_CPU_OPERATOR(Sum, SumOp<CPUContext>);
REGISTER_CPU_OPERATOR(SumGradient, SumGradientOp<CPUContext>);
REGISTER_GRADIENT(Sum, GetSumGradient);

OPERATOR_SCHEMA(Sum)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(CostInferenceForSum)
    .InputsCanCrossDevices()
    .IdenticalTypeAndShapeOfInput(0)
    .SetDoc(R"DOC(
Element-wise sum of each of the input tensors. The first input tensor can be used
in-place as the output tensor, in which case the sum will be done in place and
results will be accumulated the first input tensor. All inputs and outputs must
have the same data type.

This operation supports numpy-style broadcasting when shape between inputs is
different. Essentially the shape between two tensors would be compared starting
from trailing dimensions, moving itsway forward. Two dimensions are compatible
when they are either equal or one of them is 1.

When summation is done in in-place mode, the shape of first input must be the
same as final output shape after broadcasting.

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

<details>

<summary> <b>Example 3</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Sum",
    ["A",  "B"],
    ["C"],
)

workspace.FeedBlob("A", np.array([2, 2, 2]).astype(np.float32))
workspace.FeedBlob("B", np.array([[9,5,6],[6,7,8]]).astype(np.float32))
print("A:", workspace.FetchBlob("A"))
print("B:", workspace.FetchBlob("B"))
workspace.RunOperatorOnce(op)
print("C after Sum:", workspace.FetchBlob("C"))

```

**Result**

```

A: [2. 2. 2.]
B: [[9. 5. 6.]
 [6. 7. 8.]]
C after Sum: [[11.  7. 8.]
 [8. 9. 10.]]

```

</details>

)DOC")
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

OPERATOR_SCHEMA(SumGradient)
    .NumInputs(2, INT_MAX)
    .NumInputsOutputs([](int n_input, int n_output) {
      return n_input - 1 == n_output;
    })
    .AllowInplace({{0, 0}});
} // namespace caffe2
