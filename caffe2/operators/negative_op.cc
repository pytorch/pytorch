#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

struct NegativeCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* /*device_context*/) {
    EigenVectorMap<T>(y, n) = -ConstEigenVectorMap<T>(x, n);
    // for (int i = 0; i < n; ++i) {
    //  y[i] = -x[i];
    //}
  }
};

REGISTER_CPU_OPERATOR(
    Negative, UnaryElementwiseOp<
        TensorTypes<float, double, int, long>, CPUContext, NegativeCPUFunctor>);

// Input: X, output: Y
OPERATOR_SCHEMA(Negative)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Computes the element-wise negative of the input.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/negative_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "Negative",
    ["X"],
    ["Y"]
)

workspace.FeedBlob("X", (np.random.rand(3,3).astype(np.float32)))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))
```

**Result**

```
X: [[0.83296907 0.61407167 0.32562155]
 [0.59304523 0.03111175 0.29365504]
 [0.09478621 0.5424558  0.73940724]]
Y: [[-0.83296907 -0.61407167 -0.32562155]
 [-0.59304523 -0.03111175 -0.29365504]
 [-0.09478621 -0.5424558  -0.73940724]]
```

</details>

)DOC")
    .Input(0, "X", "*(type: Tensor`<float>`)* 1D input tensor.")
    .Output(0, "Y", "*(type: Tensor`<float>`)* 1D output tensor.")
    .InheritOnnxSchema("Neg");

class GetNegativeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Negative", "",
        vector<string>{GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Negative, GetNegativeGradient);
}  // namespace caffe2
