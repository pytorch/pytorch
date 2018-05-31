#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

struct SigmoidCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* /*device_context*/) {
    ConstEigenVectorArrayMap<T> xM(x, n);
    EigenVectorArrayMap<T>(y, n) = 1. / (1. + (-xM).exp());
  }
};

struct SigmoidGradientCPUFunctor {
  template <typename T>
  inline void Run(
      const int n,
      const T* y,
      const T* dy,
      T* dx,
      CPUContext* /*device_context*/) {
    ConstEigenVectorArrayMap<T> yM(y, n), dyM(dy, n);
    EigenVectorArrayMap<T>(dx, n) = dyM * yM * (1. - yM);
  }
};

REGISTER_CPU_OPERATOR(
    Sigmoid, UnaryElementwiseOp<
        TensorTypes<float>, CPUContext, SigmoidCPUFunctor>);
REGISTER_CPU_OPERATOR(
    SigmoidGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        WithoutBroadcast<SigmoidGradientCPUFunctor>>);

// Input: X, output: Y
OPERATOR_SCHEMA(Sigmoid)
  .NumInputs(1)
  .NumOutputs(1)
  .AllowInplace({{0, 0}})
  .IdenticalTypeAndShape()
  .SetDoc(R"DOC(
Apply the Sigmoid function element-wise to the input tensor. This is often used
as a non-linear activation function in a neural network. The sigmoid function is
defined as:

$$Sigmoid(x) = \frac{1}{1+\exp(-x)}$$

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sigmoid_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Sigmoid",
    ["X"],
    ["Y"]
)

workspace.FeedBlob("X", np.random.randn(5).astype(np.float32))
print("input:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("sigmoid:", workspace.FetchBlob("Y"))

```

**Result**

```

input: [ 1.5744036   0.31632107  1.7842269   1.4450722  -2.1726978 ]
sigmoid: [0.8284105  0.57842743 0.85621804 0.80923885 0.10222916]

```

</details>


)DOC")
  .Input(0, "X", "*(type: Tensor`<float>`)* Input tensor.")
  .Output(0, "Y", "*(type: Tensor`<float>`)* Output tensor.")
  .InheritOnnxSchema("Sigmoid");
// Input: Y, dY, output: dX
OPERATOR_SCHEMA(SigmoidGradient)
  .NumInputs(2)
  .NumOutputs(1)
  .AllowInplace({{1, 0}})
  .SetDoc(R"DOC(
SigmoidGradient takes both Y and dY and uses this to update dX according to the
chain rule and derivatives of the sigmoid function.
)DOC");

class GetSigmoidGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SigmoidGradient", "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Sigmoid, GetSigmoidGradient);
}  // namespace caffe2
