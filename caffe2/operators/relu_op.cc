#include "caffe2/operators/relu_op.h"

#include <algorithm>
#include <functional>
#include <string>

#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

template <>
template <typename T>
bool ReluFunctor<CPUContext>::
operator()(const int N, const T* X, T* Y, CPUContext* /* context */) const {
  EigenVectorMap<T>(Y, N) = ConstEigenVectorMap<float>(X, N).cwiseMax(T(0));
  return true;
}

#ifdef CAFFE2_USE_ACCELERATE

template <>
template <>
bool ReluFunctor<CPUContext>::operator()<float>(
    const int N,
    const float* X,
    float* Y,
    CPUContext* /* context */) const {
  const float zero = 0.0f;
  vDSP_vthres(X, 1, &zero, Y, 1, N);
  return true;
}

#endif // CAFFE2_USE_ACCELERATE

template <>
template <typename T>
bool ReluGradientFunctor<CPUContext>::Forward(
    const std::vector<int>& Y_dims,
    const std::vector<int>& /* dY_dims */,
    const T* Y,
    const T* dY,
    T* dX,
    CPUContext* /* context */) const {
  const int size = std::accumulate(
      Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
  EigenVectorArrayMap<T>(dX, size) =
      (ConstEigenVectorArrayMap<T>(Y, size) > T(0))
          .select(ConstEigenVectorArrayMap<T>(dY, size), T(0));
  return true;
}

namespace {

OpSchema::Cost CostInferenceForRelu(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  struct OpSchema::Cost cost = PointwiseCostInference<0>(def, in);
  cost.params_bytes = 0;
  return cost;
}

} // namespace

REGISTER_CPU_OPERATOR(
    Relu,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        ReluFunctor<CPUContext>>);
REGISTER_CPU_GRADIENT_OPERATOR(
    ReluGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        ReluGradientFunctor<CPUContext>>);

// Input: X, output: Y
OPERATOR_SCHEMA(Relu)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(CostInferenceForRelu)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Applies rectified linear unit operation to the input data element-wise. The Relu operation takes one input $X$, produces one output $Y$, and is defined as:

$$Y = max(0,X)$$

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/relu_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/relu_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
  "Relu",
  ["X"],
  ["Y"]
  )

workspace.FeedBlob("X", np.random.randn(4, 4).astype(np.float32)) // NCHW
print("X:\n", workspace.FetchBlob("X"), "\n")

workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))

```

**Result**

```

X:
 [[-1.4655551   0.64575136  0.7921748   0.4150579 ]
 [ 0.41085166 -0.2837964   0.9881425  -1.9300346 ]
 [ 0.39705405  0.44639114  0.9940703   0.2926532 ]
 [-0.6726489   0.01330667  1.101319    0.33858967]]

Y:
 [[0.         0.64575136 0.7921748  0.4150579 ]
 [0.41085166 0.         0.9881425  0.        ]
 [0.39705405 0.44639114 0.9940703  0.2926532 ]
 [0.         0.01330667 1.101319   0.33858967]]

```

</details>


)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D output tensor with same shape as input")
    .InheritOnnxSchema();

// Input: Y, dY, output: dX
GRADIENT_OPERATOR_SCHEMA(ReluGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}})
    .IdenticalTypeAndShapeOfInput(1)
    .SetDoc(R"DOC(
ReluGradient takes both Y and dY and uses this to update dX according to the
chain rule and derivatives of the rectified linear function.
)DOC");

namespace {

class GetReluGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        std::vector<std::string>{O(0), GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Relu, GetReluGradient);

} // namespace caffe2
