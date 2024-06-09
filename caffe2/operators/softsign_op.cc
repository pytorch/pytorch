#include "caffe2/operators/softsign_op.h"

#include "caffe2/utils/eigen_utils.h"

#include <algorithm>
#include <functional>

namespace caffe2 {

template <>
template <typename T>
bool SoftsignFunctor<CPUContext>::
operator()(const int N, const T* X, T* Y, CPUContext* /* context */) const {
  ConstEigenVectorArrayMap<T> X_arr(X, N);
  EigenVectorMap<T>(Y, N) = (T(1) + X_arr.abs()).inverse() * X_arr;
  return true;
}

template <>
template <typename T>
bool SoftsignGradientFunctor<CPUContext>::Forward(
    const std::vector<int>& X_dims,
    const std::vector<int>& /* dY_dims */,
    const T* X,
    const T* dY,
    T* dX,
    CPUContext* /* context */) const {
  const int size = std::accumulate(
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
  ConstEigenVectorArrayMap<T> dY_arr(dY, size);
  ConstEigenVectorArrayMap<T> X_arr(X, size);
  EigenVectorMap<T>(dX, size) =
      dY_arr * (T(1) + X_arr.abs()).square().inverse();
  return true;
}

REGISTER_CPU_OPERATOR(
    Softsign,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        SoftsignFunctor<CPUContext>>);
REGISTER_CPU_GRADIENT_OPERATOR(
    SoftsignGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        SoftsignGradientFunctor<CPUContext>>);

OPERATOR_SCHEMA(Softsign)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
*Softsign* takes one input data tensor $X$ and produces one output data $Y,$ where the softsign function, $y = \frac{x}{1+ |x|}$, is applied to $X$ elementwise. This operation can be done in an in-place fashion too, by providing the same input and output blobs.

Github Links:

- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/softsign_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Softsign",
    ["X"],
    ["Y"],
)

workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32))
print("X:\n", workspace.FetchBlob("X"), "\n")

workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))

```

**Result**

```

X:
 [[-1.3060539   0.7242748  -1.9907674 ]
 [-0.64802396 -0.03244735  0.7455406 ]
 [-0.298492   -0.5774271   2.8364444 ]]

Y:
 [[-0.5663588   0.420046   -0.6656376 ]
 [-0.39321268 -0.03142761  0.4271116 ]
 [-0.2298759  -0.36605626  0.739342  ]]

```

</details>


)DOC")
    .Input(0, "input", "Input data blob to be operated on.")
    .Output(0, "output", "Output data blob with same shape as input")
    .InheritOnnxSchema();

GRADIENT_OPERATOR_SCHEMA(SoftsignGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}})
    .SetDoc(R"DOC(
Calculates the softsign gradient (sgn(x)/(1+|x|)^2) of the given input tensor
element-wise.
)DOC")
    .Input(0, "input", "1-D input tensor")
    .Input(1, "input", "1-D input tensor")
    .Output(
        0,
        "output",
        "The softsign gradient (sgn(x)/(1+|x|)^2) values of the input tensor "
        "computed element-wise");

namespace {

class GetSoftsignGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(
        I(0) != O(0),
        "Cannot compute softsign gradient "
        "if you choose to do an in-place calculation.");

    return SingleGradientDef(
        "SoftsignGradient",
        "",
        std::vector<std::string>{I(0), GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Softsign, GetSoftsignGradient);

} // namespace caffe2
