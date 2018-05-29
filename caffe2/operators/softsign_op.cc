#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

struct SoftsignCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* /*device_context*/) {
    ConstEigenVectorArrayMap<T> x_arr(x, n);
    EigenVectorMap<T>(y, n) = (1 + x_arr.abs()).inverse() * x_arr;
  }
};

struct SoftsignGradientCPUFunctor {
  template <typename T>
  inline void Run(
      const int n,
      const T* x,
      const T* dy,
      T* dx,
      CPUContext* /*device_context*/) {
    ConstEigenVectorArrayMap<T> dy_arr(dy, n);
    ConstEigenVectorArrayMap<T> x_arr(x, n);
    EigenVectorMap<T>(dx, n) = dy_arr * (1 + x_arr.abs()).pow(2).inverse();
  }
};

REGISTER_CPU_OPERATOR(
    Softsign,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, SoftsignCPUFunctor>);
REGISTER_CPU_OPERATOR(
    SoftsignGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        WithoutBroadcast<SoftsignGradientCPUFunctor>>);

OPERATOR_SCHEMA(Softsign)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
*Softsign* takes one input data tensor $X$ and produces one output data $Y,$ where the softsign function, $y = \frac{x}{1+ |x|}$, is applied to $X$ elementwise. This operation can be done in an in-place fashion too, by providing the same input and output blobs.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softsign_op.cc


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
    .Output(0,"output", "Output data blob with same shape as input")
    .InheritOnnxSchema("Softsign");

OPERATOR_SCHEMA(SoftsignGradient)
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

class GetSoftsignGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(
        I(0) != O(0),
        "Cannot compute softsign gradient "
        "if you choose to do an in-place calculation.");

    return SingleGradientDef(
        "SoftsignGradient",
        "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(Softsign, GetSoftsignGradient);

} // namespace caffe2
