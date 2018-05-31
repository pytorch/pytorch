#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

struct CosCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* device_context) {
    math::Cos<T, CPUContext>(n, x, y, device_context);
  }
};

struct CosGradientCPUFunctor {
  template <typename T>
  inline void
  Run(const int n, const T* x, const T* dy, T* dx, CPUContext* /* unused */) {
    ConstEigenVectorArrayMap<T> dyM(dy, n);
    ConstEigenVectorArrayMap<T> xM(x, n);
    EigenVectorMap<T>(dx, n) = -dyM * sin(xM);
  }
};

REGISTER_CPU_OPERATOR(
    Cos,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, CosCPUFunctor>);
REGISTER_CPU_OPERATOR(
    CosGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        WithoutBroadcast<CosGradientCPUFunctor>>);

OPERATOR_SCHEMA(Cos)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the cosine of the given input tensor, element-wise.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cos_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Cos",
    ["X"],
    ["Y"]
)

workspace.FeedBlob("X", np.random.rand(5).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))

```

**Result**

```

X: [0.6816719  0.76771533 0.933932   0.01404487 0.11862425]
Y: [0.7765203  0.71949923 0.5946774  0.99990135 0.9929724 ]

```

</details>


)DOC")
    .Input(0, "X", "*(type: Tensor`<float>`)* Input tensor.")
    .Output(
        0,
        "Y",
        "*(type: Tensor`<float>`)* Output tensor calculated as the cosine of the input tensor, element-wise.");

OPERATOR_SCHEMA(CosGradient).NumInputs(2).NumOutputs(1).IdenticalTypeAndShape();

class GetCosGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "CosGradient",
        "",
        std::vector<string>{I(0), GO(0)},
        std::vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Cos, GetCosGradient);
} // namespace caffe2
