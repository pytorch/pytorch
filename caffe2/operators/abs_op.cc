#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

struct AbsCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* device_context) {
    math::Abs<T, CPUContext>(n, x, y, device_context);
  }
};

struct AbsGradientCPUFunctor {
  template <typename T>
  inline void
  Run(const int n, const T* x, const T* dy, T* dx, CPUContext* /* unused */) {
    ConstEigenVectorArrayMap<T> dyM(dy, n);
    ConstEigenVectorArrayMap<T> xM(x, n);
    EigenVectorMap<T>(dx, n) =
        (xM == T(0)).select(T(0), (xM > T(0)).select(dyM, -dyM));
  }
};

REGISTER_CPU_OPERATOR(
    Abs,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, AbsCPUFunctor>);
REGISTER_CPU_OPERATOR(
    AbsGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        WithoutBroadcast<AbsGradientCPUFunctor>>);

OPERATOR_SCHEMA(Abs)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the absolute value of the given input tensor, element-wise.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/abs_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "Abs",
    ["X"],
    ["Y"]
)

workspace.FeedBlob("X", np.random.randn(5).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))

```

**Result**

```

X: [ 0.3005476   1.551666   -1.3591481   0.39191285 -0.21866608]
Y: [0.3005476  1.551666   1.3591481  0.39191285 0.21866608]

```

</details>

)DOC")
    .Input(0, "X", "*(type: Tensor<float\>)* Input tensor.")
    .Output(
        0,
        "Y",
        "*(type: Tensor`<float>`)* Absolute value of input element-wise.")
    .InheritOnnxSchema("Abs");

OPERATOR_SCHEMA(AbsGradient).NumInputs(2).NumOutputs(1).IdenticalTypeAndShape();

class GetAbsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "AbsGradient",
        "",
        std::vector<string>{I(0), GO(0)},
        std::vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Abs, GetAbsGradient);
} // namespace caffe2
