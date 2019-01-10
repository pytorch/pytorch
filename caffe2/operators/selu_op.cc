#include "caffe2/operators/selu_op.h"

#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool SeluOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);

  auto* Y = Output(0, X.sizes(), at::dtype<float>());

  ConstEigenVectorArrayMap<float> Xvec(X.data<float>(), X.numel());
  EigenVectorArrayMap<float> Yvec(
      Y->template mutable_data<float>(), Y->numel());
  Yvec = lambda_ * (Xvec > 0).select(Xvec, (alpha_ * Xvec.exp() - alpha_));
  return true;
}

template <>
bool SeluGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);

  CAFFE_ENFORCE_EQ(dY.numel(), Y.numel());
  auto* dX = Output(0, Y.sizes(), at::dtype<float>());

  ConstEigenVectorArrayMap<float> Yvec(Y.data<float>(), Y.numel());
  ConstEigenVectorArrayMap<float> dYvec(dY.data<float>(), dY.numel());
  EigenVectorArrayMap<float> dXvec(
      dX->template mutable_data<float>(), dX->numel());

  const float la = lambda_ * alpha_;
  dXvec = (Yvec > 0).select(lambda_ * dYvec, dYvec * (Yvec + la));
  return true;
}

REGISTER_CPU_OPERATOR(Selu, SeluOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(SeluGradient, SeluGradientOp<float, CPUContext>);

// Input: X; output: Y
OPERATOR_SCHEMA(Selu)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(

The *Selu* op takes one input tensor $X$, an argument $alpha$, an argument $scale$, and produces one output tensor $Y$ of the same shape as $X.$ The op performs the element wise *Selu* operation, defined as

$$y=selu(x) =\begin{cases}scale (\alpha e^{x} - \alpha) & x < 0\\scale * x & otherwise\end{cases}$$

The default value of *alpha* is 1.6732632423543772848170429916717 and the default value of *scale* is 1.0507009873554804934193349852946. See [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515) for more information.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/selu_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/selu_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Selu",
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
 [[ 1.1613879  -0.27111396 -1.2076733 ]
 [ 1.3442237  -1.0701777   1.2070968 ]
 [ 0.23810555  0.9740916  -1.7872391 ]]

Y:
 [[ 1.2202715  -0.4174965  -1.2326177 ]
 [ 1.4123772  -1.1551634   1.2682979 ]
 [ 0.25017774  1.023479   -1.4637551 ]]

```

</details>

)DOC")
    .Arg(
        "alpha",
        "*(type: float; default: 1.673263~)* Alpha constant in equation.")
    .Arg(
        "scale",
        "*(type: float; default: 1.050700~; must be > 1.0)* Scale constant in equation.")
    .Input(0, "X", "Input tensor of data to be operated on.")
    .Output(0, "Y", "Output tensor with same shape as input.")
    .InheritOnnxSchema();

// Input: Y, dY; output: dX
OPERATOR_SCHEMA(SeluGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}})
    .SetDoc(R"DOC(
SeluGradient takes both Y and dY and uses this to update dX according to the
chain rule and derivatives of the selu function.
)DOC")
    .Arg(
        "alpha",
        "(float) default to 1.6732~; affects the activation function itself."
        "This should go with the weight initialization in the paper. "
        " See https://arxiv.org/abs/1706.02515 ")
    .Arg(
        "scale",
        "(float) default to 1.0507~; affects the activation function itself.")
    .Input(0, "Y", "input tensor")
    .Input(1, "dY", "input tensor");

class GetSeluGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Selu, GetSeluGradient);

} // namespace caffe2
