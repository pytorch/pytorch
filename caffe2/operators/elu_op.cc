#include "caffe2/operators/elu_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool EluOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  // Otherwise inplace gradient and Elu dosen't make sense.
  CAFFE_ENFORCE_GE(alpha_, 0);
  Y->ResizeLike(X);
  const auto* Xdata = X.template data<float>();
  auto* Ydata = Y->template mutable_data<float>();
  ConstEigenVectorArrayMap<float> Xvec(Xdata, X.size());
  EigenVectorArrayMap<float> Yvec(Ydata, Y->size());
  Yvec = Xvec.cwiseMax(0.f) + (alpha_ * (Xvec.exp() - 1.0f)).cwiseMin(0.f);
  return true;
}

template <>
bool EluGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  DCHECK_GT(Y.size(), 0);
  DCHECK_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);

  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  ConstEigenVectorArrayMap<float> Yvec(Ydata, Y.size());
  ConstEigenVectorArrayMap<float> dYvec(dYdata, dY.size());
  EigenVectorArrayMap<float> dXvec(dXdata, dX->size());
  dXvec = (Yvec > 0).select(dYvec, dYvec * (Yvec + alpha_));
  return true;
}

REGISTER_CPU_OPERATOR(Elu, EluOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(EluGradient, EluGradientOp<float, CPUContext>);

// Input: X, output: Y
OPERATOR_SCHEMA(Elu)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(

This op implements the exponential linear unit (ELU) activation function as described in [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289). The op takes an input tensor $X$ of arbitrary shape, computes the elementwise elu operation, and returns a vector $Y$ of the same shape as output. The alpha parameter may be passed as an argument, but defaults to 1. The elu operation is defined as

$$y=f(x) =\begin{cases}\alpha(e^x-1) & x < 0 \\ x & otherwise\end{cases}$$

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elu_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elu_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "Elu",
    ["X"],
    ["Y"],
    alpha=1.1
)

workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32))
print("X:\n", workspace.FetchBlob("X"), "\n")

workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))

```

**Result**

```

X:
 [[ 0.35339102  1.1860217  -0.10710736]
 [-3.1173866  -0.1889988  -0.20330353]
 [ 1.8525308  -0.368949    0.506277  ]]

Y:
 [[ 0.35339102  1.1860217  -0.11172786]
 [-1.0513     -0.18943374 -0.20236646]
 [ 1.8525308  -0.33939326  0.506277  ]]

```

</details>

)DOC")
    .Input(0, "X", "1D input tensor of data to be operated on.")
    .Output(0, "Y", "1D input tensor, calculated as described above.")
    .Arg("alpha", "*(type: float; default: 1.0)* Defines alpha parameter used in calculation.")
    .InheritOnnxSchema("Elu");


// Input: Y, dY, output: dX
OPERATOR_SCHEMA(EluGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}})
    .SetDoc(R"DOC(
EluGradient takes both Y and dY and uses this to update dX according to the
chain rule and derivatives of the rectified linear function.
)DOC");

class GetEluGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Elu, GetEluGradient);

} // namespace caffe2
