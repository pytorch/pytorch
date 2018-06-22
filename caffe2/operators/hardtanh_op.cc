#include "caffe2/operators/hardtanh_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool HardtanhOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);

  ConstEigenVectorArrayMap<float> Xvec(X.data<float>(), X.size());
  EigenVectorMap<float>(Y->mutable_data<float>(), X.size()) =
      ConstEigenVectorMap<float>(X.data<float>(), X.size()).cwiseMax(min_val_).cwiseMin(max_val_);
  return true;
}

template <>
bool HardtanhGradientOp<float, CPUContext>::RunOnDevice() {
  const auto& Y = Input(1);
  const auto& dY = Input(0);
  auto* dX = Output(0);
  // Sizes are known to be equal because Hardtanh is element-wise
  CAFFE_ENFORCE_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);

  ConstEigenVectorArrayMap<float> Yvec(Y.data<float>(), Y.size());
  ConstEigenVectorArrayMap<float> dYvec(dY.data<float>(), dY.size());
  EigenVectorArrayMap<float> dXvec(dX->mutable_data<float>(), dX->size());

  // dXvec should be 0 if Y = max_val_ or min_val, dYvec otherwise
  // Nest select within initial select
  dXvec = (Yvec >= max_val_).select(0, (Yvec <= min_val_).select(0, dYvec));
  return true;
}

REGISTER_CPU_OPERATOR(Hardtanh, HardtanhOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(HardtanhGradient, HardtanhGradientOp<float, CPUContext>);

// Input: X; output: Y
OPERATOR_SCHEMA(Hardtanh)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(

The *Hardtanh* op takes one input tensor $X$, an argument $min\_val$, an argument $max\_val$, and produces one output tensor $Y$ of the same shape as $X.$ The op performs the element wise *Hardtanh* operation, defined as

$$y=hardtanh(x) =\begin{cases}max\_val & x > max\_val\\min\_val & x < min\_val\\x & otherwise\end{cases}$$

The default value of *min_val* is -1 and the default value of *max_val* is 1.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/hardtanh_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/hardtanh_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Hardtanh",
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
 [[-1.109332  , -0.09768647,  0.7041394 ]
 [-0.37329328,  1.0993835 , -1.1562432 ]
 [ 0.03309243,  2.2267528 ,  0.4989338 ]]

Y:
 [[ -1.        , -0.09768647,  0.7041394 ]
 [-0.37329328,  1.        , -1.        ]
 [ 0.03309243,  1.        ,  0.4989338 ]]

```

</details>

)DOC")
    .Arg("min_val", "*(type: float; default: -1~)* min_val constant in equation.")
    .Arg("max_val", "*(type: float; default: 1~; must be > min_val)* max_val constant in equation.")
    .Input(0, "X", "Input tensor of data to be operated on.")
    .Output(0, "Y", "Output tensor with same shape as input.")
    .InheritOnnxSchema("Hardtanh");

// Input: dY, Y; output: dX
OPERATOR_SCHEMA(HardtanhGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}})
    .SetDoc(R"DOC(
HardtanhGradient takes both dY and Y and uses this to update dX according to the
derivatives of the hardtanh function.
)DOC")
    .Arg(
        "min_val",
        "(float) default to -1~; affects the activation function itself.")
    .Arg(
        "max_val",
        "(float) default to 1~; affects the activation function itself.")
    .Input(0, "Y", "input tensor")
    .Input(1, "dY", "input tensor");

class GetHardtanhGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        vector<string>{GO(0), O(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Hardtanh, GetHardtanhGradient);

} // namespace caffe2
