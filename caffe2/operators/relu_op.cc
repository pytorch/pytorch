#include "caffe2/operators/relu_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool ReluOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);

#ifdef CAFFE2_USE_ACCELERATE
  const float zero = 0.0f;
  vDSP_vthres(X.data<float>(), 1, &zero, Y->mutable_data<float>(), 1, X.size());
#else
  EigenVectorMap<float>(Y->mutable_data<float>(), X.size()) =
      ConstEigenVectorMap<float>(X.data<float>(), X.size()).cwiseMax(0.f);
#endif
  /* Naive implementation
  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  for (int i = 0; i < X.size(); ++i) {
    Ydata[i] = std::max(Xdata[i], 0.f);
  }
  */
  return true;
}

template <>
bool ReluGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  CAFFE_ENFORCE_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);

  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  // TODO: proper vectorization with Eigen
  EigenVectorArrayMap<float> dXvec(dXdata, dX->size());
  ConstEigenVectorArrayMap<float> Yvec(Ydata, Y.size());
  ConstEigenVectorArrayMap<float> dYvec(dYdata, dY.size());
  dXvec = dYvec * Yvec.cwiseSign();
  /* Previous implementation
  for (int i = 0; i < Y.size(); ++i) {
    dXdata[i] = Ydata[i] > 0 ? dYdata[i] : 0;
  }
  */
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

REGISTER_CPU_OPERATOR(Relu, ReluOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(ReluGradient, ReluGradientOp<float, CPUContext>);

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

workspace.FeedBlob("X", np.random.randn(4, 4).astype(np.float32)) # NCHW
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
    .InheritOnnxSchema("Relu");

// Input: Y, dY, output: dX
OPERATOR_SCHEMA(ReluGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}})
    .SetDoc(R"DOC(
ReluGradient takes both Y and dY and uses this to update dX according to the
chain rule and derivatives of the rectified linear function.
)DOC");

class GetReluGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Relu, GetReluGradient);
REGISTER_GRADIENT(ReluFp16, GetReluGradient);

}  // namespace caffe2
