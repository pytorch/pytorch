#include "elementwise_linear_op.h"

namespace caffe2 {

template<>
bool ElementwiseLinearOp<float, CPUContext>::RunOnDevice(){
  const auto& X = Input(0);
  const auto& a = Input(1);
  const auto& b = Input(2);

  const auto canonical_axis = X.canonical_axis_index(axis_);
  const int N = X.size_to_dim(canonical_axis);
  const int D = X.size_from_dim(canonical_axis);

  CAFFE_ENFORCE_EQ(a.dim(), 1, a.dim());
  CAFFE_ENFORCE_EQ(a.size(0), D, a.dim());
  CAFFE_ENFORCE_EQ(b.dim(), 1, b.dim());
  CAFFE_ENFORCE_EQ(b.size(0), D, b.dim());

  auto* Y = Output(0, X.sizes(), at::dtype<float>());

  const float* X_data = X.data<float>();
  const float* a_data = a.data<float>();
  const float* b_data = b.data<float>();
  float* Y_data = Y->template mutable_data<float>();

  int p = 0;
  for (int n = 0; n < N; ++n) {
    for (int d = 0; d < D; ++d) {
      Y_data[p] = X_data[p] * a_data[d] + b_data[d];
      p++;
    }
  }
  return true;
}

template<>
bool ElementwiseLinearGradientOp<float, CPUContext>::RunOnDevice(){
  const auto& g_o = Input(0);
  const auto& X = Input(1);
  const auto& a = Input(2);

  const auto canonical_axis = X.canonical_axis_index(axis_);
  const int N = X.size_to_dim(canonical_axis);
  const int D = X.size_from_dim(canonical_axis);

  CAFFE_ENFORCE_EQ(a.dim(), 1, a.dim());
  CAFFE_ENFORCE_EQ(a.size(0), D, a.dim());

  auto* g_X = Output(0, X.sizes(), at::dtype<float>());
  auto* g_a = Output(1, a.sizes(), at::dtype<float>());
  auto* g_b = Output(2, a.sizes(), at::dtype<float>());

  const float* g_o_data = g_o.data<float>();
  const float* X_data = X.data<float>();
  const float* a_data = a.data<float>();
  float* g_X_data = g_X->template mutable_data<float>();
  float* g_a_data = g_a->template mutable_data<float>();
  float* g_b_data = g_b->template mutable_data<float>();

  math::Set<float, CPUContext>(g_a->numel(), 0.f, g_a_data, &context_);
  math::Set<float, CPUContext>(g_b->numel(), 0.f, g_b_data, &context_);

  int p = 0;
  for (int n = 0; n < N; ++n) {
    for (int d = 0; d < D; ++d) {
      g_X_data[p] = g_o_data[p] * a_data[d];
      g_a_data[d] += g_o_data[p] * X_data[p];
      g_b_data[d] += g_o_data[p];
      p++;
    }
  }
  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
  ElementwiseLinear,
  ElementwiseLinearOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
  ElementwiseLinearGradient,
  ElementwiseLinearGradientOp<float, CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(ElementwiseLinear)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
This op computes the elementwise linear combination of a batch of input vectors with a weight vector and bias vector. As input, the op takes an input tensor $X$ of shape $NxD$, a weight vector $w$ of length $D$, and a bias vector $b$ of length $D$. Here, $N$ represents the batch size and $D$ represents the length of the feature vectors. The output, $Y$, is a tensor of shape $NxD$ and is calculated as

$$Y_{ij} = X_{ij}w_j + b_j \ for \ i\in{N}, j\in{D}$$

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_linear_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_linear_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "ElementwiseLinear",
    ["X", "w", "b"],
    ["Y"]
)

// Create X
X = np.array([[1,2,3,4,5],[6,8,9,16,10]])
print("X:\n",X)

// Create w
w = np.array([1,1/2.,1/3.,1/4.,1/5.])
print("w:\n",w)

// Create b
b = np.array([1.,1.,1.,1.,1.])
print("b:\n",b)


// Feed X & w & b into workspace
workspace.FeedBlob("X", X.astype(np.float32))
workspace.FeedBlob("w", w.astype(np.float32))
workspace.FeedBlob("b", b.astype(np.float32))

// Run op
workspace.RunOperatorOnce(op)

// Collect Output
print("Y:\n", workspace.FetchBlob("Y"))

```

**Result**

```

X:
 [[ 1  2  3  4  5]
 [ 6  8  9 16 10]]
w:
 [1.  0.5  0.33333333 0.25  0.2]
b:
 [1. 1. 1. 1. 1.]
Y:
 [[2. 2. 2. 2. 2.]
 [7. 5. 4. 5. 3.]]

```

</details>

  )DOC")
    .Input(
        0,
        "X",
        "2D input tensor of size $NxD$. This input represents the input data to be operated on.")
    .Input(
        1,
        "w",
        "1D scaling factors, or weights, of size $D$. This input contains the weights that will be multiplied by the data.")
    .Input(
        2,
        "b",
        "1D biases of size $D$. This input contains the biases that will be added to the products of the weights and data.")
    .Output(
        0,
        "Y",
        "2D output tensor of size $NxD$. Calculated as described above.")
    .Arg(
        "axis",
        "*(type: int; default: 1)* Describes the axis of the inputs; defaults to one because the 0th axis most likely describes the batch size.")
    .InheritOnnxSchema();

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(ElementwiseLinearGradient)
  .NumInputs(3)
  .NumOutputs(3);

struct GetElementwiseLinearGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
      "ElementwiseLinearGradient",
      "",
      vector<string>{GO(0), I(0), I(1)},
      vector<string>{GI(0), GI(1), GI(2)});
    }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(
  ElementwiseLinear,
  GetElementwiseLinearGradient
);

}  // namespace caffe2
