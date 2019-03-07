#include "caffe2/operators/distance_op.h"
#include "caffe2/utils/eigen_utils.h"
#ifdef CAFFE2_USE_MKLDNN
#include <caffe2/ideep/operators/operator_fallback_ideep.h>
#include <caffe2/ideep/utils/ideep_operator.h>
#endif

namespace caffe2 {

template<>
bool SquaredL2DistanceOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);

  CAFFE_ENFORCE_EQ(X.dim(), Y.dim());
  for (int i = 0; i < X.dim(); ++i) {
    CAFFE_ENFORCE_EQ(X.dim32(i), Y.dim32(i));
  }
  int N = X.dim() > 0 ? X.dim32(0) : 1;
  auto* distance = Output(0, {N}, at::dtype<float>());
  int D = N > 0 ? X.numel() / N : 0;
  float* distance_data = distance->template mutable_data<float>();
  const float* X_data = X.data<float>();
  const float* Y_data = Y.data<float>();
  for (int i = 0; i < N; ++i) {
    float Xscale, Yscale, cross;
    math::Dot<float, CPUContext>(
        D, X_data + i * D, X_data + i * D, &Xscale, &context_);
    math::Dot<float, CPUContext>(
        D, Y_data + i * D, Y_data + i * D, &Yscale, &context_);
    math::Dot<float, CPUContext>(
        D, X_data + i * D, Y_data + i * D, &cross, &context_);
    distance_data[i] = (Xscale + Yscale) * 0.5 - cross;
  }
  return true;
}

template <>
bool L1DistanceOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);

  CAFFE_ENFORCE_EQ(X.dim(), Y.dim());
  for (int i = 0; i < X.dim(); ++i) {
    CAFFE_ENFORCE_EQ(X.dim32(i), Y.dim32(i));
  }
  int N = X.dim() > 0 ? X.dim32(0) : 1;
  auto* distance = Output(0, {N}, at::dtype<float>());
  int D = N > 0 ? X.numel() / N : 0;

  const float* X_data = X.data<float>();
  const float* Y_data = Y.data<float>();

  for (int i = 0; i < N; ++i) {
    (distance->template mutable_data<float>())[i] =
        (ConstEigenVectorMap<float>(X_data + i * D, D).array() -
         ConstEigenVectorMap<float>(Y_data + i * D, D).array())
            .abs()
            .sum();
  }
  return true;
}

template <>
bool L1DistanceGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dDistance = Input(2);

  CAFFE_ENFORCE_EQ(X.dim(), Y.dim());
  for (int i = 0; i < X.dim(); ++i) {
    CAFFE_ENFORCE_EQ(X.dim32(i), Y.dim32(i));
  }
  int N = X.dim() > 0 ? X.dim32(0) : 1;
  int D = N > 0 ? X.numel() / N : 0;
  CAFFE_ENFORCE(X.dim() == Y.dim());
  for (int i = 0; i < X.dim(); ++i) {
    CAFFE_ENFORCE(X.dim32(i) == Y.dim32(i));
  }
  CAFFE_ENFORCE(dDistance.dim() == 1);
  CAFFE_ENFORCE(dDistance.dim32(0) == N);
  auto* dX = Output(0, X.sizes(), at::dtype<float>());
  auto* dY = Output(1, Y.sizes(), at::dtype<float>());

  for (int i = 0; i < N; ++i) {
    auto offset = i * D;
    for (int j = 0; j < D; ++j) {
      const float temp =
          (X.data<float>())[offset + j] - (Y.data<float>())[offset + j];
      const float kEps = 1e-12f;
      if (temp < -kEps) {
        dX->template mutable_data<float>()[offset + j] =
            -(dDistance.data<float>())[i];
        dY->template mutable_data<float>()[offset + j] =
            (dDistance.data<float>())[i];
      } else if (temp > kEps) {
        dX->template mutable_data<float>()[offset + j] =
            (dDistance.data<float>())[i];
        dY->template mutable_data<float>()[offset + j] =
            -(dDistance.data<float>())[i];
      } else {
        dX->template mutable_data<float>()[offset + j] = 0;
        dY->template mutable_data<float>()[offset + j] = 0;
      }
    }
  }
  return true;
}

template <>
bool CosineSimilarityOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(X_IN);
  auto& Y = Input(Y_IN);

  CAFFE_ENFORCE_EQ(X.dim(), Y.dim());
  for (int i = 0; i < X.dim(); ++i) {
    CAFFE_ENFORCE_EQ(X.dim32(i), Y.dim32(i));
  }
  const int N = X.dim() > 0 ? X.dim32(0) : 1;
  const int D = X.size_from_dim(1);
  auto* result = Output(COS_OUT, {N}, at::dtype<float>());
  float* result_data = result->template mutable_data<float>();
  const float* X_data = X.data<float>();
  const float* Y_data = Y.data<float>();
  float X2, Y2;
  const float kEps = 1e-12f;
  for (int i = 0; i < N; ++i) { // TODO: multithreading
    auto offset = i * D;
    math::Dot<float, CPUContext>(
        D, X_data + offset, X_data + offset, &X2, &context_);
    math::Dot<float, CPUContext>(
        D, Y_data + offset, Y_data + offset, &Y2, &context_);
    math::Dot<float, CPUContext>(
        D, X_data + offset, Y_data + offset, result_data + i, &context_);
    result_data[i] /= std::sqrt(std::max(X2, kEps) * std::max(Y2, kEps));
  }
  return true;
}

template <>
bool CosineSimilarityGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(X_IN);
  auto& Y = Input(Y_IN);
  auto& dCos = Input(DER_COS_IN);

  const int N = X.dim() > 0 ? X.dim32(0) : 1;
  const int D = X.size_from_dim(1);
  CAFFE_ENFORCE(X.dim() == Y.dim());
  for (int i = 0; i < X.dim(); ++i) {
    CAFFE_ENFORCE(X.dim32(i) == Y.dim32(i));
  }
  CAFFE_ENFORCE(dCos.dim() == 1);
  CAFFE_ENFORCE(dCos.dim32(0) == N);
  auto* dX = Output(DER_X_OUT, X.sizes(), at::dtype<float>());
  auto* dY = Output(DER_Y_OUT, Y.sizes(), at::dtype<float>());

  const auto* X_data = X.template data<float>();
  const auto* Y_data = Y.template data<float>();
  const auto* dCos_data = dCos.template data<float>();
  auto* dX_data = dX->template mutable_data<float>();
  auto* dY_data = dY->template mutable_data<float>();
  float XN, YN, XY;
  const float kEps = 1e-12f;
  for (int i = 0; i < N; ++i) { // TODO: multithreading
    auto offset = i * D;

    // TODO: cache these result from the forward pass
    // ||x||
    math::Dot<float, CPUContext>(
        D, X_data + offset, X_data + offset, &XN, &context_);
    XN = std::sqrt(std::max(XN, kEps));
    // ||y||
    math::Dot<float, CPUContext>(
        D, Y_data + offset, Y_data + offset, &YN, &context_);
    YN = std::sqrt(std::max(YN, kEps));
    // ||x|| * || y ||
    float XYN = XN * YN;
    // x^Ty
    math::Dot<float, CPUContext>(
        D, X_data + offset, Y_data + offset, &XY, &context_);

    math::Scale<float, float, CPUContext>(
        D, dCos_data[i] / XYN, Y_data + offset, dX_data + offset, &context_);
    math::Axpy(
        D,
        -dCos_data[i] * XY / (XN * XN * XYN),
        X_data + offset,
        dX_data + offset,
        &context_);

    math::Scale<float, float, CPUContext>(
        D, dCos_data[i] / XYN, X_data + offset, dY_data + offset, &context_);
    math::Axpy(
        D,
        -dCos_data[i] * XY / (YN * YN * XYN),
        Y_data + offset,
        dY_data + offset,
        &context_);
  }

  return true;
}

template <>
bool DotProductOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(X_IN);
  auto& Y = Input(Y_IN);

  CAFFE_ENFORCE_EQ(X.dim(), Y.dim());
  for (int i = 0; i < X.dim(); ++i) {
    CAFFE_ENFORCE_EQ(X.dim32(i), Y.dim32(i), "dimension at ", i);
  }
  int N, D;
  if (X.numel() > 0) {
    N = X.dim() > 0 ? X.dim32(0) : 1;
    D = X.numel() / N;
  } else {
    N = 0;
    D = 0;
  }
  auto* result = Output(DOT_OUT, {N}, at::dtype<float>());
  float* result_data = result->template mutable_data<float>();
  const float* X_data = X.template data<float>();
  const float* Y_data = Y.template data<float>();
  for (int i = 0; i < N; ++i) { // TODO: multithreading
    auto offset = i * D;
    math::Dot<float, CPUContext>(
        D, X_data + offset, Y_data + offset, result_data + i, &context_);
  }
  return true;
}

vector<TensorShape> TensorInferenceForDotProduct(
    const OperatorDef& /* def */,
    const vector<TensorShape>& in) {
  CAFFE_ENFORCE_GT(in.size(), 0);

  vector<int64_t> dims(1);
  dims[0] = in[0].dims().size() > 0 ? in[0].dims(0) : 1;
  return vector<TensorShape>{CreateTensorShape(dims, in[0].data_type())};
}

OpSchema::Cost CostInferenceForDotProduct(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  std::vector<TensorShape> out = TensorInferenceForDotProduct(def, in);
  CAFFE_ENFORCE_GT(out.size(), 0);
  CAFFE_ENFORCE_EQ(out[0].dims().size(), 1);

  struct OpSchema::Cost c = PointwiseCostInference<2>(def, in);
  c.bytes_written = out[0].dims(0) * sizeof(out[0].data_type());
  c.params_bytes = 0;
  return c;
}

template <>
bool DotProductGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(X_IN);
  auto& Y = Input(Y_IN);
  auto& dDot = Input(DER_DOT_IN);

  int N, D;
  if (X.numel() > 0) {
    N = X.dim() > 0 ? X.dim32(0) : 1;
    D = X.numel() / N;
  } else {
    N = 0;
    D = 0;
  }
  CAFFE_ENFORCE(X.dim() == Y.dim());
  for (int i = 0; i < X.dim(); ++i) {
    CAFFE_ENFORCE(X.dim32(i) == Y.dim32(i));
  }
  CAFFE_ENFORCE(dDot.dim() == 1);
  CAFFE_ENFORCE(dDot.dim32(0) == N);
  auto* dX = Output(DER_X_OUT, X.sizes(), at::dtype<float>());
  auto* dY = Output(DER_Y_OUT, Y.sizes(), at::dtype<float>());

  const auto* X_data = X.template data<float>();
  const auto* Y_data = Y.template data<float>();
  const auto* dDot_data = dDot.template data<float>();
  auto* dX_data = dX->template mutable_data<float>();
  auto* dY_data = dY->template mutable_data<float>();
  for (int i = 0; i < N; ++i) { // TODO: multithreading
    auto offset = i * D;
    math::Scale<float, float, CPUContext>(
        D, dDot_data[i], X_data + offset, dY_data + offset, &context_);
    math::Scale<float, float, CPUContext>(
        D, dDot_data[i], Y_data + offset, dX_data + offset, &context_);
  }
  return true;
}

template <>
bool DotProductWithPaddingOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(X_IN);
  auto& Y = Input(Y_IN);

  CAFFE_ENFORCE_EQ(X.dim(), Y.dim());
  CAFFE_ENFORCE_EQ(X.dim32(0), Y.dim32(0));

  int N, D, DX, DY, restD;
  if (X.numel() > 0) {
    N = X.dim() > 0 ? X.dim32(0) : 1;
    DX = X.numel() / N;
    DY = Y.numel() / N;
  } else {
    N = 0;
    DX = 0;
    DY = 0;
  }

  D = std::min(DX, DY);
  restD = std::max(DX, DY) - D;
  auto* result = Output(DOT_OUT, {N}, at::dtype<float>());
  float* result_data = result->template mutable_data<float>();
  const float* X_data = X.data<float>();
  const float* Y_data = Y.data<float>();
  for (int i = 0; i < N; ++i) { // TODO: multithreading
    auto offsetX = i * DX, offsetY = i * DY;
    if (replicate_) {
      // L_ for longer vector and S_ for shorter vector
      const float *L_data, *S_data;
      int DL, DS;
      if (DX > DY) {
        L_data = X_data + offsetX;
        S_data = Y_data + offsetY;
        DL = DX;
        DS = DY;
      } else {
        L_data = Y_data + offsetY;
        S_data = X_data + offsetX;
        DL = DY;
        DS = DX;
      }
      float sum = 0.0;
      float tmp = 0.0;
      for (int j = 0; j < DL / DS; j++) {
        math::Dot<float, CPUContext>(
            DS, L_data + j * DS, S_data, &tmp, &context_);
        sum += tmp;
      }
      *(result_data + i) = sum;
    } else {
      math::Dot<float, CPUContext>(
          D, X_data + offsetX, Y_data + offsetY, result_data + i, &context_);
    }

    if (!replicate_ && DX != DY) {
      const float* rest_data;
      float rest_sum = 0;
      if (DX > DY) {
        rest_data = X_data + offsetX + D;
      } else {
        rest_data = Y_data + offsetY + D;
      }
      math::Sum<float, CPUContext>(restD, rest_data, &rest_sum, &context_);
      result_data[i] += rest_sum * pad_value_;
    }
  }
  return true;
}

// L2
REGISTER_CPU_OPERATOR(SquaredL2Distance,
                      SquaredL2DistanceOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(SquaredL2DistanceGradient,
                      SquaredL2DistanceGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(SquaredL2Distance)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInputDim(0, 0)
    .SetDoc(R"DOC(
Given two input float tensors X, Y, and produces one output float tensor
of the L2 difference between X and Y that is computed as ||(X - Y)^2 / 2||.
)DOC")
    .Input(0, "X", "1D or 2D input tensor")
    .Input(1, "Y", "1D or 2D input tensor (must have the same shape as X)")
    .Output(0, "Z", "1D output tensor");

OPERATOR_SCHEMA(SquaredL2DistanceGradient).NumInputs(3).NumOutputs(2);

class GetSquaredL2DistanceGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SquaredL2DistanceGradient", "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0), GI(1)});
  }
};
REGISTER_GRADIENT(SquaredL2Distance, GetSquaredL2DistanceGradient);

// L1
REGISTER_CPU_OPERATOR(L1Distance, L1DistanceOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    L1DistanceGradient,
    L1DistanceGradientOp<float, CPUContext>);
#ifdef CAFFE2_USE_MKLDNN
REGISTER_IDEEP_OPERATOR(
    L1DistanceGradient,
    IDEEPFallbackOp<L1DistanceGradientOp<float, CPUContext>>);
#endif

OPERATOR_SCHEMA(L1Distance)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInputDim(0, 0)
    .SetDoc(R"DOC(
Computes the row-wise L1 Distance between the two input tensors $X$ and $Y$, which is defined as

$$L1Distance(\mathbf{x},\mathbf{y}) = \sum_{i}\mid x_i - y_i\mid$$

Note, both inputs must either be 1-dimensional or 2-dimensional and both must have the same shape. The output $Z$ will be 1-dimensional regardless and its length will equal the number of rows in the inputs.

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "L1Distance",
    ["X", "Y"],
    ["Z"]
)

// Create X
X = 5*np.ones((1, 4))
print("X:\n",X)

// Create Y
Y = np.ones((1, 4))
print("Y:\n",Y)

// Feed X & Y into workspace
workspace.FeedBlob("X", X.astype(np.float32))
workspace.FeedBlob("Y", Y.astype(np.float32))

// Run op
workspace.RunOperatorOnce(op)

// Collect Output
print("Z:\n", workspace.FetchBlob("Z"))

```

**Result**

```

X:
 [[5. 5. 5. 5.]]
Y:
 [[1. 1. 1. 1.]]
Z:
 [16.]

```

</details>

)DOC")
    .Input(0, "X", "First input tensor. (1D or 2D)")
    .Input(1, "Y", "Second input tensor. (must have the same shape as $X$)")
    .Output(0, "Z", "1D output tensor. One value for each row of the inputs.");

OPERATOR_SCHEMA(L1DistanceGradient).NumInputs(3).NumOutputs(2);

class GetL1DistanceGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "L1DistanceGradient",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0), GI(1)});
  }
};

REGISTER_GRADIENT(L1Distance, GetL1DistanceGradient);

// Dot Product
REGISTER_CPU_OPERATOR(DotProduct, DotProductOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    DotProductGradient,
    DotProductGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(DotProduct)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInputDim(0, 0)
    .SetDoc(R"DOC(
Computes and outputs the dot product of the two input float tensors `X` and `Y`.
Note that `X` and `Y` must be either 1D or 2D, and they must be the same shape.
The output tensor is 1D, which represents either the product of each element in
a respective dimension if the inputs are 1D, or the sum of the products in a
given dimension if the inputs are 2D matrices. Note that the actual dot product
is a scalar value, which is effectively the sum of the elements in the 1D
output tensor.

For 1D inputs:
Given two vectors $X = [x_0, x_1, x_2]$ and $Y = [y_0, y_1, y_2]$; $Z = [x_0 * y_0, x_1 * y_1, x_2 * y_2]$

For 2D inputs:
Given two matrices:
$$X = [[x_0^0, x_1^0, x_2^0], \\ [x_0^1, x_1^1, x_2^1], \\ [x_0^2, x_1^2, x_2^2], \\ ..., \\ [x_0^n, x_1^n, x_2^n]]$$

and

$$Y = [[y_0^0, y_1^0, y_2^0], \\ [y_0^1, y_1^1, y_2^1], \\ [y_0^2, y_1^2, y_2^2], \\ ..., \\ [y_0^n, y_1^n, y_2^n]]$$

then

$$Z =  \biggl[\Big((x_0^0 * y_0^0) + (x_1^0 * y_1^0) + (x_2^0 * y_2^0)\Big), \\ \Big((x_0^1 * y_0^1) + (x_1^1 * y_1^1) + (x_2^1 * y_2^1)\Big), \\ \Big((x_0^2 * y_0^2) + (x_1^2 * y_1^2) + (x_2^2 * y_2^2)\Big), \\ ..., \\ \Big((x_0^n * y_0^n) + (x_1^n * y_1^n) + (x_2^n * y_2^n)\Big)\biggr]$$

Github Link:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "DotProduct",
    ["X",  "Y"],
    ["Z"]
)

workspace.FeedBlob("X", np.random.randint(20, size=(5)).astype(np.float32))
workspace.FeedBlob("Y", np.random.randint(20, size=(5)).astype(np.float32))
print("X:\n", workspace.FetchBlob("X"))
print("Y:\n", workspace.FetchBlob("Y"))
workspace.RunOperatorOnce(op)
print("Z:\n", workspace.FetchBlob("X"))


workspace.ResetWorkspace()
workspace.FeedBlob("X", np.random.randint(10, size=(3,3)).astype(np.float32))
workspace.FeedBlob("Y", np.random.randint(10, size=(3,3)).astype(np.float32))
print("X:\n", workspace.FetchBlob("X"))
print("Y:\n", workspace.FetchBlob("Y"))
workspace.RunOperatorOnce(op)
print("Z:\n", workspace.FetchBlob("Z"))

```

**Result**

```

X:
 [ 2. 15.  2.  7. 12.]
Y:
 [ 3. 12.  9.  3. 18.]
Z:
 [ 2. 15.  2.  7. 12.]
X:
 [[2. 0. 4.]
 [7. 7. 4.]
 [7. 9. 9.]]
Y:
 [[2. 0. 8.]
 [9. 6. 1.]
 [7. 8. 0.]]
Z:
 [ 36. 109. 121.]

```

</details>

)DOC")
    .Input(0, "X", "*(type: Tensor`<float>`)* 1D or 2D input tensor.")
    .Input(
        1,
        "Y",
        "*(type: Tensor`<float>`)* 1D or 2D input tensor (must have the same shape as X).")
    .Output(0, "Z", "*(type: Tensor`<float>`)* 1D output tensor.")
    .TensorInferenceFunction(TensorInferenceForDotProduct)
    .CostInferenceFunction(
        OpSchema::CostInferenceFunctionType(CostInferenceForDotProduct))
    .InheritOnnxSchema();

OPERATOR_SCHEMA(DotProductGradient).NumInputs(3).NumOutputs(2);

class GetDotProductGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "DotProductGradient",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0), GI(1)});
  }
};
REGISTER_GRADIENT(DotProduct, GetDotProductGradient);

// Cosine Similarity
REGISTER_CPU_OPERATOR(CosineSimilarity, CosineSimilarityOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    CosineSimilarityGradient,
    CosineSimilarityGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(CosineSimilarity)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInputDim(0, 0)
    .SetDoc(R"DOC(
This op takes two input float tensors of the same size, $X$ and $Y$, and produces one output float tensor , $Z$, calculated as the cosine similarity between $X$ and $Y$. Recall, the cosine similarity between two tensors $X$ and $Y$ is defined as:

$$\mathbf{Z}=CosineSimilarity(\mathbf{X},\mathbf{Y}) = \frac{\mathbf{X}\cdot\mathbf{Y}}{\|\mathbf{X}\|\|\mathbf{Y}\|} = \frac{\sum_n^{i=1}X_iY_i}{\sqrt{\sum_n^{i=1}X_i^2}\sqrt{\sum_n^{i=1}Y_i^2}}$$

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "CosineSimilarity",
    ["X", "Y"],
    ["Z"]
)

// Create X
X = np.random.randn(3, 3)
print("X:\n",X)

// Create Y
Y = np.random.randn(3, 3)
print("Y:\n",Y)

// Feed X & Y into workspace
workspace.FeedBlob("X", X.astype(np.float32))
workspace.FeedBlob("Y", Y.astype(np.float32))

// Run op
workspace.RunOperatorOnce(op)

// Collect Output
print("Z:\n", workspace.FetchBlob("Z"))

```

**Result**

```

X:
 [[-0.42635564 -0.23831588 -0.25515547]
 [ 1.43914719 -1.05613228  1.01717373]
 [ 0.06883105  0.33386519 -1.46648334]]
Y:
 [[-0.90648691 -0.14241514 -1.1070837 ]
 [ 0.92152729 -0.28115511 -0.17756722]
 [-0.88394254  1.34654037 -0.80080998]]
Z:
 [-1.7849885e-23  1.7849885e-23 -1.0842022e-07]

```

</details>

)DOC")
    .Input(0, "X", "1D or 2D input tensor")
    .Input(1, "Y", "1D or 2D input tensor (must have the same shape as X)")
    .Output(0, "Z", "1D output tensor");

OPERATOR_SCHEMA(CosineSimilarityGradient).NumInputs(3).NumOutputs(2);

class GetCosineSimilarityGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "CosineSimilarityGradient",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0), GI(1)});
  }
};
REGISTER_GRADIENT(CosineSimilarity, GetCosineSimilarityGradient);

// Dot Product allows padding
REGISTER_CPU_OPERATOR(
    DotProductWithPadding,
    DotProductWithPaddingOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    DotProductWithPaddingGradient,
    DotProductWithPaddingGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(DotProductWithPadding)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given two input float tensors X, Y with different shapes and produces one
output float tensor of the dot product between X and Y. We currently support
two kinds of strategies to achieve this. Before doing normal dot_product 1)
pad the smaller tensor (using pad_value) to the same shape as the other one.
2) replicate the smaller tensor to the same shape as the other one. Note the
first dimension of X, Y must be equal. Only the second dimension of X or Y
can be padded.
)DOC")
    .Input(0, "X", "1D or 2D input tensor")
    .Input(1, "Y", "1D or 2D input tensor")
    .Output(0, "Z", "1D output tensor")
    .IdenticalTypeAndShapeOfInputDim(0, 0)
    .Arg("pad_value", "the padding value for tensors with smaller dimension")
    .Arg("replicate", "whether to replicate the smaller tensor or not");

OPERATOR_SCHEMA(DotProductWithPaddingGradient).NumInputs(3).NumOutputs(2);

class GetDotProductWithPaddingGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    float pad_value = 0;
    bool replicate = false;
    if (ArgumentHelper::HasArgument(Def(), "pad_value")) {
      pad_value = GetArgument(Def(), "pad_value").f();
    }
    if (ArgumentHelper::HasArgument(Def(), "replicate")) {
      replicate = GetArgument(Def(), "replicate").i();
    }

    const auto dot_arg =
        vector<Argument>{MakeArgument<float>("pad_value", pad_value),
                         MakeArgument<bool>("replicate", replicate)};

    return SingleGradientDef(
        "DotProductWithPaddingGradient",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0), GI(1)},
        dot_arg);
  }
};
REGISTER_GRADIENT(DotProductWithPadding, GetDotProductWithPaddingGradient);
}  // namespace caffe2
