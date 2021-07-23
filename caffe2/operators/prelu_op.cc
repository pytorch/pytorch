#include "caffe2/operators/prelu_op.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

#include "caffe2/core/types.h"
#include "caffe2/utils/cpu_neon.h"

namespace caffe2 {

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
namespace {

void runNeonPrelu(float* out, const float* in, int size, float w) {
  float32x4_t vZero = vdupq_n_f32(0.0f);
  float32x4_t vW = vdupq_n_f32(w);

  constexpr int kVecSizeInFloat = sizeof(float32x4_t) / sizeof(float);

  if (size < kVecSizeInFloat) {
    for (int i = 0; i < size; ++i) {
      float v = in[i];
      out[i] = v > 0 ? v : v * w;
    }

    return;
  }

  // We want to load aligned from the input, but assume the output is unaligned
  int prologue = kVecSizeInFloat -
      // remainder in floats
      (((uintptr_t)in) % (sizeof(float32x4_t))) / sizeof(float);

  int i = 0;

  // Prologue loop
  for (; i < prologue; ++i) {
    float v = in[i];
    out[i] = v > 0 ? v : v * w;
  }

  // The loop is manually unrolled by 6; seems to be the limit for
  // armv7 to avoid register spills
  constexpr int kUnroll = 6;
  constexpr int kFloatsPerLoop = kUnroll * kVecSizeInFloat;

  int remainder = size - prologue;
  int vectorizable = prologue + (remainder / kFloatsPerLoop) * kFloatsPerLoop;

  for (; i < vectorizable; i += kFloatsPerLoop) {
    float32x4_t v0 = vld1q_f32_aligned(in + i + 0);
    float32x4_t v1 = vld1q_f32_aligned(in + i + 4);
    float32x4_t v2 = vld1q_f32_aligned(in + i + 8);
    float32x4_t v3 = vld1q_f32_aligned(in + i + 12);
    float32x4_t v4 = vld1q_f32_aligned(in + i + 16);
    float32x4_t v5 = vld1q_f32_aligned(in + i + 20);

    uint32x4_t gz0 = vcgtq_f32(v0, vZero);
    uint32x4_t gz1 = vcgtq_f32(v1, vZero);
    uint32x4_t gz2 = vcgtq_f32(v2, vZero);
    uint32x4_t gz3 = vcgtq_f32(v3, vZero);
    uint32x4_t gz4 = vcgtq_f32(v4, vZero);
    uint32x4_t gz5 = vcgtq_f32(v5, vZero);

    float32x4_t v0neg = vmulq_f32(v0, vW);
    float32x4_t v1neg = vmulq_f32(v1, vW);
    float32x4_t v2neg = vmulq_f32(v2, vW);
    float32x4_t v3neg = vmulq_f32(v3, vW);
    float32x4_t v4neg = vmulq_f32(v4, vW);
    float32x4_t v5neg = vmulq_f32(v5, vW);

    // v0 > 0 ? v0 : v0 * w
    v0 = vbslq_f32(gz0, v0, v0neg);
    v1 = vbslq_f32(gz1, v1, v1neg);
    v2 = vbslq_f32(gz2, v2, v2neg);
    v3 = vbslq_f32(gz3, v3, v3neg);
    v4 = vbslq_f32(gz4, v4, v4neg);
    v5 = vbslq_f32(gz5, v5, v5neg);

    vst1q_f32(out + i + 0, v0);
    vst1q_f32(out + i + 4, v1);
    vst1q_f32(out + i + 8, v2);
    vst1q_f32(out + i + 12, v3);
    vst1q_f32(out + i + 16, v4);
    vst1q_f32(out + i + 20, v5);
  }

  for (; i < size; ++i) {
    float v = in[i];
    out[i] = v > 0 ? v : v * w;
  }
}

} // namespace
#endif // defined(__ARM_NEON__) || defined(__ARM_NEON)

template <>
bool PReluOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& W = Input(1);

  auto* Y = Output(0, X.sizes(), at::dtype<float>());
  const auto* Xdata = X.template data<float>();
  const auto* Wdata = W.template data<float>();
  auto* Ydata = Y->template mutable_data<float>();

  const auto C = order_ == StorageOrder::NCHW ? X.size(1) : X.size(X.dim() - 1);
  const auto C_shared = (W.numel() == 1);

  if (!C_shared) {
    CAFFE_ENFORCE_EQ(C, W.numel());
  }

  if (C_shared) {
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
    // The function is completely pointwise
    runNeonPrelu(Ydata, Xdata, X.size(), Wdata[0]);
#else
    ConstEigenVectorMap<float> Xvec(Xdata, X.numel());
    EigenVectorMap<float> Yvec(Ydata, Y->numel());
    Yvec = Xvec.cwiseMax(0.f) + Xvec.cwiseMin(0.f) * Wdata[0];
#endif // defined(__ARM_NEON__) || defined(__ARM_NEON)
    return true;
  }

  // non-shared case.
  switch (order_) {
    case StorageOrder::NCHW: {
      const auto N = X.size(0);
      const auto dim = X.size_from_dim(2);

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
      // Pointwise for each channel
      for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
          runNeonPrelu(
              Ydata + (n * C + c) * dim,
              Xdata + (n * C + c) * dim,
              dim,
              Wdata[c]);
        }
      }
#else
      int nc = 0;
      for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
          ConstEigenVectorMap<float> Xvec(Xdata + nc * dim, dim);
          EigenVectorMap<float>(Ydata + nc * dim, dim) =
              Xvec.cwiseMax(0.f) + Xvec.cwiseMin(0.f) * Wdata[c];
          nc++;
        }
      }
#endif
      break;
    }
    case StorageOrder::NHWC: {
      // Lay out matrix as (NHW, C) and multiply by C
      const auto NHW = X.numel() / C;
      ConstEigenArrayMap<float> Xmat(Xdata, C, NHW);
      ConstEigenVectorArrayMap<float> Wvec(Wdata, C);
      EigenArrayMap<float> Ymat(Ydata, C, NHW);
      Ymat = (Xmat > 0).select(Xmat, Xmat.colwise() * Wvec);
      break;
    }
    default:
      CAFFE_THROW("Unknown storage order: ", order_);
  }
  return true;
}

template <>
bool PReluGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto& X = Input(2);
  auto& W = Input(3);

  CAFFE_ENFORCE(&Y != &X, "Cannot backpropagate through an in-place PReLU");

  DCHECK_EQ(dY.numel(), Y.numel());
  auto* dX = Output(0, Y.sizes(), at::dtype<float>());
  auto* dW = Output(1, W.sizes(), at::dtype<float>());

  const auto C = order_ == StorageOrder::NCHW ? X.size(1) : X.size(X.dim() - 1);
  const auto C_shared = (W.numel() == 1);

  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  const float* Xdata = X.data<float>();
  const float* Wdata = W.data<float>();
  float* dXdata = dX->template mutable_data<float>();
  float* dWdata = dW->template mutable_data<float>();

  // non-shared case.
  switch (order_) {
    case StorageOrder::NCHW: {
      const auto dim = X.size_from_dim(2);
      const auto div_factor = C_shared ? C : 1;
      for (auto c = 0; c < W.numel(); ++c) {
        dWdata[c] = 0;
      }

      for (int i = 0; i < Y.numel(); ++i) {
        if (Xdata[i] <= 0) {
          // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
          int c = (i / dim) % C / div_factor;
          dWdata[c] += dYdata[i] * Xdata[i];
        }
      }

      for (int i = 0; i < Y.numel(); ++i) {
        if (Xdata[i] > 0) {
          dXdata[i] = dYdata[i];
        } else {
          // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
          int c = (i / dim) % C / div_factor;
          dXdata[i] = Wdata[c] * dYdata[i];
        }
      }
      break;
    }
    case StorageOrder::NHWC: {
      const auto NHW = X.numel() / C;
      ConstEigenVectorArrayMap<float> Wvec(Wdata, W.numel());
      EigenVectorArrayMap<float> dWvec(dWdata, dW->numel());

      ConstEigenArrayMap<float> Ymat(Ydata, C, NHW);
      ConstEigenArrayMap<float> dYmat(dYdata, C, NHW);
      ConstEigenArrayMap<float> Xmat(Xdata, C, NHW);
      EigenArrayMap<float> dXmat(dXdata, C, NHW);

      if (C_shared) {
        dXmat = (Xmat > 0).select(dYmat, dYmat * Wdata[0]);
        dWdata[0] =
            (Xmat > 0)
                .select(
                    Xmat.cwiseMin(0.0f), // zero gradients on the 'if' path.
                    dYmat * Xmat)
                .sum();
      } else {
        dXmat = (Xmat > 0).select(dYmat, dYmat.colwise() * Wvec);
        dWvec = (Xmat > 0)
                    .select(
                        Xmat.cwiseMin(0.0f), // zero gradients on the 'if' path.
                        dYmat * Xmat)
                    .rowwise()
                    .sum();
      }
      break;
    }
    default:
      CAFFE_THROW("Unknown storage order: ", order_);
  }

  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(PRelu, PReluOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_GRADIENT_OPERATOR(
    PReluGradient,
    PReluGradientOp<float, CPUContext>);

// Input: X, Slope, output: Y
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(PRelu)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShapeOfInput(0)
    .SetDoc(R"DOC(

The *PRelu* op takes input data tensor $X$, an input slope tensor $slope$, and produces one output tensor $Y$ of the same shape as $X.$ The op performs the element wise *PRelu* operation, defined as

$$y=prelu(x) =\begin{cases}slope * x & x < 0\\x & otherwise\end{cases}$$

Note, is slope is size 1, the value is shared across the channels, otherwise $X$ and $slope$ must be the same shape. See [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852) for more information.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/prelu_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/prelu_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "PRelu",
    ["X","Slope"],
    ["Y"],
)

workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32))
print("X:\n", workspace.FetchBlob("X"), "\n")

workspace.FeedBlob("Slope", np.array([0.1]).astype(np.float32))
print("Slope:\n", workspace.FetchBlob("Slope"), "\n")

workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))

```

**Result**

```

X:
 [[ 0.3957382  -0.19725518 -0.26991343]
 [ 1.5513182  -0.27427664 -0.14584002]
 [-0.4121164   0.9292345   0.96426094]]

Slope:
 [0.1]

Y:
 [[ 0.3957382  -0.01972552 -0.02699134]
 [ 1.5513182  -0.02742766 -0.014584  ]
 [-0.04121164  0.9292345   0.96426094]]

```

</details>


)DOC")
    .Input(0, "X", "Input tensor of data to be operated on.")
    .Input(
        1,
        "Slope",
        "1D input slope tensor. If `Slope` is of size 1, the value is shared across different channels")
    .Output(0, "Y", "Output tensor, with same shape as $X$.")
    .InheritOnnxSchema();

// Input: Y, dY, output: dX
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
GRADIENT_OPERATOR_SCHEMA(PReluGradient)
    .NumInputs(4)
    .NumOutputs(2)
    .SetDoc(R"DOC(

PReluGradient takes both Y and dY and uses this to update dX and dW according
to the chain rule and derivatives of the rectified linear function.

)DOC")
    .IdenticalTypeAndShapeOfMultipleInputs({2, 3});

class GetPReluGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        vector<string>{O(0), GO(0), I(0), I(1)},
        vector<string>{GI(0), GI(1)});
  }
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(PRelu, GetPReluGradient);

} // namespace caffe2
