#include "caffe2/operators/prelu_op.h"
#include "caffe2/utils/math.h"

#include "caffe2/core/types.h"
#include "caffe2/utils/cpu_neon.h"

namespace caffe2 {

#ifdef __ARM_NEON__
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
  int prologue =
    kVecSizeInFloat -
    // remainder in floats
    (((uintptr_t) in) % (sizeof(float32x4_t))) / sizeof(float);

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

}
#endif // __ARM_NEON__

template <>
bool PReluOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& W = Input(1);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  const auto* Xdata = X.template data<float>();
  const auto* Wdata = W.template data<float>();
  auto* Ydata = Y->template mutable_data<float>();

  const auto C = order_ == StorageOrder::NCHW ? X.dim(1) : X.dim(X.ndim() - 1);
  const auto C_shared = (W.size() == 1);

  if (!C_shared) {
    CAFFE_ENFORCE_EQ(C, W.size());
  }

  if (C_shared) {
#ifdef __ARM_NEON__
    // The function is completely pointwise
    runNeonPrelu(Ydata, Xdata, X.size(), Wdata[0]);
#else
    ConstEigenVectorMap<float> Xvec(Xdata, X.size());
    EigenVectorMap<float> Yvec(Ydata, Y->size());
    Yvec = Xvec.cwiseMax(0.f) + Xvec.cwiseMin(0.f) * Wdata[0];
#endif // __ARM_NEON__
    return true;
  }

  // non-shared case.
  switch (order_) {
    case StorageOrder::NCHW: {
      const auto N = X.dim(0);
      const auto dim = X.size_from_dim(2);

#ifdef __ARM_NEON__
      // Pointwise for each channel
      for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
          runNeonPrelu(Ydata + (n * C + c) * dim,
                       Xdata + (n * C + c) * dim,
                       dim, Wdata[c]);
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
      const auto NHW = X.size() / C;
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
  auto* dX = Output(0);
  auto* dW = Output(1);

  DCHECK_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);
  dW->ResizeLike(W);

  const auto C = order_ == StorageOrder::NCHW ? X.dim(1) : X.dim(X.ndim() - 1);
  const auto C_shared = (W.size() == 1);

  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  const float* Xdata = X.data<float>();
  const float* Wdata = W.data<float>();
  float* dXdata = dX->mutable_data<float>();
  float* dWdata = dW->mutable_data<float>();

  // non-shared case.
  switch (order_) {
    case StorageOrder::NCHW: {
      const auto dim = X.size_from_dim(2);
      const auto div_factor = C_shared ? C : 1;
      for (auto c = 0; c < W.size(); ++c) {
        dWdata[c] = 0;
      }

      for (int i = 0; i < Y.size(); ++i) {
        if (Xdata[i] <= 0) {
          int c = (i / dim) % C / div_factor;
          dWdata[c] += dYdata[i] * Xdata[i];
        }
      }

      for (int i = 0; i < Y.size(); ++i) {
        if (Xdata[i] > 0) {
          dXdata[i] = dYdata[i];
        } else {
          int c = (i / dim) % C / div_factor;
          dXdata[i] = Wdata[c] * dYdata[i];
        }
      }
      break;
    }
    case StorageOrder::NHWC: {
      const auto NHW = X.size() / C;
      ConstEigenVectorArrayMap<float> Wvec(Wdata, W.size());
      EigenVectorArrayMap<float> dWvec(dWdata, dW->size());

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

REGISTER_CPU_OPERATOR(PRelu, PReluOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(PReluGradient, PReluGradientOp<float, CPUContext>);

// Input: X, Slope, output: Y
OPERATOR_SCHEMA(PRelu)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShapeOfInput(0)
    .SetDoc(R"DOC(

PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.

)DOC")
    .Input(0, "X", "1D input tensor")
    .Input(
        1,
        "Slope",
        "1D slope tensor. If `Slope` is of size 1, the value is shared"
        "across different channels")
    .Output(0, "Y", "1D input tensor")
    .InheritOnnxSchema("PRelu");

// Input: Y, dY, output: dX
OPERATOR_SCHEMA(PReluGradient).NumInputs(4).NumOutputs(2).SetDoc(R"DOC(

PReluGradient takes both Y and dY and uses this to update dX and dW according
to the chain rule and derivatives of the rectified linear function.

)DOC");

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
REGISTER_GRADIENT(PRelu, GetPReluGradient);

} // namespace caffe2
