#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <string>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

#include <fbgemm/FbgemmConvert.h>
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"
#include "fp16_fma.h"

C10_DECLARE_bool(caffe2_fbgemm_fake_fp16_clamp);

namespace caffe2 {


class LayerNormUtils {
  public:
  static void calcY(
      const int M,
      const int N,
      const float* X,
      const float* mean,
      const float* std,
      const float* gamma,
      const float* beta,
      float* Y);

  static void calcMeanStd(
      const int M,
      const int N,
      const float eps,
      const float* X,
      float* mean,
      float* std);

  static float ReducedAdd(const std::vector<float>& vec);
};

template <bool quantizeOutput=false>
class LayerNormFakeFp16Op final : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit LayerNormFakeFp16Op(Args&&... args)
      : Operator<CPUContext>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int, "axis", axis_, 1),
        OP_SINGLE_ARG(float, "epsilon", epsilon_, 1e-5f),
        OP_SINGLE_ARG(bool, "elementwise_affine", elementwise_affine_, false) {}
  ~LayerNormFakeFp16Op() noexcept override {}

  bool RunOnDevice() override {
    return DoRunWithType();
  }

  bool DoRunWithType() {
    const auto& X = Input(INPUT);
    vector <float> Y_fp16;

    Tensor *Y;
    if (!quantizeOutput) {
      Y = Output(OUTPUT, X.sizes(), at::dtype<float>());
    } else {
      Y_fp16.resize(X.numel());
    }
    CAFFE_ENFORCE_GE(X.dim(), 2, "LayerNorm requires input dim >=2.");
    const int canonical_axis = X.canonical_axis_index(axis_);
    std::vector<int64_t> moments_dims(
        X.sizes().cbegin(), X.sizes().cbegin() + canonical_axis);
    moments_dims.push_back(1);
    auto* mean = Output(MEAN, moments_dims, at::dtype<float>());
    auto* sigma = Output(STD, moments_dims, at::dtype<float>());
    const int M = X.size_to_dim(canonical_axis);
    const int N = X.size_from_dim(canonical_axis);

    if (!quantizeOutput) {
      Y->ResizeLike(X);
    }

    const float* X_data = X.template data<float>();

    float *Y_data;
    if (!quantizeOutput) {
      Y_data = Y->template mutable_data<float>();
    } else {
      Y_data = Y_fp16.data();
    }

    float* mean_data = mean->template mutable_data<float>();
    float* sigma_data = sigma->template mutable_data<float>();

    std::vector<float> X_rounded(X.numel());
    fbgemm::RoundToFloat16(
        X_data,
        X_rounded.data(),
        X.numel(),
        FLAGS_caffe2_fbgemm_fake_fp16_clamp,
        false /*USE_ACC_FP16*/);
    X_data = X_rounded.data();

    // Mean and Standard Deviation computation for the input data
    LayerNormUtils::calcMeanStd(M, N, epsilon_, X_data, mean_data, sigma_data);

    const float* gamma_data = nullptr;
    const float* beta_data = nullptr;

    // Layer Normalized Output computation
    LayerNormUtils::calcY(
        M, N, X_data, mean_data, sigma_data, gamma_data, beta_data, Y_data);

    if (InputSize() == 3) {
      // handle scale and bias via fp16_fma
      std::vector<float> scale_data(N);
      std::vector<float> bias_data(N);
      fbgemm::RoundToFloat16(
          Input(1).template data<float>(),
          scale_data.data(),
          N,
          FLAGS_caffe2_fbgemm_fake_fp16_clamp,
          false /*USE_ACC_FP16*/);
      fbgemm::RoundToFloat16(
          Input(2).template data<float>(),
          bias_data.data(),
          N,
          FLAGS_caffe2_fbgemm_fake_fp16_clamp,
          false /*USE_ACC_FP16*/);

      for (const auto i : c10::irange(M)) {
        // fma_fp16(A, B, Out) -> Out = A * B + Out
        std::vector<float> out(N);
        std::memcpy(out.data(), bias_data.data(), sizeof(float) * N);
        fake_fp16::fma_fp16(N, Y_data + i * N, scale_data.data(), out.data());
        std::memcpy(Y_data + i * N, out.data(), sizeof(float) * N);
      }
    }

    // Quantize
    // We should be using the same quantization fucntion from int8quantize,
    // but we need to adjust for int8 vs uint8 bias. A simple shift of the output is not enough
    // because this causes problems when rounding inside the fma.
    // TODO: figure out how to commonize this with int8 quantize
    if (quantizeOutput) {
      auto* Y_int8 = Outputs()[0]->template GetMutable<int8::Int8TensorCPU>();
      Y_int8->t.ResizeLike(X);

      int32_t Y_offset =
          this->template GetSingleArgument<int>("Y_zero_point", 0);
      auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);

      float inv_scale = 1.0f / Y_scale;
      fbgemm::RoundToFloat16(
        &inv_scale, &inv_scale, 1, false /* no clamping */);

      Y_int8->scale = Y_scale;
      Y_int8->zero_point = Y_offset;

      int Nout = X.numel();

      std::vector<float> inv_scalev(Nout, inv_scale);
      std::vector<float> offsetv(Nout, Y_offset);
      uint8_t* Y_uint8_data = Y_int8->t.template mutable_data<uint8_t>();

      fake_fp16::fma_fp16(Nout, Y_fp16.data(), inv_scalev.data(), offsetv.data());

      const int32_t qmin = std::numeric_limits<uint8_t>::min();
      const int32_t qmax = std::numeric_limits<uint8_t>::max();

      for (const auto i : c10::irange(Nout)) {
        float halfRes = offsetv[i];
        halfRes = round(halfRes);
        if (std::isinf(halfRes)) {
          if (halfRes > 0) {
            halfRes = qmax;
          } else {
            halfRes = qmin;
          }
        }
        if (halfRes > qmax) {
          halfRes = qmax;
        }
        if (halfRes < qmin) {
          halfRes = qmin;
        }
        Y_uint8_data[i] = static_cast<uint8_t>(halfRes);
      }
    }

    return true;
  }

 private:
  const int axis_;
  const float epsilon_;
  // LayerNorm FP16 FakeLowP Op applies the scales and biases (or gamma and beta)
  // whenever those inputs are provided else it will omit them.
  // We are keeping elementwise_affine to keep it consistent with LayerNorm FP32 Op.
  const bool elementwise_affine_;

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT, MEAN, STD);
};

} // namespace caffe2
