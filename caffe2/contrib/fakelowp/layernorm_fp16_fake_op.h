#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <string>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

//#include "caffe2/fb/fbgemm/fbgemm_fp16/include/fbgemm/FbgemmFloat16.h"
//#include <fbgemm/FbgemmFloat16.h>
#include <fbgemm/FbgemmConvert.h>
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"
#include "fp16_fma.h"

C10_DECLARE_bool(caffe2_fbgemm_fake_fp16_clamp);

namespace caffe2 {

template <class Context>
class LayerNormFakeFp16Op final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

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
    auto* Y = Output(OUTPUT, X.sizes(), at::dtype<float>());
    CAFFE_ENFORCE_GE(X.dim(), 2, "LayerNorm requires input dim >=2.");
    const int canonical_axis = X.canonical_axis_index(axis_);
    std::vector<int64_t> moments_dims(
        X.sizes().cbegin(), X.sizes().cbegin() + canonical_axis);
    moments_dims.push_back(1);
    auto* mean = Output(MEAN, moments_dims, at::dtype<float>());
    auto* sigma = Output(STD, moments_dims, at::dtype<float>());
    const int M = X.size_to_dim(canonical_axis);
    const int N = X.size_from_dim(canonical_axis);
    Y->ResizeLike(X);
    const float* X_data = X.template data<float>();
    float* Y_data = Y->template mutable_data<float>();
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
    calcMeanStd(M, N, epsilon_, X_data, mean_data, sigma_data);

    const float* gamma_data = nullptr;
    const float* beta_data = nullptr;
    std::vector<float> gamma_rounded(N);
    std::vector<float> beta_rounded(N);

    if (elementwise_affine_) {
      CAFFE_ENFORCE_EQ(InputSize(), 3);
      const auto& gamma = Input(1);
      const auto& beta = Input(2);
      CAFFE_ENFORCE_EQ(gamma.numel(), N);
      CAFFE_ENFORCE_EQ(beta.numel(), N);

      gamma_data = gamma.template data<float>();
      fbgemm::RoundToFloat16(
          gamma_data,
          gamma_rounded.data(),
          N,
          FLAGS_caffe2_fbgemm_fake_fp16_clamp,
          false /*USE_ACC_FP16*/);
      gamma_data = gamma_rounded.data();

      beta_data = beta.template data<float>();
      fbgemm::RoundToFloat16(
          beta_data,
          beta_rounded.data(),
          N,
          FLAGS_caffe2_fbgemm_fake_fp16_clamp,
          false /*USE_ACC_FP16*/);
      beta_data = beta_rounded.data();
    }

    // Layer Normalized Output computation
    calcY(
        M, N, X_data, mean_data, sigma_data, gamma_data, beta_data, Y_data);

    if (InputSize() == 3 && !elementwise_affine_) {
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

      for (int i = 0; i < M; ++i) {
        // fma_fp16(A, B, Out) -> Out = A * B + Out
        std::vector<float> out(N);
        std::memcpy(out.data(), bias_data.data(), sizeof(float) * N);
        fake_fp16::fma_fp16(N, Y_data + i * N, scale_data.data(), out.data());
        std::memcpy(Y_data + i * N, out.data(), sizeof(float) * N);
      }
    }

    return true;
  }

 private:
  void calcY(
      const int M,
      const int N,
      const float* X,
      const float* mean,
      const float* std,
      const float* gamma,
      const float* beta,
      float* Y);

  void calcMeanStd(
      const int M,
      const int N,
      const float eps,
      const float* X,
      float* mean,
      float* std);

  float ReducedAdd(const std::vector<float>& vec);

  const int axis_;
  const float epsilon_;
  const bool elementwise_affine_;

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT, MEAN, STD);
};

} // namespace caffe2
