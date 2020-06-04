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
    return DoRunWithType<float>();
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(INPUT);
    auto* Y = Output(OUTPUT, X.sizes(), at::dtype<T>());
    CAFFE_ENFORCE_GE(X.dim(), 2, "LayerNorm requires input dim >=2.");
    const int canonical_axis = X.canonical_axis_index(axis_);
    std::vector<int64_t> moments_dims(
        X.sizes().cbegin(), X.sizes().cbegin() + canonical_axis);
    moments_dims.push_back(1);
    auto* mean = Output(1, moments_dims, at::dtype<T>());
    auto* sigma = Output(2, moments_dims, at::dtype<T>());
    const int M = X.size_to_dim(canonical_axis);
    const int N = X.size_from_dim(canonical_axis);
    Y->ResizeLike(X);
    scale_.Resize(M);
    bias_.Resize(M);
    const T* X_data = X.template data<T>();
    T* Y_data = Y->template mutable_data<T>();
    T* mean_data = mean->template mutable_data<T>();
    T* sigma_data = sigma->template mutable_data<T>();
    T* scale_data = scale_.template mutable_data<T>();
    T* bias_data = bias_.template mutable_data<T>();

    std::vector<float> X_rounded(X.numel());
    fbgemm::RoundToFloat16(
        X_data,
        X_rounded.data(),
        X.numel(),
        FLAGS_caffe2_fbgemm_fake_fp16_clamp,
        false /*USE_ACC_FP16*/);
    X_data = X_rounded.data();

    const std::array<int, 2> X_dims = {M, N};
    const std::array<int, 2> Y_dims = {M, 1};
    math::Moments<T, CPUContext>(
        2,
        X_dims.data(),
        Y_dims.data(),
        X_data,
        mean_data,
        sigma_data,
        &context_);

    ComputeSigmaAndFusedParams<T>(
        M, epsilon_, mean_data, sigma_data, sigma_data, scale_data, bias_data);

    const T* gamma_data = nullptr;
    const T* beta_data = nullptr;
    if (elementwise_affine_) {
      CAFFE_ENFORCE_EQ(InputSize(), 3);
      const auto& gamma = Input(1);
      const auto& beta = Input(2);
      CAFFE_ENFORCE_EQ(gamma.numel(), N);
      CAFFE_ENFORCE_EQ(beta.numel(), N);
      gamma_data = gamma.template data<T>();
      beta_data = beta.template data<T>();
    }
    LayerNormForward<T>(
        M, N, X_data, scale_data, bias_data, gamma_data, beta_data, Y_data);

    return true;
  }

 private:
  template <typename T>
  void ComputeSigmaAndFusedParams(
      const int N,
      const float eps,
      const T* mean,
      const T* var,
      T* stddev,
      T* scale,
      T* bias);

  template <typename T>
  void LayerNormForward(
      const int M,
      const int N,
      const T* X,
      const T* scale,
      const T* bias,
      const T* gamma,
      const T* beta,
      T* Y);

  const int axis_;
  const float epsilon_;
  const bool elementwise_affine_;

  Tensor scale_{Context::GetDeviceType()};
  Tensor bias_{Context::GetDeviceType()};

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT, MEAN, STD);
};

} // namespace caffe2
