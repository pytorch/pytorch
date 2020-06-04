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

class LayerNormFakeFp16Op : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);

  template <class... Args>
  explicit LayerNormFakeFp16Op(Args&&... args)
      : Operator<CPUContext>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int, "axis", axis_, 1),
        OP_SINGLE_ARG(float, "epsilon", epsilon_, 1e-5f),
        OP_SINGLE_ARG(bool, "elementwise_affine", elementwise_affine_, false) {}
  ~LayerNormFakeFp16Op() noexcept override {}

  bool RunOnDevice() override {
    return true;
    // return DispatchHelper<InputTypes>::call(this, Input(DATA));
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
    const T* X_data = X.template data<T>();
    T* Y_data = Y->template mutable_data<T>();
    T* mean_data = mean->template mutable_data<T>();
    T* sigma_data = sigma->template mutable_data<T>();

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

    return true;
  }

 private:
  const int axis_;
  const float epsilon_;
  const bool elementwise_affine_;

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT, MEAN, STD);
};

} // namespace caffe2
