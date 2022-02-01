#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <string>
#include <vector>

#include <fbgemm/FbgemmConvert.h>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"
#include "fp16_fma.h"

C10_DECLARE_bool(caffe2_fbgemm_fake_fp16_clamp);

namespace caffe2 {

class SpatialBNFakeLoweredFp16Op : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);

  template <class... Args>
  explicit SpatialBNFakeLoweredFp16Op(Args&&... args)
      : Operator<CPUContext>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(bool, OpSchema::Arg_IsTest, is_test_, false),
        OP_SINGLE_ARG(double, "epsilon", epsilon_, 1e-5),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<std::string>("order", "NCHW"))),
        OP_SINGLE_ARG(int, "num_batches", num_batches_, 1) {
    // TODO: only support NCHW for now
    CAFFE_ENFORCE_EQ(order_, StorageOrder::NCHW);
    CAFFE_ENFORCE(
        (is_test_ && OutputSize() == 1) || (!is_test_ && OutputSize() == 5));
    CAFFE_ENFORCE_GT(epsilon_, 0);
  }

   ~SpatialBNFakeLoweredFp16Op() override = default;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(INPUT);
    const auto& scale = Input(SCALE);
    const auto& bias = Input(BIAS);

    const int ndim = X.dim();
    CAFFE_ENFORCE_GE(ndim, 2);
    const int N = X.dim32(0);
    const int C =
        (order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1));
    const std::vector<int> X_dims(X.sizes().cbegin(), X.sizes().cend());
    const int HxW =
        std::accumulate(
            X_dims.cbegin() + 1, X_dims.cend(), 1, std::multiplies<int>()) /
        C;
    CAFFE_ENFORCE_EQ(scale.numel(), C);
    CAFFE_ENFORCE_EQ(bias.numel(), C);

    auto* Y = Output(OUTPUT, X.sizes(), at::dtype<T>());
    T* Y_data = Y->template mutable_data<T>();
    ReinitializeTensor(
        &alpha_, {C}, at::dtype<T>().device(CPUContext::GetDeviceType()));
    T* alpha_data = alpha_.template mutable_data<T>();

    // We only support this case at the moment
    CAFFE_ENFORCE(is_test_);

    std::vector<float> X_fp16(X.numel());

    fbgemm::RoundToFloat16(
        X.template data<T>(),
        X_fp16.data(),
        N * C * HxW,
        FLAGS_caffe2_fbgemm_fake_fp16_clamp);

    if (N == 0) {
      return true;
    }
    const auto& mean = Input(EST_MEAN);
    const auto& var = Input(EST_VAR);
    CAFFE_ENFORCE_EQ(mean.numel(), C);
    CAFFE_ENFORCE_EQ(var.numel(), C);
    std::vector<float> mean_fp16(C), var_fp16(C);
    std::vector<float> scale_fp16(C), bias_fp16(C);

    fbgemm::RoundToFloat16(
        scale.template data<T>(),
        scale_fp16.data(),
        C,
        FLAGS_caffe2_fbgemm_fake_fp16_clamp);
    fbgemm::RoundToFloat16(
        bias.template data<T>(),
        bias_fp16.data(),
        C,
        FLAGS_caffe2_fbgemm_fake_fp16_clamp);
    fbgemm::RoundToFloat16(
        mean.template data<T>(),
        mean_fp16.data(),
        C,
        FLAGS_caffe2_fbgemm_fake_fp16_clamp);
    fbgemm::RoundToFloat16(
        var.template data<T>(),
        var_fp16.data(),
        C,
        FLAGS_caffe2_fbgemm_fake_fp16_clamp);

    EigenVectorArrayMap<T> alpha_arr(alpha_data, C);
    std::vector<float> tmp(C);
    EigenVectorArrayMap<T> tmp_arr(tmp.data(), C);

    auto epsilon = static_cast<T>(epsilon_);
    fbgemm::RoundToFloat16(
        &epsilon, &epsilon, 1, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
    tmp_arr = (ConstEigenVectorArrayMap<T>(var_fp16.data(), C) + epsilon);
    fbgemm::RoundToFloat16(
        tmp.data(), tmp.data(), C, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
    tmp_arr = tmp_arr.pow(0.5);
    fbgemm::RoundToFloat16(
        tmp.data(), tmp.data(), C, FLAGS_caffe2_fbgemm_fake_fp16_clamp);

    alpha_arr = ConstEigenVectorArrayMap<T>(scale_fp16.data(), C) / tmp_arr;
    fbgemm::RoundToFloat16(
        alpha_data, alpha_data, C, FLAGS_caffe2_fbgemm_fake_fp16_clamp);

    AffineChannel_NCHW(
        N,
        C,
        HxW,
        X_fp16.data(),
        alpha_data,
        bias_fp16.data(),
        mean_fp16.data(),
        Y_data);

    fbgemm::RoundToFloat16(
        Y_data, Y_data, N * HxW * C, FLAGS_caffe2_fbgemm_fake_fp16_clamp);

    return true;
  }

 protected:
  void AffineChannel_NCHW(
      const int N,
      const int C,
      const int HxW,
      const float* X,
      const float* scale,
      const float* bias,
      const float* mean,
      float* Y) {
    ConstEigenVectorArrayMap<float> scale_arr(scale, C);
    ConstEigenVectorArrayMap<float> bias_arr(bias, C);
    ConstEigenVectorArrayMap<float> mean_arr(mean, C);
    const int stride = C * HxW;
    const float* X_ptr = X;
    float* Y_ptr = Y;
    for (const auto i : c10::irange(N)) {
      EigenArrayMap<float>(Y_ptr, HxW, C) =
          ConstEigenArrayMap<float>(X_ptr, HxW, C).rowwise() -
          mean_arr.transpose();
      fbgemm::RoundToFloat16(
          Y_ptr, Y_ptr, HxW * C, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
      EigenArrayMap<float>(Y_ptr, HxW, C).rowwise() *= scale_arr.transpose();
      fbgemm::RoundToFloat16(
          Y_ptr, Y_ptr, HxW * C, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
      EigenArrayMap<float>(Y_ptr, HxW, C).rowwise() += bias_arr.transpose();

      X_ptr += stride;
      Y_ptr += stride;
    }
    fbgemm::RoundToFloat16(
        Y, Y, N * HxW * C, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
  }

  const bool is_test_;
  double epsilon_;
  const StorageOrder order_;
  const int num_batches_;

  Tensor alpha_;

  INPUT_TAGS(
      INPUT,
      SCALE,
      BIAS,
      EST_MEAN,
      EST_VAR,
      BATCH_MEAN_SUM,
      BATCH_VAR_SUM);
  OUTPUT_TAGS(OUTPUT, RUNNING_MEAN, RUNNING_VAR, SAVED_MEAN, SAVED_INV_STD);
};

// Emulation of the NNPI SpatialBN kernel
class SpatialBNFakeFp16Op : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);

  template <class... Args>
  explicit SpatialBNFakeFp16Op(Args&&... args)
      : Operator<CPUContext>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(bool, OpSchema::Arg_IsTest, is_test_, false),
        OP_SINGLE_ARG(float, "epsilon", epsilon_, 1e-5),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<std::string>("order", "NCHW"))),
        OP_SINGLE_ARG(int, "num_batches", num_batches_, 1) {
    // TODO: only support NCHW for now
    CAFFE_ENFORCE_EQ(order_, StorageOrder::NCHW);
    // We only support this case at the moment
    CAFFE_ENFORCE(is_test_);
    CAFFE_ENFORCE_GT(epsilon_, 0);
  }

   ~SpatialBNFakeFp16Op() override = default;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    LOG(INFO) << "Running with " << sizeof(T);
    const auto& X = Input(INPUT);
    const auto& scale = Input(SCALE);
    const auto& bias = Input(BIAS);

    const int ndim = X.dim();
    CAFFE_ENFORCE_GE(ndim, 2);
    const int N = X.dim32(0);
    const int C =
        (order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1));
    const std::vector<int> X_dims(X.sizes().cbegin(), X.sizes().cend());
    const int HxW =
        std::accumulate(
            X_dims.cbegin() + 1, X_dims.cend(), 1, std::multiplies<int>()) /
        C;
    CAFFE_ENFORCE_EQ(scale.numel(), C);
    CAFFE_ENFORCE_EQ(bias.numel(), C);

    auto* Y = Output(OUTPUT, X.sizes(), at::dtype<T>());
    T* Y_data = Y->template mutable_data<T>();
    ReinitializeTensor(
        &alpha_, {C}, at::dtype<T>().device(CPUContext::GetDeviceType()));
    ReinitializeTensor(
        &beta_, {C}, at::dtype<T>().device(CPUContext::GetDeviceType()));
    T* alpha_data = alpha_.template mutable_data<T>();
    T* beta_data = beta_.template mutable_data<T>();

    std::vector<float> X_fp16(X.numel());

    fbgemm::RoundToFloat16(
        X.template data<T>(),
        X_fp16.data(),
        N * C * HxW,
        FLAGS_caffe2_fbgemm_fake_fp16_clamp);

    const auto& mean = Input(EST_MEAN);
    const auto& var = Input(EST_VAR);
    CAFFE_ENFORCE_EQ(mean.numel(), C);
    CAFFE_ENFORCE_EQ(var.numel(), C);
    std::vector<float> mean_fp16(C), var_fp16(C);
    std::vector<float> scale_fp16(C), bias_fp16(C);

    fbgemm::RoundToFloat16(
        scale.template data<T>(),
        scale_fp16.data(),
        C,
        FLAGS_caffe2_fbgemm_fake_fp16_clamp);
    fbgemm::RoundToFloat16(
        bias.template data<T>(),
        bias_fp16.data(),
        C,
        FLAGS_caffe2_fbgemm_fake_fp16_clamp);
    fbgemm::RoundToFloat16(
        mean.template data<T>(),
        mean_fp16.data(),
        C,
        FLAGS_caffe2_fbgemm_fake_fp16_clamp);
    fbgemm::RoundToFloat16(
        var.template data<T>(),
        var_fp16.data(),
        C,
        FLAGS_caffe2_fbgemm_fake_fp16_clamp);

    // This part is run on the CPU/x86 core
    ComputeFusedParam<T>(
        C,
        scale_fp16.data(),
        bias_fp16.data(),
        mean_fp16.data(),
        var_fp16.data(),
        alpha_data,
        beta_data);
    AffineChannel_NCHW(N, C, HxW, X_fp16.data(), alpha_data, beta_data, Y_data);

    fbgemm::RoundToFloat16(
        Y_data, Y_data, N * HxW * C, FLAGS_caffe2_fbgemm_fake_fp16_clamp);

    return true;
  }

 protected:
  template <typename T>
  void ComputeFusedParam(
      const int C,
      const T* scale,
      const T* bias,
      const T* mean,
      const T* var,
      T* alpha,
      T* beta) {
    // alpha = scale / sqrt(var + epsilon)
    // beta = bias - alpha * mean
    EigenVectorArrayMap<T> alpha_arr(alpha, C);
    EigenVectorArrayMap<T> beta_arr(beta, C);

    std::vector<T> tmp(C, 0.0);
    EigenVectorArrayMap<T> tmp_arr(tmp.data(), C);
    tmp_arr = ConstEigenVectorArrayMap<T>(var, C) + static_cast<T>(epsilon_);

    // sqrt using intrinsics
    int i = 0;
    constexpr int blockSize = 8;
    for (i = 0; i + blockSize <= C; i += blockSize) {
      __m256 t = _mm256_loadu_ps(&tmp[i]);
      _mm256_storeu_ps(&tmp[i], _mm256_sqrt_ps(t));
    }
    for (; i < C; i++) {
      tmp[i] = sqrt(tmp[i]);
    }

    alpha_arr = ConstEigenVectorArrayMap<T>(scale, C) / tmp_arr;
    beta_arr = ConstEigenVectorArrayMap<T>(bias, C) -
        alpha_arr * ConstEigenVectorArrayMap<T>(mean, C);
    fbgemm::RoundToFloat16(
        alpha, alpha, C, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
    fbgemm::RoundToFloat16(beta, beta, C, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
  }

  void AffineChannel_NCHW(
      const int N,
      const int C,
      const int HxW,
      const float* X,
      const float* scale,
      const float* bias,
      float* Y) {
    ConstEigenVectorArrayMap<float> scale_arr(scale, C);
    ConstEigenVectorArrayMap<float> bias_arr(bias, C);
    const int stride = C * HxW;
    const float* X_ptr = X;
    float* Y_ptr = Y;

    // Do Y = X * scale + bias
    for (const auto i : c10::irange(N)) {
      for (const auto j : c10::irange(C)) {
        for (const auto k : c10::irange(HxW)) {
          Y_ptr[HxW * j + k] = bias[j];
        }

        std::vector<float> s2(HxW, scale[j]);
        fake_fp16::fma_fp16(
            HxW, X_ptr + j * HxW, s2.data(), Y_ptr + HxW * j); // b2.data());
      }
      X_ptr += stride;
      Y_ptr += stride;
    }
    fbgemm::RoundToFloat16(
        Y, Y, N * HxW * C, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
  }

  const bool is_test_;
  float epsilon_;
  const StorageOrder order_;
  const int num_batches_;

  Tensor alpha_;
  Tensor beta_;

  INPUT_TAGS(
      INPUT,
      SCALE,
      BIAS,
      EST_MEAN,
      EST_VAR,
      BATCH_MEAN_SUM,
      BATCH_VAR_SUM);
  OUTPUT_TAGS(OUTPUT, RUNNING_MEAN, RUNNING_VAR, SAVED_MEAN, SAVED_INV_STD);
}; // namespace caffe2

} // namespace caffe2
