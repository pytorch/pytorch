#ifndef CAFFE2_OPERATORS_SPATIAL_BATCH_NORM_OP_H_
#define CAFFE2_OPERATORS_SPATIAL_BATCH_NORM_OP_H_

#include <algorithm>
#include <array>
#include <functional>
#include <string>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class SpatialBNOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit SpatialBNOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(bool, OpSchema::Arg_IsTest, is_test_, false),
        OP_SINGLE_ARG(double, "epsilon", epsilon_, 1e-5),
        OP_SINGLE_ARG(float, "momentum", momentum_, 0.9f),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<std::string>("order", "NCHW"))),
        OP_SINGLE_ARG(int, "num_batches", num_batches_, 1) {
    CAFFE_ENFORCE_NE(
        order_,
        StorageOrder::UNKNOWN,
        "order should be either \"NCHW\" or \"NHWC\".");
    CAFFE_ENFORCE(
        (is_test_ && OutputSize() == 1) || (!is_test_ && OutputSize() == 5));
    CAFFE_ENFORCE_GT(epsilon_, 0);
    CAFFE_ENFORCE_GE(momentum_, 0);
    CAFFE_ENFORCE_LE(momentum_, 1);
  }

  virtual ~SpatialBNOp() = default;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(INPUT);
    const auto& scale = Input(SCALE);
    const auto& bias = Input(BIAS);

    const int ndim = X.dim();
    CAFFE_ENFORCE_GE(ndim, 3);
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
    const T* X_data = X.template data<T>();
    const T* scale_data = scale.template data<T>();
    const T* bias_data = bias.template data<T>();
    T* Y_data = Y->template mutable_data<T>();
    ReinitializeTensor(
        &alpha_, {C}, at::dtype<T>().device(Context::GetDeviceType()));
    ReinitializeTensor(
        &beta_, {C}, at::dtype<T>().device(Context::GetDeviceType()));
    T* alpha_data = alpha_.template mutable_data<T>();
    T* beta_data = beta_.template mutable_data<T>();
    if (is_test_) {
      if (N == 0) {
        return true;
      }
      const auto& mean = Input(EST_MEAN);
      const auto& var = Input(EST_VAR);
      CAFFE_ENFORCE_EQ(mean.numel(), C);
      CAFFE_ENFORCE_EQ(var.numel(), C);
      ComputeFusedParam<T>(
          C,
          scale_data,
          bias_data,
          mean.template data<T>(),
          var.template data<T>(),
          alpha_data,
          beta_data);
    } else {
      auto* saved_mean = Output(SAVED_MEAN, {C}, at::dtype<T>());
      auto* saved_rstd = Output(SAVED_INV_STD, {C}, at::dtype<T>());
      T* saved_mean_data = saved_mean->template mutable_data<T>();
      T* saved_rstd_data = saved_rstd->template mutable_data<T>();

      // Enforce Alias
      CAFFE_ENFORCE(
          IsInputOutputAlias(3, 1), "Input 3 and Output 1 should be alias.");
      CAFFE_ENFORCE(
          IsInputOutputAlias(4, 2), "Input 4 and Output 2 should be alias.");

      Tensor* running_mean = nullptr;
      Tensor* running_var = nullptr;
      const auto& mean = Input(EST_MEAN);
      const auto& var = Input(EST_VAR);
      if (mean.numel() != C) {
        running_mean = Output(RUNNING_MEAN, {C}, at::dtype<T>());
        C10_LOG_EVERY_MS(WARNING, 1000)
            << "[Depreacated] Running mean is not initialized in "
               "SpatialBatchNorm Op";
        math::Set<T, Context>(
            C, T(0), running_mean->template mutable_data<T>(), &context_);
      } else {
        running_mean = Output(RUNNING_MEAN, {C}, at::dtype<T>());
      }
      if (var.numel() != C) {
        running_var = Output(RUNNING_VAR, {C}, at::dtype<T>());
        math::Set<T, Context>(
            C, T(0), running_var->template mutable_data<T>(), &context_);
        C10_LOG_EVERY_MS(WARNING, 1000)
            << "[Deprecated] Running variance is not initialized in "
               "SpatialBatchNorm Op";
      } else {
        running_var = Output(RUNNING_VAR, {C}, at::dtype<T>());
      }

      T* running_mean_data = running_mean->template mutable_data<T>();
      T* running_var_data = running_var->template mutable_data<T>();
      if (N == 0) {
        math::Set<T, Context>(C, T(0), saved_mean_data, &context_);
        math::Set<T, Context>(C, T(0), saved_rstd_data, &context_);
        return true;
      }
      if (num_batches_ > 1) {
        const auto& batch_mean_sum = Input(BATCH_MEAN_SUM);
        const auto& batch_var_sum = Input(BATCH_VAR_SUM);
        CAFFE_ENFORCE_EQ(batch_mean_sum.numel(), C);
        CAFFE_ENFORCE_EQ(batch_var_sum.numel(), C);
        ComputeBatchMoments<T>(
            N,
            C,
            HxW,
            batch_mean_sum.template data<T>(),
            batch_var_sum.template data<T>(),
            saved_mean_data,
            saved_rstd_data);
      } else {
        if (order_ == StorageOrder::NCHW) {
          const std::array<int, 3> X_dims_arr = {N, C, HxW};
          const std::array<int, 3> Y_dims_arr = {1, C, 1};
          math::Moments<T, Context>(
              3,
              X_dims_arr.data(),
              Y_dims_arr.data(),
              X_data,
              saved_mean_data,
              saved_rstd_data,
              &context_);
        } else {
          const std::array<int, 2> X_dims_arr = {N * HxW, C};
          const std::array<int, 2> Y_dims_arr = {1, C};
          math::Moments<T, Context>(
              2,
              X_dims_arr.data(),
              Y_dims_arr.data(),
              X_data,
              saved_mean_data,
              saved_rstd_data,
              &context_);
        }
      }
      ComputeRunningMomentsAndFusedParam<T>(
          C,
          scale_data,
          bias_data,
          saved_mean_data,
          saved_rstd_data,
          running_mean_data,
          running_var_data,
          saved_rstd_data,
          alpha_data,
          beta_data);
    }
    if (order_ == StorageOrder::NCHW) {
      math::AffineChannel<T, Context, StorageOrder::NCHW>(
          N, C, HxW, X_data, alpha_data, beta_data, Y_data, &context_);
    } else {
      math::AffineChannel<T, Context, StorageOrder::NHWC>(
          N, C, HxW, X_data, alpha_data, beta_data, Y_data, &context_);
    }

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
    EigenVectorArrayMap<T> alpha_arr(alpha, C);
    EigenVectorArrayMap<T> beta_arr(beta, C);
    alpha_arr = ConstEigenVectorArrayMap<T>(scale, C) *
        (ConstEigenVectorArrayMap<T>(var, C) + static_cast<T>(epsilon_))
            .rsqrt();
    beta_arr = ConstEigenVectorArrayMap<T>(bias, C) -
        alpha_arr * ConstEigenVectorArrayMap<T>(mean, C);
  }

  template <typename T>
  void ComputeBatchMoments(
      const int N,
      const int C,
      const int HxW,
      const T* batch_mean_sum,
      const T* batch_var_sum,
      T* mean,
      T* var) {
    const T scale = T(1) / static_cast<T>(num_batches_ * N * HxW);
    EigenVectorArrayMap<T> mean_arr(mean, C);
    EigenVectorArrayMap<T> var_arr(var, C);
    mean_arr = ConstEigenVectorArrayMap<T>(batch_mean_sum, C) * scale;
    var_arr = ConstEigenVectorArrayMap<T>(batch_var_sum, C) * scale -
        mean_arr.square();
  }

  template <typename T>
  void ComputeRunningMomentsAndFusedParam(
      const int C,
      const T* scale,
      const T* bias,
      const T* mean,
      const T* var,
      T* running_mean,
      T* running_var,
      T* rstd,
      T* alpha,
      T* beta) {
    const T a = T(1) - static_cast<T>(momentum_);
    const T b = static_cast<T>(momentum_);
    math::Axpby<T, T, Context>(C, a, mean, b, running_mean, &context_);
    math::Axpby<T, T, Context>(C, a, var, b, running_var, &context_);
    math::InvStd<T, Context>(C, static_cast<T>(epsilon_), var, rstd, &context_);
    EigenVectorArrayMap<T> alpha_arr(alpha, C);
    EigenVectorArrayMap<T> beta_arr(beta, C);
    alpha_arr = ConstEigenVectorArrayMap<T>(scale, C) *
        ConstEigenVectorArrayMap<T>(rstd, C);
    beta_arr = ConstEigenVectorArrayMap<T>(bias, C) -
        alpha_arr * ConstEigenVectorArrayMap<T>(mean, C);
  }

  const bool is_test_;
  double epsilon_;
  const float momentum_;
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
};

template <class Context>
class SpatialBNGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit SpatialBNGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(double, "epsilon", epsilon_, 1e-5),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))),
        OP_SINGLE_ARG(int, "num_batches", num_batches_, 1) {
    CAFFE_ENFORCE_NE(
        order_,
        StorageOrder::UNKNOWN,
        "order should be either \"NCHW\" or \"NHWC\".");
    CAFFE_ENFORCE(InputSize() == 5 || InputSize() == 7);
    CAFFE_ENFORCE_EQ(OutputSize(), 3);
  }

  virtual ~SpatialBNGradientOp() = default;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(INPUT);
    const auto& dY = Input(OUTPUT_GRAD);
    const auto& scale = Input(SCALE);
    const auto& mean = Input(SAVED_MEAN);
    const auto& rstd = Input(SAVED_INV_STD);
    const int ndim = X.dim();
    CAFFE_ENFORCE_GE(ndim, 3);
    const int N = X.dim32(0);
    const int C =
        (order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1));
    const std::vector<int> X_dims(X.sizes().cbegin(), X.sizes().cend());
    const int HxW =
        std::accumulate(
            X_dims.cbegin() + 1, X_dims.cend(), 1, std::multiplies<int>()) /
        C;
    CAFFE_ENFORCE_EQ(scale.numel(), C);
    CAFFE_ENFORCE_EQ(mean.numel(), C);
    CAFFE_ENFORCE_EQ(rstd.numel(), C);

    auto* dX = Output(INPUT_GRAD, X.sizes(), at::dtype<T>());
    at::IntArrayRef dscale_sizes, dbias_sizes;
    if (num_batches_ == 1) {
      dscale_sizes = scale.sizes();
      dbias_sizes = scale.sizes();
    } else {
      const auto& dscale_sum = Input(AGGREGATE_SCALE_GRAD);
      const auto& dbias_sum = Input(AGGREGATE_BIAS_GRAD);
      // Note: previously there was alias check to decide whether to call
      // ResizeLike or not, since we only call Resize when the size does not
      // match the size of cached Tensor, this check is not necessary
      dscale_sizes = dscale_sum.sizes();
      dbias_sizes = dbias_sum.sizes();
    }
    auto* dscale = Output(SCALE_GRAD, dscale_sizes, at::dtype<T>());
    auto* dbias = Output(BIAS_GRAD, dbias_sizes, at::dtype<T>());
    const T* X_data = X.template data<T>();
    const T* dY_data = dY.template data<T>();
    const T* scale_data = scale.template data<T>();
    const T* mean_data = mean.template data<T>();
    const T* rstd_data = rstd.template data<T>();
    T* dX_data = dX->template mutable_data<T>();
    T* dscale_data = dscale->template mutable_data<T>();
    T* dbias_data = dbias->template mutable_data<T>();

    if (N == 0) {
      math::Set<T, Context>(C, T(0), dscale_data, &context_);
      math::Set<T, Context>(C, T(0), dbias_data, &context_);
      return true;
    }
    ReinitializeTensor(
        &alpha_, {C}, at::dtype<T>().device(Context::GetDeviceType()));
    ReinitializeTensor(
        &beta_, {C}, at::dtype<T>().device(Context::GetDeviceType()));
    ReinitializeTensor(
        &gamma_, {C}, at::dtype<T>().device(Context::GetDeviceType()));
    T* alpha_data = alpha_.template mutable_data<T>();
    T* beta_data = beta_.template mutable_data<T>();
    T* gamma_data = gamma_.template mutable_data<T>();
    if (num_batches_ > 1) {
      const auto& dscale_sum = Input(AGGREGATE_SCALE_GRAD);
      const auto& dbias_sum = Input(AGGREGATE_BIAS_GRAD);
      ComputeMultiBatchScaleBiasGradientsAndFusedParams<T>(
          N,
          C,
          HxW,
          scale_data,
          mean_data,
          rstd_data,
          dscale_sum.template data<T>(),
          dbias_sum.template data<T>(),
          dscale_data,
          dbias_data,
          alpha_data,
          beta_data,
          gamma_data);
    } else {
      ComputeScaleBiasGradientsAndFusedParams<T>(
          N,
          C,
          HxW,
          dY_data,
          X_data,
          scale_data,
          mean_data,
          rstd_data,
          dscale_data,
          dbias_data,
          alpha_data,
          beta_data,
          gamma_data,
          dX_data);
    }
    ComputeXGradient<T>(
        N, C, HxW, dY_data, X_data, alpha_data, beta_data, gamma_data, dX_data);

    return true;
  }

 protected:
  template <typename T>
  void ComputeMultiBatchScaleBiasGradientsAndFusedParams(
      const int N,
      const int C,
      const int HxW,
      const T* scale,
      const T* mean,
      const T* rstd,
      const T* dscale_sum,
      const T* dbias_sum,
      T* dscale,
      T* dbias,
      T* alpha,
      T* beta,
      T* gamma);

  template <typename T>
  void ComputeScaleBiasGradientsAndFusedParams(
      const int N,
      const int C,
      const int HxW,
      const T* dY,
      const T* X,
      const T* scale,
      const T* mean,
      const T* rstd,
      T* dscale,
      T* dbias,
      T* alpha,
      T* beta,
      T* gamma,
      T* scratch);

  template <typename T>
  void ComputeXGradient(
      const int N,
      const int C,
      const int HxW,
      const T* dY,
      const T* X,
      const T* alpha,
      const T* beta,
      const T* gamma,
      T* dX);

  double epsilon_;
  const StorageOrder order_;
  const int num_batches_;

  Tensor alpha_;
  Tensor beta_;
  Tensor gamma_;
  Tensor ones_;

  INPUT_TAGS(
      INPUT,
      SCALE,
      OUTPUT_GRAD,
      SAVED_MEAN,
      SAVED_INV_STD,
      AGGREGATE_SCALE_GRAD,
      AGGREGATE_BIAS_GRAD);
  OUTPUT_TAGS(INPUT_GRAD, SCALE_GRAD, BIAS_GRAD);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SPATIAL_BATCH_NORM_OP_H_
