#include "caffe2/operators/spatial_batch_norm_op.h"

#include <array>
#include <functional>
#include <numeric>
#include <vector>

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"
#include "caffe2/operators/spatial_batch_norm_op_impl.cuh"
#include "caffe2/utils/math.h"

#if CUDNN_VERSION_MIN(5, 0, 0)

namespace caffe2 {

namespace {

void SetTensorDescriptor(
    const cudnnDataType_t data_type,
    const cudnnBatchNormMode_t mode,
    const StorageOrder order,
    const std::vector<int>& input_dims,
    cudnnTensorDescriptor_t data_desc,
    cudnnTensorDescriptor_t param_desc) {
  const int ndim = input_dims.size();
  const int N = input_dims[0];
  const int C = order == StorageOrder::NCHW ? input_dims[1] : input_dims.back();
  if (ndim == 3) {
    const int H = 1;
    const int W = order == StorageOrder::NCHW ? input_dims[2] : input_dims[1];
    CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
        data_desc, GetCudnnTensorFormat(order), data_type, N, C, H, W));
  } else if (ndim == 4) {
    const int H = order == StorageOrder::NCHW ? input_dims[2] : input_dims[1];
    const int W = order == StorageOrder::NCHW ? input_dims[3] : input_dims[2];
    CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
        data_desc, GetCudnnTensorFormat(order), data_type, N, C, H, W));
  } else {
    const int H = order == StorageOrder::NCHW ? input_dims[2] : input_dims[1];
    const int W = order == StorageOrder::NCHW ? input_dims[3] : input_dims[2];
    const auto l_iter = order == StorageOrder::NCHW ? input_dims.cbegin() + 4
                                                    : input_dims.cbegin() + 3;
    const auto r_iter =
        order == StorageOrder::NCHW ? input_dims.cend() : input_dims.cend() - 1;
    const int D = std::accumulate(l_iter, r_iter, 1, std::multiplies<int>());
    const std::array<int, 5> dims = {N, C, H, W, D};
    const std::array<int, 5> strides = order == StorageOrder::NCHW
        ? std::array<int, 5>{C * H * W * D, H * W * D, W * D, D, 1}
        : std::array<int, 5>{C * H * W * D, 1, W * D * C, D * C, C};
    CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
        data_desc, data_type, 5, dims.data(), strides.data()));
  }
  CUDNN_ENFORCE(cudnnDeriveBNTensorDescriptor(param_desc, data_desc, mode));
}

} // namespace

class CuDNNSpatialBNOp final : public SpatialBNOp<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  CuDNNSpatialBNOp(const OperatorDef& operator_def, Workspace* ws)
      : SpatialBNOp<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_),
#if CUDNN_VERSION_MIN(7, 0, 0)
        // TODO(T31829456): The new CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode was
        // introduced in CuDNN 7 for performance optimization, but it results in
        // accuracy losses in convolution models such as ResNeXt-101 and
        // video R(2+1)D. We will fall back to the normal
        // CUDNN_BATCHNORM_SPATIAL for now
        mode_(CUDNN_BATCHNORM_SPATIAL) {
#else
        mode_(CUDNN_BATCHNORM_SPATIAL) {
#endif
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&data_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&param_desc_));
    if (epsilon_ < CUDNN_BN_MIN_EPSILON) {
      LOG(ERROR) << "Provided epsilon is smaller than CUDNN_BN_MIN_EPSILON. "
                    "Setting it to CUDNN_BN_MIN_EPSILON instead.";
      epsilon_ = CUDNN_BN_MIN_EPSILON;
    }
  }

  ~CuDNNSpatialBNOp() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(param_desc_));
  }

  bool RunOnDevice() override {
    // CuDNN doesn't support multi-batch SpatialBN and it's NHWC order SpatialBN
    // is much slower, so in such cases fallback to SpatialBNOp<CUDAContext>.
    if (num_batches_ > 1 || order_ == StorageOrder::NHWC) {
      return SpatialBNOp<CUDAContext>::RunOnDevice();
    }
    return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    typedef typename cudnnTypeWrapper<T>::BNParamType BNParamType;

    const auto& X = Input(INPUT);
    const auto& scale = Input(SCALE);
    const auto& bias = Input(BIAS);
    auto* Y = Output(OUTPUT);
    const int ndim = X.ndim();
    CAFFE_ENFORCE_GE(ndim, 3);
    const int N = X.dim32(0);
    const int C =
        (order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1));
    CAFFE_ENFORCE_EQ(scale.size(), C);
    CAFFE_ENFORCE_EQ(bias.size(), C);
    Y->ResizeLike(X);
    const T* X_data = X.data<T>();
    const BNParamType* scale_data = scale.data<BNParamType>();
    const BNParamType* bias_data = bias.data<BNParamType>();
    T* Y_data = Y->mutable_data<T>();

    if (N > 0) {
      const std::vector<int> input_dims(X.sizes().cbegin(), X.sizes().cend());
      if (input_dims != data_dims_) {
        data_dims_ = input_dims;
        SetTensorDescriptor(
            cudnnTypeWrapper<T>::type,
            mode_,
            order_,
            input_dims,
            data_desc_,
            param_desc_);
      }
    }
    if (is_test_) {
      const auto& mean = Input(EST_MEAN);
      const auto& var = Input(EST_VAR);
      CAFFE_ENFORCE_EQ(mean.size(), C);
      CAFFE_ENFORCE_EQ(var.size(), C);
      if (N == 0) {
        return true;
      }
      CUDNN_ENFORCE(cudnnBatchNormalizationForwardInference(
          cudnn_wrapper_.inline_cudnn_handle(),
          // Note: PERSISTENT not implemented for inference
          CUDNN_BATCHNORM_SPATIAL,
          cudnnTypeWrapper<T>::kOne(),
          cudnnTypeWrapper<T>::kZero(),
          data_desc_,
          X_data,
          data_desc_,
          Y_data,
          param_desc_,
          scale_data,
          bias_data,
          mean.data<BNParamType>(),
          var.data<BNParamType>(),
          epsilon_));
    } else {
      auto* saved_mean = Output(SAVED_MEAN);
      auto* saved_inv_std = Output(SAVED_INV_STD);
      saved_mean->Resize(C);
      saved_inv_std->Resize(C);
      BNParamType* saved_mean_data = saved_mean->mutable_data<BNParamType>();
      BNParamType* saved_inv_std_data =
          saved_inv_std->mutable_data<BNParamType>();
      auto* running_mean = Output(RUNNING_MEAN);
      auto* running_var = Output(RUNNING_VAR);
      if (running_mean->size() != C) {
        running_mean->Resize(C);
        math::Set<BNParamType, CUDAContext>(
            C,
            BNParamType(0),
            running_mean->mutable_data<BNParamType>(),
            &context_);
      }
      if (running_var->size() != C) {
        running_var->Resize(C);
        math::Set<BNParamType, CUDAContext>(
            C,
            BNParamType(0),
            running_var->mutable_data<BNParamType>(),
            &context_);
      }
      BNParamType* running_mean_data =
          running_mean->mutable_data<BNParamType>();
      BNParamType* running_var_data = running_var->mutable_data<BNParamType>();
      if (N == 0) {
        math::Set<BNParamType, CUDAContext>(
            C, BNParamType(0), saved_mean_data, &context_);
        math::Set<BNParamType, CUDAContext>(
            C, BNParamType(0), saved_inv_std_data, &context_);
        return true;
      }
      const double alpha = static_cast<double>(1.0f - momentum_);

#if CUDNN_VERSION_MIN(8, 0, 0)
      // Currently not supporting CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION
      auto op = CUDNN_BATCHNORM_OPS_BN;

      // Calculate the workspace size
      size_t workspace_size;
      CUDNN_ENFORCE(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
          cudnn_wrapper_.inline_cudnn_handle(),
          mode_,
          op,
          data_desc_,
          NULL,
          data_desc_,
          param_desc_,
          NULL,
          &workspace_size));

      // Calculate the reserved space size - common function for forward and backward
      size_t reserve_size;
      CUDNN_ENFORCE(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
          cudnn_wrapper_.inline_cudnn_handle(),
          mode_,
          op,
          NULL,
          data_desc_,
          &reserve_size));

      // CUDNN state is needed to access the workspace
      size_t cudnn_state_(OperatorBase::GetSingleArgument<int>("cudnn_state", 0));
      cudnn_wrapper_.with_cudnn_state(
        cudnn_state_, [&](CuDNNState* state) {
          CUDNN_ENFORCE(cudnnBatchNormalizationForwardTrainingEx(
              cudnn_wrapper_.inline_cudnn_handle(),
              mode_,
              CUDNN_BATCHNORM_OPS_BN,
              cudnnTypeWrapper<T>::kOne(),
              cudnnTypeWrapper<T>::kZero(),
              data_desc_,
              X_data,
              NULL,
              NULL,
              data_desc_,
              Y_data,
              param_desc_,
              scale_data,
              bias_data,
              alpha,
              running_mean_data,
              running_var_data,
              epsilon_,
              saved_mean_data,
              saved_inv_std_data,
              NULL,
              state->workspace().get(workspace_size),
              workspace_size,
              state->workspace().get(reserve_size),
              reserve_size));
          });
#else
      CUDNN_ENFORCE(cudnnBatchNormalizationForwardTraining(
          cudnn_wrapper_.inline_cudnn_handle(),
          mode_,
          cudnnTypeWrapper<T>::kOne(),
          cudnnTypeWrapper<T>::kZero(),
          data_desc_,
          X_data,
          data_desc_,
          Y_data,
          param_desc_,
          scale_data,
          bias_data,
          alpha,
          running_mean_data,
          running_var_data,
          epsilon_,
          saved_mean_data,
          saved_inv_std_data));
#endif // CUDNN_VERSION_MIN(8, 0, 0)
    }
    return true;
  }

 private:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t data_desc_;
  cudnnTensorDescriptor_t param_desc_;
  cudnnBatchNormMode_t mode_;

  std::vector<int> data_dims_;
};

class CuDNNSpatialBNGradientOp final : public SpatialBNGradientOp<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  CuDNNSpatialBNGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : SpatialBNGradientOp<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_),
#if CUDNN_VERSION_MIN(7, 0, 0)
        // TODO(T31829456): The new CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode was
        // introduced in CuDNN 7 for performance optimization, but it results in
        // accuracy losses in convolution models such as ResNeXt-101 and
        // video R(2+1)D. We will fall back to the normal
        // CUDNN_BATCHNORM_SPATIAL for now
        mode_(CUDNN_BATCHNORM_SPATIAL) {
#else
        mode_(CUDNN_BATCHNORM_SPATIAL) {
#endif
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&data_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&param_desc_));
    if (epsilon_ < CUDNN_BN_MIN_EPSILON) {
      LOG(ERROR) << "Provided epsilon is smaller than CUDNN_BN_MIN_EPSILON. "
                    "Setting it to CUDNN_BN_MIN_EPSILON instead.";
      epsilon_ = CUDNN_BN_MIN_EPSILON;
    }
  }

  ~CuDNNSpatialBNGradientOp() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(param_desc_));
  }

  bool RunOnDevice() override {
    // CuDNN doesn't support multi-batch SpatialBN and it's NHWC order SpatialBN
    // is much slower, so in such cases fallback to SpatialBNOp<CUDAContext>.
    if (num_batches_ > 1 || order_ == StorageOrder::NHWC) {
      return SpatialBNGradientOp<CUDAContext>::RunOnDevice();
    }
    return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    typedef typename cudnnTypeWrapper<T>::BNParamType BNParamType;

    const auto& X = Input(INPUT);
    const auto& scale = Input(SCALE);
    const auto& dY = Input(OUTPUT_GRAD);
    const auto& saved_mean = Input(SAVED_MEAN);
    const auto& saved_rstd = Input(SAVED_INV_STD);
    auto* dX = Output(INPUT_GRAD);
    auto* dscale = Output(SCALE_GRAD);
    auto* dbias = Output(BIAS_GRAD);
    const int ndim = X.ndim();
    CAFFE_ENFORCE_GE(ndim, 3);
    const int N = X.dim32(0);
    const int C =
        (order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1));
    CAFFE_ENFORCE_EQ(scale.size(), C);
    CAFFE_ENFORCE_EQ(saved_mean.size(), C);
    CAFFE_ENFORCE_EQ(saved_rstd.size(), C);
    dX->ResizeLike(X);
    dscale->ResizeLike(scale);
    dbias->ResizeLike(scale);
    const T* X_data = X.template data<T>();
    const T* scale_data = scale.template data<T>();
    const T* dY_data = dY.template data<T>();
    const BNParamType* saved_mean_data =
        saved_mean.template data<BNParamType>();
    const BNParamType* saved_rstd_data =
        saved_rstd.template data<BNParamType>();
    T* dX_data = dX->template mutable_data<T>();
    BNParamType* dscale_data = dscale->template mutable_data<BNParamType>();
    BNParamType* dbias_data = dbias->template mutable_data<BNParamType>();
    if (N == 0) {
      math::Set<BNParamType, CUDAContext>(
          C, BNParamType(0), dscale_data, &context_);
      math::Set<BNParamType, CUDAContext>(
          C, BNParamType(0), dbias_data, &context_);
      return true;
    }

    const std::vector<int> input_dims(X.sizes().cbegin(), X.sizes().cend());
    if (input_dims != data_dims_) {
      data_dims_ = input_dims;
      SetTensorDescriptor(
          cudnnTypeWrapper<T>::type,
          mode_,
          order_,
          input_dims,
          data_desc_,
          param_desc_);
    }
#if CUDNN_VERSION_MIN(8, 0, 0)
    // Currently not supporting CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION
    auto op = CUDNN_BATCHNORM_OPS_BN;

    size_t workspace_size;
    CUDNN_ENFORCE(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        mode_,
        op,
        data_desc_,
        NULL,
        data_desc_,
        NULL,
        data_desc_,
        param_desc_,
        NULL,
        &workspace_size));

    // Calculate the reserved space size - common function for forward and backward
    size_t reserve_size;
    CUDNN_ENFORCE(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        mode_,
        op,
        NULL,
        data_desc_,
        &reserve_size));

    // CUDNN state is needed to access the workspace
    size_t cudnn_state_(OperatorBase::GetSingleArgument<int>("cudnn_state", 0));
    cudnn_wrapper_.with_cudnn_state(
      cudnn_state_, [&](CuDNNState* state) {
        CUDNN_ENFORCE(cudnnBatchNormalizationBackwardEx(
            cudnn_wrapper_.inline_cudnn_handle(),
            mode_,
            op,
            cudnnTypeWrapper<T>::kOne(),
            cudnnTypeWrapper<T>::kZero(),
            cudnnTypeWrapper<T>::kOne(),
            cudnnTypeWrapper<T>::kZero(),
            data_desc_,
            X_data,
            NULL,
            NULL,
            data_desc_,
            dY_data,
            NULL,
            NULL,
            data_desc_,
            dX_data,
            param_desc_,
            scale_data,
            NULL,
            dscale_data,
            dbias_data,
            epsilon_,
            saved_mean_data,
            saved_rstd_data,
            NULL,
            state->workspace().get(workspace_size),
            workspace_size,
            state->workspace().get(reserve_size),
            reserve_size));
      });
#else
    CUDNN_ENFORCE(cudnnBatchNormalizationBackward(
        cudnn_wrapper_.inline_cudnn_handle(),
        mode_,
        cudnnTypeWrapper<T>::kOne(),
        cudnnTypeWrapper<T>::kZero(),
        cudnnTypeWrapper<T>::kOne(),
        cudnnTypeWrapper<T>::kZero(),
        data_desc_,
        X_data,
        data_desc_,
        dY_data,
        data_desc_,
        dX_data,
        param_desc_,
        scale_data,
        dscale_data,
        dbias_data,
        epsilon_,
        saved_mean_data,
        saved_rstd_data));
#endif // CUDNN_VERSION_MIN(8, 0, 0)
    return true;
  }

 private:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t data_desc_;
  cudnnTensorDescriptor_t param_desc_;
  cudnnBatchNormMode_t mode_;

  // TODO: int -> int64_t
  std::vector<int> data_dims_;
};

REGISTER_CUDNN_OPERATOR(SpatialBN, CuDNNSpatialBNOp);
REGISTER_CUDNN_OPERATOR(SpatialBNGradient, CuDNNSpatialBNGradientOp);

} // namespace caffe2

#endif // CUDNN_VERSION_MIN(5, 0, 0)
