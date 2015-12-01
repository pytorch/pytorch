#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/spatial_batch_norm_op.h"

#if CUDNN_VERSION >= 4000

namespace caffe2 {

constexpr cudnnBatchNormMode_t kSpatialBNMode = CUDNN_BATCHNORM_SPATIAL;

template <typename T>
class CudnnSpatialBNOp final : public SpatialBNOpBase<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  CudnnSpatialBNOp(const OperatorDef& operator_def, Workspace* ws)
      : SpatialBNOpBase<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&device_context_) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&data_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_param_desc_));
    if (epsilon_ < CUDNN_BN_MIN_EPSILON) {
      CAFFE_LOG_ERROR << "Provided epsilon is smaller than "
                      << "CUDNN_BN_MIN_EPSILON. Setting it to "
                      << "CUDNN_BN_MIN_EPSILON instead.";
      epsilon_ = CUDNN_BN_MIN_EPSILON;
    }
  }

  ~CudnnSpatialBNOp() {
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bn_param_desc_));
  }

  bool RunOnDevice() override;

 protected:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t data_desc_;
  cudnnTensorDescriptor_t bn_param_desc_;
  vector<int> cudnn_input_dims_;
  DISABLE_COPY_AND_ASSIGN(CudnnSpatialBNOp);
};


template <typename T>
class CudnnSpatialBNGradientOp final
    : public SpatialBNGradientOpBase<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  CudnnSpatialBNGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : SpatialBNGradientOpBase<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&device_context_) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&data_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_param_desc_));
    if (epsilon_ < CUDNN_BN_MIN_EPSILON) {
      CAFFE_LOG_ERROR << "Provided epsilon is smaller than "
                      << "CUDNN_BN_MIN_EPSILON. Setting it to "
                      << "CUDNN_BN_MIN_EPSILON instead.";
      epsilon_ = CUDNN_BN_MIN_EPSILON;
    }
  }

  ~CudnnSpatialBNGradientOp() {
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bn_param_desc_));
  }

  bool RunOnDevice() override;

 protected:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t data_desc_;
  cudnnTensorDescriptor_t bn_param_desc_;
  vector<int> cudnn_input_dims_;
  DISABLE_COPY_AND_ASSIGN(CudnnSpatialBNGradientOp);
};


////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool CudnnSpatialBNOp<T>::RunOnDevice() {
  const auto& X = Input(INPUT);
  const auto& scale = Input(SCALE);
  const auto& bias = Input(BIAS);

  CAFFE_DCHECK_EQ(X.ndim(), 4);
  const int N = X.dim(0);
  const int C = (order_ == StorageOrder::NCHW ? X.dim(1) : X.dim(3));
  const int H = (order_ == StorageOrder::NCHW ? X.dim(2) : X.dim(1));
  const int W = (order_ == StorageOrder::NCHW ? X.dim(3) : X.dim(2));
  CAFFE_DCHECK_EQ(scale.ndim(), 1);
  CAFFE_DCHECK_EQ(bias.ndim(), 1);
  CAFFE_DCHECK_EQ(scale.dim(0), C);
  CAFFE_DCHECK_EQ(bias.dim(0), C);
  // See if we need to reshape.
  if (X.dims() != cudnn_input_dims_) {
    CAFFE_VLOG(1) << "Setting descriptors.";
    cudnn_input_dims_ = X.dims();
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        data_desc_, GetCudnnTensorFormat(order_),
        cudnnTypeWrapper<T>::type, N, C, H, W));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(
        bn_param_desc_, data_desc_, kSpatialBNMode));
  }

  // Now, depending on whether we are running test or not, we have two paths.
  const typename cudnnTypeWrapper<T>::ScalingParamType kOne = 1;
  const typename cudnnTypeWrapper<T>::ScalingParamType kZero = 0;
  if (is_test_) {
    // Run inference mode.
    const auto& est_mean = Input(EST_MEAN);
    const auto& est_inv_var = Input(EST_INV_VAR);
    CAFFE_DCHECK_EQ(est_mean.ndim(), 1);
    CAFFE_DCHECK_EQ(est_inv_var.ndim(), 1);
    CAFFE_DCHECK_EQ(est_mean.dim(0), C);
    CAFFE_DCHECK_EQ(est_inv_var.dim(0), C);

    auto* Y = Output(OUTPUT);
    Y->ReshapeLike(X);
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
        cudnn_wrapper_.cudnn_handle(), kSpatialBNMode, &kOne, &kZero,
        data_desc_, X.template data<T>(),
        data_desc_, Y->template mutable_data<T>(),
        bn_param_desc_, scale.template data<T>(), bias.template data<T>(),
        est_mean.template data<T>(), est_inv_var.template data<T>(),
        epsilon_));
  } else {
    // Run training mode.
    auto* Y = Output(OUTPUT);
    Y->ReshapeLike(X);
    // obtain running mean and running inv var, and see if we need to
    // initialize them.
    auto* running_mean = Output(RUNNING_MEAN);
    auto* running_inv_var = Output(RUNNING_INV_VAR);
    double this_momentum;
    if (running_mean->size() == 0) {
      CAFFE_VLOG(1) << "Initializing running mean and var.";
      // Need to do initialization
      running_mean->Reshape(C);
      running_inv_var->Reshape(C);
      this_momentum = 1;
    } else {
      // Does not need to do initialization.
      CAFFE_DCHECK_EQ(running_mean->ndim(), 1);
      CAFFE_DCHECK_EQ(running_inv_var->ndim(), 1);
      CAFFE_DCHECK_EQ(running_mean->dim(0), C);
      CAFFE_DCHECK_EQ(running_inv_var->dim(0), C);
      this_momentum = momentum_;
    }
    // If specified, save the mean and inv var results.
    void* save_mean_data = nullptr;
    void* save_inv_var_data = nullptr;
    if (OutputSize() == 5) {
      auto* save_mean = Output(SAVED_MEAN);
      auto* save_inv_var = Output(SAVED_INV_VAR);
      save_mean->Reshape(C);
      save_inv_var->Reshape(C);
      save_mean_data = save_mean->template mutable_data<T>();
      save_inv_var_data = save_inv_var->template mutable_data<T>();
    }

    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
        cudnn_wrapper_.cudnn_handle(), kSpatialBNMode, &kOne, &kZero,
        data_desc_, X.template data<T>(),
        data_desc_, Y->template mutable_data<T>(),
        bn_param_desc_, scale.template data<T>(), bias.template data<T>(),
        this_momentum, running_mean->template mutable_data<T>(),
        running_inv_var->template mutable_data<T>(), epsilon_,
        save_mean_data, save_inv_var_data));
  }
  return true;
}


template <typename T>
bool CudnnSpatialBNGradientOp<T>::RunOnDevice() {
  const auto& X = Input(INPUT);
  const auto& scale = Input(SCALE);
  const auto& dY = Input(OUTPUT_GRAD);

  CAFFE_DCHECK_EQ(X.ndim(), 4);
  const int N = X.dim(0);
  const int C = (order_ == StorageOrder::NCHW ? X.dim(1) : X.dim(3));
  const int H = (order_ == StorageOrder::NCHW ? X.dim(2) : X.dim(1));
  const int W = (order_ == StorageOrder::NCHW ? X.dim(3) : X.dim(2));
  CAFFE_DCHECK_EQ(scale.ndim(), 1);
  CAFFE_DCHECK_EQ(scale.dim(0), C);
  // See if we need to reshape.
  if (X.dims() != cudnn_input_dims_) {
    cudnn_input_dims_ = X.dims();
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        data_desc_, GetCudnnTensorFormat(order_),
        cudnnTypeWrapper<T>::type, N, C, H, W));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(
        bn_param_desc_, data_desc_, kSpatialBNMode));
  }

  auto* dX = Output(INPUT_GRAD);
  auto* dScale = Output(SCALE_GRAD);
  auto* dBias = Output(BIAS_GRAD);
  dX->ReshapeLike(X);
  dScale->ReshapeLike(scale);
  dBias->ReshapeLike(scale);

  const void* saved_mean_data = nullptr;
  const void* saved_inv_var_data = nullptr;
  if (InputSize() == 5) {
    const auto& saved_mean = Input(SAVED_MEAN);
    const auto& saved_inv_var = Input(SAVED_INV_VAR);
    saved_mean_data = saved_mean.template data<T>();
    saved_inv_var_data = saved_inv_var.template data<T>();
  }

  const typename cudnnTypeWrapper<T>::ScalingParamType kOne = 1;
  const typename cudnnTypeWrapper<T>::ScalingParamType kZero = 0;
  CUDNN_CHECK(cudnnBatchNormalizationBackward(
      cudnn_wrapper_.cudnn_handle(), kSpatialBNMode, &kOne, &kZero,
      data_desc_, X.template data<T>(), data_desc_, dY.template data<T>(),
      data_desc_, dX->template mutable_data<T>(),
      bn_param_desc_, scale.template data<T>(),
      dScale->template mutable_data<T>(), dBias->template mutable_data<T>(),
      epsilon_, saved_mean_data, saved_inv_var_data));
  return true;
}

namespace {
// Since there is no default implementation for spatial batch normalization,
// we will register the cudnn version as the default as well.
REGISTER_CUDA_OPERATOR(SpatialBN, CudnnSpatialBNOp<float>);
REGISTER_CUDA_OPERATOR(SpatialBNGradient, CudnnSpatialBNGradientOp<float>);

REGISTER_CUDNN_OPERATOR(SpatialBN, CudnnSpatialBNOp<float>);
REGISTER_CUDNN_OPERATOR(SpatialBNGradient, CudnnSpatialBNGradientOp<float>);
}  // namespace
}  // namespace caffe2

#endif  // CUDNN_VERSION >= 4000

