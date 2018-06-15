#include "caffe2/operators/affine_channel_op.h"

#include <algorithm>
#include <array>
#include <vector>

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

class CuDNNAffineChannelOpBase : public Operator<CUDAContext> {
 public:
  CuDNNAffineChannelOpBase(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<std::string>("order", "NCHW"))),
        OP_SINGLE_ARG(bool, "is_learnable", is_learnable_, false),
        cudnn_wrapper_(&context_) {
    CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);

    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&X_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&scale_desc_));
    CUDNN_ENFORCE(cudnnCreateOpTensorDescriptor(&mul_desc_));
  }

  virtual ~CuDNNAffineChannelOpBase() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(X_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(scale_desc_));
    CUDNN_ENFORCE(cudnnDestroyOpTensorDescriptor(mul_desc_));
  }

 protected:
  void SetTensorDesc4D(
      const cudnnDataType_t cudnn_type,
      const int N,
      const int C,
      const int H,
      const int W) {
    CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
        X_desc_, GetCudnnTensorFormat(order_), cudnn_type, N, C, H, W));
    CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
        scale_desc_, GetCudnnTensorFormat(order_), cudnn_type, 1, C, 1, 1));
  }

  void SetTensorDescND(
      const cudnnDataType_t cudnn_type,
      const std::vector<int>& X_dims) {
    const int ndim = X_dims.size();
    const int C_dim = order_ == StorageOrder::NCHW ? 1 : ndim - 1;
    const int C = X_dims[C_dim];
    std::vector<int> X_strides(ndim);
    X_strides.back() = 1;
    for (int i = ndim - 1; i > 0; --i) {
      X_strides[i - 1] = X_strides[i] * X_dims[i];
    }
    std::vector<int> scale_dims(ndim, 1);
    scale_dims[C_dim] = C;
    std::vector<int> scale_strides(ndim);
    std::fill(scale_strides.begin(), scale_strides.begin() + C_dim, C);
    std::fill(scale_strides.begin() + C_dim, scale_strides.end(), 1);
    CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
        X_desc_, cudnn_type, ndim, X_dims.data(), X_strides.data()));
    CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
        scale_desc_,
        cudnn_type,
        ndim,
        scale_dims.data(),
        scale_strides.data()));
  }

  const StorageOrder order_;
  const bool is_learnable_;

  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t X_desc_;
  cudnnTensorDescriptor_t scale_desc_;
  cudnnOpTensorDescriptor_t mul_desc_;
};

class CuDNNAffineChannelOp final : public CuDNNAffineChannelOpBase {
 public:
  CuDNNAffineChannelOp(const OperatorDef& operator_def, Workspace* ws)
      : CuDNNAffineChannelOpBase(operator_def, ws) {
    CUDNN_ENFORCE(cudnnCreateOpTensorDescriptor(&add_desc_));
  }

  ~CuDNNAffineChannelOp() {
    CUDNN_ENFORCE(cudnnDestroyOpTensorDescriptor(add_desc_));
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);
    const auto& scale = Input(1);
    const auto& bias = Input(2);
    auto* Y = Output(0);
    if (is_learnable_) {
      CAFFE_ENFORCE_NE(
          Y,
          &X,
          "In-place affine_channel_op is not supported when "
          "is_learnable = true.");
    }
    Y->ResizeLike(X);
    const T* X_data = X.data<T>();
    const T* scale_data = scale.data<T>();
    const T* bias_data = bias.data<T>();
    T* Y_data = Y->mutable_data<T>();
    const int ndim = X.ndim();
    CAFFE_ENFORCE_GE(ndim, 4);
    const cudnnDataType_t cudnn_type = cudnnTypeWrapper<T>::type;
    if (ndim == 4) {
      const int N = X.dim32(0);
      const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(3);
      const int H = order_ == StorageOrder::NCHW ? X.dim32(2) : X.dim32(1);
      const int W = order_ == StorageOrder::NCHW ? X.dim32(3) : X.dim32(2);
      SetTensorDesc4D(cudnn_type, N, C, H, W);
    } else {
      const std::vector<int> X_dims(X.dims().cbegin(), X.dims().cend());
      SetTensorDescND(cudnn_type, X_dims);
    }
    CUDNN_ENFORCE(cudnnSetOpTensorDescriptor(
        mul_desc_, CUDNN_OP_TENSOR_MUL, cudnn_type, CUDNN_PROPAGATE_NAN));
    CUDNN_ENFORCE(cudnnOpTensor(
        cudnn_wrapper_.inline_cudnn_handle(),
        mul_desc_,
        cudnnTypeWrapper<T>::kOne(),
        X_desc_,
        X_data,
        cudnnTypeWrapper<T>::kOne(),
        scale_desc_,
        scale_data,
        cudnnTypeWrapper<T>::kZero(),
        X_desc_,
        Y_data));
    if (ndim == 4) {
      CUDNN_ENFORCE(cudnnAddTensor(
          cudnn_wrapper_.inline_cudnn_handle(),
          cudnnTypeWrapper<T>::kOne(),
          scale_desc_,
          bias_data,
          cudnnTypeWrapper<T>::kOne(),
          X_desc_,
          Y_data));
    } else {
      CUDNN_ENFORCE(cudnnSetOpTensorDescriptor(
          add_desc_, CUDNN_OP_TENSOR_ADD, cudnn_type, CUDNN_PROPAGATE_NAN));
      CUDNN_ENFORCE(cudnnOpTensor(
          cudnn_wrapper_.inline_cudnn_handle(),
          add_desc_,
          cudnnTypeWrapper<T>::kOne(),
          X_desc_,
          Y_data,
          cudnnTypeWrapper<T>::kOne(),
          scale_desc_,
          bias_data,
          cudnnTypeWrapper<T>::kZero(),
          X_desc_,
          Y_data));
    }
    return true;
  }

 private:
  cudnnOpTensorDescriptor_t add_desc_;
};

class CuDNNAffineChannelGradientOp final : public CuDNNAffineChannelOpBase {
 public:
  CuDNNAffineChannelGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : CuDNNAffineChannelOpBase(operator_def, ws) {
#if CUDNN_VERSION_MIN(6, 0, 0)
    CUDNN_ENFORCE(cudnnCreateReduceTensorDescriptor(&reduce_desc_));
#endif
  }

  ~CuDNNAffineChannelGradientOp() {
#if CUDNN_VERSION_MIN(6, 0, 0)
    CUDNN_ENFORCE(cudnnDestroyReduceTensorDescriptor(reduce_desc_));
#endif
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& dY = Input(0);
    const auto& scale = is_learnable_ ? Input(2) : Input(1);
    auto* dX = Output(0);
    dX->ResizeLike(dY);
    const T* dY_data = dY.data<T>();
    const T* scale_data = scale.data<T>();
    T* dX_data = dX->mutable_data<T>();
    const int ndim = dY.ndim();
    CAFFE_ENFORCE_GE(ndim, 4);
    const cudnnDataType_t cudnn_type = cudnnTypeWrapper<T>::type;
    const std::vector<int> X_dims(dY.dims().cbegin(), dY.dims().cend());
    SetTensorDescND(cudnn_type, X_dims);
    CUDNN_ENFORCE(cudnnSetOpTensorDescriptor(
        mul_desc_, CUDNN_OP_TENSOR_MUL, cudnn_type, CUDNN_PROPAGATE_NAN));
    CUDNN_ENFORCE(cudnnOpTensor(
        cudnn_wrapper_.inline_cudnn_handle(),
        mul_desc_,
        cudnnTypeWrapper<T>::kOne(),
        X_desc_,
        dY_data,
        cudnnTypeWrapper<T>::kOne(),
        scale_desc_,
        scale_data,
        cudnnTypeWrapper<T>::kZero(),
        X_desc_,
        dX_data));
    if (is_learnable_) {
      const auto& X = Input(1);
      const T* X_data = X.data<T>();
      auto* dscale = Output(1);
      auto* dbias = Output(2);
      dscale->ResizeLike(scale);
      dbias->ResizeLike(scale);
      T* dscale_data = dscale->mutable_data<T>();
      T* dbias_data = dbias->mutable_data<T>();
      if (X.size() == scale.size()) {
        CUDNN_ENFORCE(cudnnOpTensor(
            cudnn_wrapper_.inline_cudnn_handle(),
            mul_desc_,
            cudnnTypeWrapper<T>::kOne(),
            X_desc_,
            dY_data,
            cudnnTypeWrapper<T>::kOne(),
            X_desc_,
            X_data,
            cudnnTypeWrapper<T>::kZero(),
            X_desc_,
            dscale_data));
        context_.Copy<T, CUDAContext, CUDAContext>(
            dY.size(), dY_data, dbias_data);
      } else {
        dYxX_.ResizeLike(X);
        T* dYxX_data = dYxX_.mutable_data<T>();
        CUDNN_ENFORCE(cudnnOpTensor(
            cudnn_wrapper_.inline_cudnn_handle(),
            mul_desc_,
            cudnnTypeWrapper<T>::kOne(),
            X_desc_,
            dY_data,
            cudnnTypeWrapper<T>::kOne(),
            X_desc_,
            X_data,
            cudnnTypeWrapper<T>::kZero(),
            X_desc_,
            dYxX_data));
#if CUDNN_VERSION_MIN(6, 0, 0)
        ComputeScaleBiasGradient<T>(
            dYxX_data, dY_data, dscale_data, dbias_data);
#else
        const int N = X.dim32(0);
        const int C =
            order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
        const int HxW = X.size() / (N * C);
        ComputeScaleBiasGradientFallback<T>(
            N, C, HxW, dYxX_data, dY_data, dscale_data, dbias_data);
#endif
      }
    }
    return true;
  }

 private:
#if CUDNN_VERSION_MIN(6, 0, 0)
  template <typename T>
  void
  ComputeScaleBiasGradient(const T* dYxX, const T* dY, T* dscale, T* dbias) {
    const cudnnDataType_t cudnn_type = cudnnTypeWrapper<T>::type;
    CUDNN_ENFORCE(cudnnSetReduceTensorDescriptor(
        reduce_desc_,
        CUDNN_REDUCE_TENSOR_ADD,
        cudnn_type,
        CUDNN_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES,
        CUDNN_32BIT_INDICES));
    std::size_t workspace_size = 0;
    CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        reduce_desc_,
        X_desc_,
        scale_desc_,
        &workspace_size));
    workspace_buff_.Resize((workspace_size + sizeof(T) - 1) / sizeof(T));
    T* workspace_data = workspace_buff_.mutable_data<T>();
    CUDNN_ENFORCE(cudnnReduceTensor(
        cudnn_wrapper_.inline_cudnn_handle(),
        reduce_desc_,
        nullptr,
        0,
        workspace_data,
        workspace_size,
        cudnnTypeWrapper<T>::kOne(),
        X_desc_,
        dYxX,
        cudnnTypeWrapper<T>::kZero(),
        scale_desc_,
        dscale));
    CUDNN_ENFORCE(cudnnReduceTensor(
        cudnn_wrapper_.inline_cudnn_handle(),
        reduce_desc_,
        nullptr,
        0,
        workspace_data,
        workspace_size,
        cudnnTypeWrapper<T>::kOne(),
        X_desc_,
        dY,
        cudnnTypeWrapper<T>::kZero(),
        scale_desc_,
        dbias));
  }
#else
  template <typename T>
  void ComputeScaleBiasGradientFallback(
      const int N,
      const int C,
      const int HxW,
      const T* dYxX,
      const T* dY,
      T* dscale,
      T* dbias) {
    if (order_ == StorageOrder::NCHW) {
      std::array<int, 3> dims = {N, C, HxW};
      std::array<int, 2> axes = {0, 2};
      math::ReduceSum<T, CUDAContext>(
          3, dims.data(), 2, axes.data(), dYxX, dscale, &context_);
      math::ReduceSum<T, CUDAContext>(
          3, dims.data(), 2, axes.data(), dY, dbias, &context_);
    } else {
      std::array<int, 2> dims = {N * HxW, C};
      const int axis = 0;
      math::ReduceSum<T, CUDAContext>(
          2, dims.data(), 1, &axis, dYxX, dscale, &context_);
      math::ReduceSum<T, CUDAContext>(
          2, dims.data(), 1, &axis, dY, dbias, &context_);
    }
  }
#endif

  Tensor<CUDAContext> dYxX_;

#if CUDNN_VERSION_MIN(6, 0, 0)
  cudnnReduceTensorDescriptor_t reduce_desc_;

  Tensor<CUDAContext> workspace_buff_;
#endif
};

} // namespace

REGISTER_CUDNN_OPERATOR(AffineChannel, CuDNNAffineChannelOp);
REGISTER_CUDNN_OPERATOR(AffineChannelGradient, CuDNNAffineChannelGradientOp);

} // namespace caffe2
