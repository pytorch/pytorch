#include "caffe2/operators/pool_op.h"

#include <algorithm>
#include <array>
#include <type_traits>
#include <vector>

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"

namespace caffe2 {

namespace {

void SetTensorDescriptor(
    const cudnnDataType_t data_type,
    const StorageOrder order,
    const std::vector<std::int64_t>& dims,
    cudnnTensorDescriptor_t* desc) {
  const int ndim = dims.size();
  const int N = dims[0];
  const int C = order == StorageOrder::NCHW ? dims[1] : dims[ndim - 1];
  switch (ndim) {
    case 4: {
      const int H = order == StorageOrder::NCHW ? dims[2] : dims[1];
      const int W = order == StorageOrder::NCHW ? dims[3] : dims[2];
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          *desc, GetCudnnTensorFormat(order), data_type, N, C, H, W));
      break;
    }
    case 5: {
      const int D = order == StorageOrder::NCHW ? dims[2] : dims[1];
      const int H = order == StorageOrder::NCHW ? dims[3] : dims[2];
      const int W = order == StorageOrder::NCHW ? dims[4] : dims[3];
      const std::array<int, 5> dims_arr = {N, C, D, H, W};
      const std::array<int, 5> strides_arr = order == StorageOrder::NCHW
          ? std::array<int, 5>{C * D * H * W, D * H * W, H * W, W, 1}
          : std::array<int, 5>{D * H * W * C, 1, H * W * C, W * C, C};
      CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
          *desc, data_type, 5, dims_arr.data(), strides_arr.data()));
      break;
    }
    default: {
      CAFFE_THROW("Unsupported tensor dim: ", ndim);
      break;
    }
  }
}

template <class Functor>
class CuDNNPoolOp final : public ConvPoolOpBase<CUDAContext> {
 public:
  template <class... Args>
  explicit CuDNNPoolOp(Args&&... args)
      : ConvPoolOpBase<CUDAContext>(std::forward<Args>(args)...),
        cudnn_wrapper_(&context_),
        functor_(*this),
        equal_padding_(std::equal(
            pads_.cbegin(),
            pads_.cbegin() + kernel_.size(),
            pads_.cbegin() + kernel_.size())) {
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&X_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&Y_desc_));
    CUDNN_ENFORCE(cudnnCreatePoolingDescriptor(&pooling_desc_));
    if (!global_pooling_ && equal_padding_) {
      if (kernel_.size() == 2) {
        CUDNN_ENFORCE(cudnnSetPooling2dDescriptor(
            pooling_desc_,
            functor_.GetPoolingMode(),
            CUDNN_NOT_PROPAGATE_NAN,
            kernel_h(),
            kernel_w(),
            pad_t(),
            pad_l(),
            stride_h(),
            stride_w()));
      } else if (kernel_.size() == 3) {
        CUDNN_ENFORCE(cudnnSetPoolingNdDescriptor(
            pooling_desc_,
            functor_.GetPoolingMode(),
            CUDNN_NOT_PROPAGATE_NAN,
            kernel_.size(),
            kernel_.data(),
            pads_.data(),
            stride_.data()));
      }
    }
  }

  ~CuDNNPoolOp() override {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(X_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(Y_desc_));
    CUDNN_ENFORCE(cudnnDestroyPoolingDescriptor(pooling_desc_));
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);
    const int ndim = X.dim();
    const int N = X.dim32(0);
    const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
    auto sizes = ConvPoolOpBase<CUDAContext>::GetOutputSize(X, C);
    auto* Y = Output(0, sizes, at::dtype<T>());
    const T* X_data = X.template data<T>();
    T* Y_data = Y->template mutable_data<T>();

    if (N == 0) {
      return true;
    }

    if (global_pooling_) {
      const int HxW = X.numel() / (N * C);
      if (order_ == StorageOrder::NCHW) {
        return functor_.template GlobalPoolingForward<T, StorageOrder::NCHW>(
            N, C, HxW, X_data, Y_data, &context_);
      } else {
        return functor_.template GlobalPoolingForward<T, StorageOrder::NHWC>(
            N, C, HxW, X_data, Y_data, &context_);
      }
    }

    const std::vector<int> X_HW_dims = GetDims(X);
    const std::vector<int> Y_HW_dims = GetDims(*Y);
    if (order_ == StorageOrder::NHWC) {
      // CuDNN Pooling on NHWC order is very slow, fallback to CUDA
      // implementation.
      return functor_.template Forward<T, StorageOrder::NHWC>(
          N,
          C,
          X_HW_dims,
          Y_HW_dims,
          kernel_,
          dilation_,
          stride_,
          pads_,
          X.template data<T>(),
          Y->template mutable_data<T>(),
          &context_);
    } else if (!equal_padding_ || ndim == 3) {
      return functor_.template Forward<T, StorageOrder::NCHW>(
          N,
          C,
          X_HW_dims,
          Y_HW_dims,
          kernel_,
          dilation_,
          stride_,
          pads_,
          X.template data<T>(),
          Y->template mutable_data<T>(),
          &context_);
    }

    const std::vector<std::int64_t> X_dims = X.sizes().vec();
    const std::vector<std::int64_t> Y_dims = Y->sizes().vec();
    if (cached_X_dims_ != X_dims) {
      constexpr cudnnDataType_t data_type = cudnnTypeWrapper<T>::type;
      SetTensorDescriptor(data_type, order_, X_dims, &X_desc_);
      SetTensorDescriptor(data_type, order_, Y_dims, &Y_desc_);
      cached_X_dims_ = X_dims;
    }
    CUDNN_ENFORCE(cudnnPoolingForward(
        cudnn_wrapper_.inline_cudnn_handle(),
        pooling_desc_,
        cudnnTypeWrapper<T>::kOne(),
        X_desc_,
        X_data,
        cudnnTypeWrapper<T>::kZero(),
        Y_desc_,
        Y_data));

    return true;
  }

 private:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t X_desc_;
  cudnnTensorDescriptor_t Y_desc_;
  cudnnPoolingDescriptor_t pooling_desc_;

  const Functor functor_;

  const bool equal_padding_;
  std::vector<std::int64_t> cached_X_dims_;
};

template <class Functor>
class CuDNNPoolGradientOp final : public ConvPoolOpBase<CUDAContext> {
 public:
  template <class... Args>
  explicit CuDNNPoolGradientOp(Args&&... args)
      : ConvPoolOpBase<CUDAContext>(std::forward<Args>(args)...),
        cudnn_wrapper_(&context_),
        functor_(*this),
        equal_padding_(std::equal(
            pads_.cbegin(),
            pads_.cbegin() + kernel_.size(),
            pads_.cbegin() + kernel_.size())) {
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&X_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&Y_desc_));
    CUDNN_ENFORCE(cudnnCreatePoolingDescriptor(&pooling_desc_));
    if (!global_pooling_ && equal_padding_) {
      if (kernel_.size() == 2) {
        CUDNN_ENFORCE(cudnnSetPooling2dDescriptor(
            pooling_desc_,
            functor_.GetPoolingMode(),
            CUDNN_NOT_PROPAGATE_NAN,
            kernel_h(),
            kernel_w(),
            pad_t(),
            pad_l(),
            stride_h(),
            stride_w()));
      } else if (kernel_.size() == 3) {
        CUDNN_ENFORCE(cudnnSetPoolingNdDescriptor(
            pooling_desc_,
            functor_.GetPoolingMode(),
            CUDNN_NOT_PROPAGATE_NAN,
            kernel_.size(),
            kernel_.data(),
            pads_.data(),
            stride_.data()));
      }
    }
  }

  ~CuDNNPoolGradientOp() override {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(X_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(Y_desc_));
    CUDNN_ENFORCE(cudnnDestroyPoolingDescriptor(pooling_desc_));
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);
    const auto& Y = Input(1);
    const auto& dY = Input(2);
    auto* dX = Output(0, X.sizes(), at::dtype<T>());
    const int ndim = X.dim();
    const int N = X.dim32(0);
    const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
    const std::vector<int> X_HW_dims = GetDims(X);
    const std::vector<int> Y_HW_dims = GetDims(Y);
    ConvPoolOpBase<CUDAContext>::ComputePads(X_HW_dims);
    const T* dY_data = dY.template data<T>();
    const T* X_data = X.template data<T>();
    const T* Y_data = Y.template data<T>();
    T* dX_data = dX->template mutable_data<T>();

    if (N == 0) {
      return true;
    }

    if (global_pooling_) {
      const int HxW = X.numel() / (N * C);
      if (order_ == StorageOrder::NCHW) {
        return functor_.template GlobalPoolingBackward<T, StorageOrder::NCHW>(
            N, C, HxW, dY_data, X_data, Y_data, dX_data, &context_);
      } else {
        return functor_.template GlobalPoolingBackward<T, StorageOrder::NHWC>(
            N, C, HxW, dY_data, X_data, Y_data, dX_data, &context_);
      }
    }

    if (order_ == StorageOrder::NHWC) {
      // CuDNN Pooling on NHWC order is very slow, fallback to CUDA
      // implementation.
      return functor_.template Backward<T, StorageOrder::NHWC>(
          N,
          C,
          X_HW_dims,
          Y_HW_dims,
          kernel_,
          dilation_,
          stride_,
          pads_,
          dY_data,
          X_data,
          Y_data,
          dX_data,
          &context_);
    } else if (!equal_padding_ || ndim == 3) {
      return functor_.template Backward<T, StorageOrder::NCHW>(
          N,
          C,
          X_HW_dims,
          Y_HW_dims,
          kernel_,
          dilation_,
          stride_,
          pads_,
          dY_data,
          X_data,
          Y_data,
          dX_data,
          &context_);
    }

    const std::vector<std::int64_t> X_dims = X.sizes().vec();
    const std::vector<std::int64_t> Y_dims = Y.sizes().vec();
    if (cached_X_dims_ != X_dims) {
      constexpr cudnnDataType_t data_type = cudnnTypeWrapper<T>::type;
      SetTensorDescriptor(data_type, order_, X_dims, &X_desc_);
      SetTensorDescriptor(data_type, order_, Y_dims, &Y_desc_);
      cached_X_dims_ = X_dims;
    }
    CUDNN_ENFORCE(cudnnPoolingBackward(
        cudnn_wrapper_.inline_cudnn_handle(),
        pooling_desc_,
        cudnnTypeWrapper<T>::kOne(),
        Y_desc_,
        Y_data,
        Y_desc_,
        dY_data,
        X_desc_,
        X_data,
        cudnnTypeWrapper<T>::kZero(),
        X_desc_,
        dX_data));

    return true;
  }

 private:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t X_desc_;
  cudnnTensorDescriptor_t Y_desc_;
  cudnnPoolingDescriptor_t pooling_desc_;

  const Functor functor_;

  const bool equal_padding_;
  std::vector<std::int64_t> cached_X_dims_;
};

struct CuDNNAveragePoolFunctor {
  explicit CuDNNAveragePoolFunctor(const OperatorBase& op)
      : avg_pool_functor(op) {}

  cudnnPoolingMode_t GetPoolingMode() const {
    return avg_pool_functor.count_include_pad
        ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
        : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  }

  template <typename T, StorageOrder kOrder>
  bool GlobalPoolingForward(
      const int N,
      const int C,
      const int HxW,
      const T* X,
      T* Y,
      CUDAContext* context) const {
      return avg_pool_functor.GlobalPoolingForward<T, kOrder>(
          N, C, HxW, X, Y, context);
  }

  template <typename T, StorageOrder kOrder>
  bool Forward(
      const int N,
      const int C,
      const std::vector<int>& X_dims,
      const std::vector<int>& Y_dims,
      const std::vector<int>& kernel,
      const std::vector<int>& dilation,
      const std::vector<int>& stride,
      const std::vector<int>& pads,
      const T* X,
      T* Y,
      CUDAContext* context) const {
      return avg_pool_functor.Forward<T, kOrder>(
          N, C, X_dims, Y_dims, kernel, dilation, stride, pads, X, Y, context);
  }

  template <typename T, StorageOrder kOrder>
  bool GlobalPoolingBackward(
      const int N,
      const int C,
      const int HxW,
      const T* dY,
      const T* X,
      const T* Y,
      T* dX,
      CUDAContext* context) const {
      return avg_pool_functor.GlobalPoolingBackward<T, kOrder>(
          N, C, HxW, dY, X, Y, dX, context);
  }

  template <typename T, StorageOrder kOrder>
  bool Backward(
      const int N,
      const int C,
      const std::vector<int>& X_dims,
      const std::vector<int>& Y_dims,
      const std::vector<int>& kernel,
      const std::vector<int>& dilation,
      const std::vector<int>& stride,
      const std::vector<int>& pads,
      const T* dY,
      const T* X,
      const T* Y,
      T* dX,
      CUDAContext* context) const {
      return avg_pool_functor.Backward<T, kOrder>(
          N,
          C,
          X_dims,
          Y_dims,
          kernel,
          dilation,
          stride,
          pads,
          dY,
          X,
          Y,
          dX,
          context);
  }

  const AveragePoolFunctor<CUDAContext> avg_pool_functor;
};

struct CuDNNMaxPoolFunctor {
  explicit CuDNNMaxPoolFunctor(const OperatorBase& op)
      : max_pool_functor(op),
        deterministic(op.GetSingleArgument<bool>("deterministic", false)) {}

  cudnnPoolingMode_t GetPoolingMode() const {
    return deterministic ? CUDNN_POOLING_MAX_DETERMINISTIC : CUDNN_POOLING_MAX;
  }

  template <typename T, StorageOrder kOrder>
  bool GlobalPoolingForward(
      const int N,
      const int C,
      const int HxW,
      const T* X,
      T* Y,
      CUDAContext* context) const {
      return max_pool_functor.GlobalPoolingForward<T, kOrder>(
          N, C, HxW, X, Y, context);
  }

  template <typename T, StorageOrder kOrder>
  bool Forward(
      const int N,
      const int C,
      const std::vector<int>& X_dims,
      const std::vector<int>& Y_dims,
      const std::vector<int>& kernel,
      const std::vector<int>& dilation,
      const std::vector<int>& stride,
      const std::vector<int>& pads,
      const T* X,
      T* Y,
      CUDAContext* context) const {
      return max_pool_functor.Forward<T, kOrder>(
          N, C, X_dims, Y_dims, kernel, dilation, stride, pads, X, Y, context);
  }

  template <typename T, StorageOrder kOrder>
  bool GlobalPoolingBackward(
      const int N,
      const int C,
      const int HxW,
      const T* dY,
      const T* X,
      const T* Y,
      T* dX,
      CUDAContext* context) const {
      return max_pool_functor.GlobalPoolingBackward<T, kOrder>(
          N, C, HxW, dY, X, Y, dX, context);
  }

  template <typename T, StorageOrder kOrder>
  bool Backward(
      const int N,
      const int C,
      const std::vector<int>& X_dims,
      const std::vector<int>& Y_dims,
      const std::vector<int>& kernel,
      const std::vector<int>& dilation,
      const std::vector<int>& stride,
      const std::vector<int>& pads,
      const T* dY,
      const T* X,
      const T* Y,
      T* dX,
      CUDAContext* context) const {
      return max_pool_functor.Backward<T, kOrder>(
          N,
          C,
          X_dims,
          Y_dims,
          kernel,
          dilation,
          stride,
          pads,
          dY,
          X,
          Y,
          dX,
          context);
  }

  const MaxPoolFunctor<CUDAContext> max_pool_functor;
  const bool deterministic;
};

} // namespace

REGISTER_CUDNN_OPERATOR(AveragePool, CuDNNPoolOp<CuDNNAveragePoolFunctor>);
REGISTER_CUDNN_OPERATOR(
    AveragePoolGradient,
    CuDNNPoolGradientOp<CuDNNAveragePoolFunctor>);

REGISTER_CUDNN_OPERATOR(AveragePool1D, CuDNNPoolOp<CuDNNAveragePoolFunctor>);
REGISTER_CUDNN_OPERATOR(
    AveragePool1DGradient,
    CuDNNPoolGradientOp<CuDNNAveragePoolFunctor>);

REGISTER_CUDNN_OPERATOR(AveragePool2D, CuDNNPoolOp<CuDNNAveragePoolFunctor>);
REGISTER_CUDNN_OPERATOR(
    AveragePool2DGradient,
    CuDNNPoolGradientOp<CuDNNAveragePoolFunctor>);

REGISTER_CUDNN_OPERATOR(AveragePool3D, CuDNNPoolOp<CuDNNAveragePoolFunctor>);
REGISTER_CUDNN_OPERATOR(
    AveragePool3DGradient,
    CuDNNPoolGradientOp<CuDNNAveragePoolFunctor>);

REGISTER_CUDNN_OPERATOR(MaxPool, CuDNNPoolOp<CuDNNMaxPoolFunctor>);
REGISTER_CUDNN_OPERATOR(
    MaxPoolGradient,
    CuDNNPoolGradientOp<CuDNNMaxPoolFunctor>);

REGISTER_CUDNN_OPERATOR(MaxPool1D, CuDNNPoolOp<CuDNNMaxPoolFunctor>);
REGISTER_CUDNN_OPERATOR(
    MaxPool1DGradient,
    CuDNNPoolGradientOp<CuDNNMaxPoolFunctor>);

REGISTER_CUDNN_OPERATOR(MaxPool2D, CuDNNPoolOp<CuDNNMaxPoolFunctor>);
REGISTER_CUDNN_OPERATOR(
    MaxPool2DGradient,
    CuDNNPoolGradientOp<CuDNNMaxPoolFunctor>);

REGISTER_CUDNN_OPERATOR(MaxPool3D, CuDNNPoolOp<CuDNNMaxPoolFunctor>);
REGISTER_CUDNN_OPERATOR(
    MaxPool3DGradient,
    CuDNNPoolGradientOp<CuDNNMaxPoolFunctor>);

} // namespace caffe2
