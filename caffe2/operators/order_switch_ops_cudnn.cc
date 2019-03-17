#include "caffe2/operators/order_switch_ops.h"

#include <algorithm>
#include <functional>
#include <vector>

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"
#include "caffe2/core/types.h"

namespace caffe2 {

namespace {

class CuDNNOrderSwithOpBase : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  template <class... Args>
  explicit CuDNNOrderSwithOpBase(Args&&... args)
      : Operator<CUDAContext>(std::forward<Args>(args)...),
        cudnn_wrapper_(&context_) {
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&X_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&Y_desc_));
  }

  ~CuDNNOrderSwithOpBase() override {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(X_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(Y_desc_));
  }

 protected:
  // TODO: std::vector<int> -> std::vector<int64_t>
  void SetTensorDescriptor(
      const cudnnDataType_t data_type,
      const StorageOrder order,
      const std::vector<int>& data_dims,
      cudnnTensorDescriptor_t data_desc) const {
    const int ndim = data_dims.size();
    const int N = data_dims[0];
    const int C = order == StorageOrder::NCHW ? data_dims[1] : data_dims.back();
    if (ndim == 3) {
      const int H = 1;
      const int W = order == StorageOrder::NCHW ? data_dims[2] : data_dims[1];
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          data_desc, GetCudnnTensorFormat(order), data_type, N, C, H, W));
    } else if (ndim == 4) {
      const int H = order == StorageOrder::NCHW ? data_dims[2] : data_dims[1];
      const int W = order == StorageOrder::NCHW ? data_dims[3] : data_dims[2];
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          data_desc, GetCudnnTensorFormat(order), data_type, N, C, H, W));
    } else {
      const int H = order == StorageOrder::NCHW ? data_dims[2] : data_dims[1];
      const int W = order == StorageOrder::NCHW ? data_dims[3] : data_dims[2];
      const auto l_iter = order == StorageOrder::NCHW ? data_dims.cbegin() + 4
                                                      : data_dims.cbegin() + 3;
      const auto r_iter =
          order == StorageOrder::NCHW ? data_dims.cend() : data_dims.cend() - 1;
      const int D = std::accumulate(l_iter, r_iter, 1, std::multiplies<int>());
      const std::array<int, 5> dims = {N, C, H, W, D};
      const std::array<int, 5> strides = order == StorageOrder::NCHW
          ? std::array<int, 5>{C * H * W * D, H * W * D, W * D, D, 1}
          : std::array<int, 5>{C * H * W * D, 1, W * D * C, D * C, C};
      CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
          data_desc, data_type, 5, dims.data(), strides.data()));
    }
  }

  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t X_desc_;
  cudnnTensorDescriptor_t Y_desc_;

  std::vector<int> cached_X_dims_;
};

class CuDNNNHWC2NCHWOp final : public CuDNNOrderSwithOpBase {
 public:
  template <class... Args>
  explicit CuDNNNHWC2NCHWOp(Args&&... args)
      : CuDNNOrderSwithOpBase(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);

    const int ndim = X.dim();
    const int N = X.dim32(0);
    const int C = X.dim32(ndim - 1);
    const std::vector<int> X_dims(X.sizes().cbegin(), X.sizes().cend());
    std::vector<int> Y_dims(ndim);
    Y_dims[0] = N;
    Y_dims[1] = C;
    std::copy(X_dims.cbegin() + 1, X_dims.cend() - 1, Y_dims.begin() + 2);
    std::vector<int64_t> Y_dims_64;
    std::copy(Y_dims.cbegin(), Y_dims.cend(), std::back_inserter(Y_dims_64));
    auto* Y = Output(0, Y_dims_64, at::dtype<T>());
    if (cached_X_dims_ != X_dims) {
      cached_X_dims_ = X_dims;
      SetTensorDescriptor(
          cudnnTypeWrapper<T>::type, StorageOrder::NHWC, X_dims, X_desc_);
      SetTensorDescriptor(
          cudnnTypeWrapper<T>::type, StorageOrder::NCHW, Y_dims, Y_desc_);
    }
    CUDNN_ENFORCE(cudnnTransformTensor(
        cudnn_wrapper_.inline_cudnn_handle(),
        cudnnTypeWrapper<T>::kOne(),
        X_desc_,
        X.template data<T>(),
        cudnnTypeWrapper<T>::kZero(),
        Y_desc_,
        Y->template mutable_data<T>()));
    return true;
  }
};

class CuDNNNCHW2NHWCOp final : public CuDNNOrderSwithOpBase {
 public:
  template <class... Args>
  explicit CuDNNNCHW2NHWCOp(Args&&... args)
      : CuDNNOrderSwithOpBase(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);

    const int ndim = X.dim();
    const int N = X.dim32(0);
    const int C = X.dim32(1);
    const std::vector<int> X_dims(X.sizes().cbegin(), X.sizes().cend());
    std::vector<int> Y_dims(ndim);
    Y_dims[0] = N;
    Y_dims[ndim - 1] = C;
    std::copy(X_dims.cbegin() + 2, X_dims.cend(), Y_dims.begin() + 1);
    std::vector<int64_t> Y_dims_64;
    std::copy(Y_dims.cbegin(), Y_dims.cend(), std::back_inserter(Y_dims_64));
    auto* Y = Output(0, Y_dims_64, at::dtype<T>());
    if (cached_X_dims_ != X_dims) {
      cached_X_dims_ = X_dims;
      SetTensorDescriptor(
          cudnnTypeWrapper<T>::type, StorageOrder::NCHW, X_dims, X_desc_);
      SetTensorDescriptor(
          cudnnTypeWrapper<T>::type, StorageOrder::NHWC, Y_dims, Y_desc_);
    }
    CUDNN_ENFORCE(cudnnTransformTensor(
        cudnn_wrapper_.inline_cudnn_handle(),
        cudnnTypeWrapper<T>::kOne(),
        X_desc_,
        X.template data<T>(),
        cudnnTypeWrapper<T>::kZero(),
        Y_desc_,
        Y->template mutable_data<T>()));
    return true;
  }
};

} // namespace

REGISTER_CUDNN_OPERATOR(NHWC2NCHW, CuDNNNHWC2NCHWOp);
REGISTER_CUDNN_OPERATOR(NCHW2NHWC, CuDNNNCHW2NHWCOp);

} // namespace caffe2
