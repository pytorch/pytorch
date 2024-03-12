#include "caffe2/operators/transpose_op.h"

#include <algorithm>
#include <limits>
#include <type_traits>
#include <vector>

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

class CuDNNTransposeOp final : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  template <class... Args>
  explicit CuDNNTransposeOp(Args&&... args)
      : Operator<CUDAContext>(std::forward<Args>(args)...),
        cudnn_wrapper_(&context_),
        axes_(OperatorBase::GetRepeatedArgument<int>("axes")) {
    // Checks the legality of axes_: it should be from 0 to axes_.size().
    std::vector<int> axes_sorted(axes_);
    std::sort(axes_sorted.begin(), axes_sorted.end());
    for (std::size_t i = 0; i < axes_sorted.size(); ++i) {
      if (axes_sorted[i] != i) {
        CAFFE_THROW("Axes should be a permutation of 0 to ndim.");
      }
    }

    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&X_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&Y_desc_));
  }

  ~CuDNNTransposeOp() override {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(X_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(Y_desc_));
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, int>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);
    const int ndim = X.dim();
    if (axes_.empty()) {
      axes_.resize(ndim);
      std::iota(axes_.rbegin(), axes_.rend(), 0);
    } else {
      CAFFE_ENFORCE_EQ(axes_.size(), ndim);
    }
    std::vector<std::int64_t> X_dims = X.sizes().vec();
    std::vector<std::int64_t> Y_dims(ndim);
    for (int i = 0; i < ndim; ++i) {
      Y_dims[i] = X_dims[axes_[i]];
    }
    auto* Y = Output(0, Y_dims, at::dtype<T>());
    const T* X_data = X.template data<T>();
    T* Y_data = Y->template mutable_data<T>();
    if (X.numel() == 0) {
      return true;
    }
    if (!IsFloatType<T>() || !IsCuDNNValidTensor(X)) {
      math::Transpose<std::int64_t, T, CUDAContext>(
          ndim, X_dims.data(), axes_.data(), X_data, Y_data, &context_);
      return true;
    }
    if (cudnnTypeWrapper<T>::type != cached_dtype_ ||
        X_dims != cached_X_dims_) {
      SetTensorDescriptor(cudnnTypeWrapper<T>::type, X_dims, Y_dims);
      cached_dtype_ = cudnnTypeWrapper<T>::type;
      cached_X_dims_ = X_dims;
    }
    CUDNN_ENFORCE(cudnnTransformTensor(
        cudnn_wrapper_.inline_cudnn_handle(),
        cudnnTypeWrapper<T>::kOne(),
        X_desc_,
        X_data,
        cudnnTypeWrapper<T>::kZero(),
        Y_desc_,
        Y_data));
    return true;
  }

 private:
  template <typename T>
  constexpr bool IsFloatType() const {
    return std::is_same<T, float>::value || std::is_same<T, double>::value ||
        std::is_same<T, at::Half>::value;
  }

  bool IsCuDNNValidTensor(const Tensor& X) const {
    const int ndim = X.dim();
    return ndim >= 3 && ndim <= CUDNN_DIM_MAX &&
        X.numel() < std::numeric_limits<int32_t>::max();
  }

  void SetTensorDescriptor(
      const cudnnDataType_t data_type,
      const std::vector<std::int64_t>& X_dims,
      const std::vector<std::int64_t>& Y_dims) {
    const int ndim = X_dims.size();
    std::vector<int> dims(Y_dims.cbegin(), Y_dims.cend());
    std::vector<int> X_strides(ndim);
    std::vector<int> X_buff(ndim);
    std::vector<int> Y_strides(ndim);
    X_buff.back() = 1;
    Y_strides.back() = 1;
    for (int i = ndim - 1; i > 0; --i) {
      X_buff[i - 1] = X_buff[i] * X_dims[i];
      Y_strides[i - 1] = Y_strides[i] * Y_dims[i];
    }
    for (int i = 0; i < ndim; ++i) {
      X_strides[i] = X_buff[axes_[i]];
    }
    CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
        X_desc_, data_type, ndim, dims.data(), X_strides.data()));
    CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
        Y_desc_, data_type, ndim, dims.data(), Y_strides.data()));
  }

  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t X_desc_;
  cudnnTensorDescriptor_t Y_desc_;

  cudnnDataType_t cached_dtype_ = cudnnTypeWrapper<float>::type;
  std::vector<std::int64_t> cached_X_dims_;
  std::vector<std::int32_t> axes_;
};


} // namespace

REGISTER_CUDNN_OPERATOR(Transpose, CuDNNTransposeOp);

} // namespace caffe2
