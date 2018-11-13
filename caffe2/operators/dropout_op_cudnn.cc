#include "caffe2/operators/dropout_op.h"

#include <mutex>
#include <string>
#include <vector>

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"

namespace caffe2 {

// cudnnRestoreDropoutDescriptor is needed for correctness and doesn't exist
// prior to CuDNN v7.
#if CUDNN_VERSION_MIN(7, 0, 0)

namespace {

std::string GetStatesBlobName(const std::string& mask_blob_name) {
  return "cudnn_dropout_states_" + mask_blob_name;
}

class CuDNNDropoutOpBase : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  CuDNNDropoutOpBase(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_),
        OP_SINGLE_ARG(bool, OpSchema::Arg_IsTest, is_test_, false),
        OP_SINGLE_ARG(float, "ratio", ratio_, 0.5f),
        states_initialized_(false),
        random_seed_(operator_def.device_option().random_seed()) {
    CAFFE_ENFORCE_GE(ratio_, 0);
    CAFFE_ENFORCE_LT(ratio_, 1);

    if (!is_test_) {
      CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&data_desc_));
      CUDNN_ENFORCE(cudnnCreateDropoutDescriptor(&dropout_desc_));
      CUDNN_ENFORCE(cudnnDropoutGetStatesSize(
          cudnn_wrapper_.inline_cudnn_handle(), &states_size_in_bytes_));
    }
  }

  virtual ~CuDNNDropoutOpBase() {
    if (!is_test_) {
      CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(data_desc_));
      CUDNN_ENFORCE(cudnnDestroyDropoutDescriptor(dropout_desc_));
    }
  }

 protected:
  void SetUpDataDesc(const cudnnDataType_t type, const std::int64_t N) {
    if (cached_data_numel_ != N) {
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          data_desc_,
          GetCudnnTensorFormat(StorageOrder::NCHW),
          type,
          1,
          1,
          1,
          N));
      cached_data_numel_ = N;
      CUDNN_ENFORCE(cudnnDropoutGetReserveSpaceSize(
          data_desc_, &reserve_space_size_in_bytes_));
    }
  }

  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t data_desc_;
  cudnnDropoutDescriptor_t dropout_desc_;

  const bool is_test_;
  const float ratio_;

  // track whether states have been initialized - only needs to happen once
  bool states_initialized_;
  std::size_t states_size_in_bytes_;
  std::size_t reserve_space_size_in_bytes_;

  const std::uint64_t random_seed_;
  std::int64_t cached_data_numel_;

  Blob* states_blob_ = nullptr;
  Tensor* states_ = nullptr;
};

class CuDNNDropoutOp final : public CuDNNDropoutOpBase {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  CuDNNDropoutOp(const OperatorDef& operator_def, Workspace* ws)
      : CuDNNDropoutOpBase(operator_def, ws) {
    if (!is_test_) {
      states_blob_ = ws->CreateBlob(GetStatesBlobName(operator_def.output(1)));
      CAFFE_ENFORCE(states_blob_ != nullptr);
      states_ = BlobGetMutableTensor(states_blob_, CUDA);
      CAFFE_ENFORCE(states_ != nullptr);
    }
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);
    auto* Y = Output(0);
    Y->ResizeLike(X);
    const int N = X.numel();
    const T* X_data = X.template data<T>();
    T* Y_data = Y->template mutable_data<T>();

    if (is_test_) {
      if (Y != &X) {
        context_.CopySameDevice<T>(N, X_data, Y_data);
      }
      return true;
    }

    auto* mask = Output(1);
    mask->Resize(reserve_space_size_in_bytes_);
    std::uint8_t* mask_data = mask->template mutable_data<std::uint8_t>();
    SetDropoutDescriptor();
    SetUpDataDesc(cudnnTypeWrapper<T>::type, N);
    CUDNN_ENFORCE(cudnnDropoutForward(
        cudnn_wrapper_.inline_cudnn_handle(),
        dropout_desc_,
        data_desc_,
        X_data,
        data_desc_,
        Y_data,
        mask_data,
        reserve_space_size_in_bytes_));
    return true;
  }

 private:
  void SetDropoutDescriptor() {
    if (!states_initialized_) {
      {
        // Need to protect as clashes with NCCL
        std::lock_guard<std::mutex> lock(CUDAContext::mutex());
        states_->Resize(states_size_in_bytes_);
        uint8_t* states_data = states_->template mutable_data<uint8_t>();
        CUDNN_ENFORCE(cudnnSetDropoutDescriptor(
            dropout_desc_,
            cudnn_wrapper_.inline_cudnn_handle(),
            ratio_,
            states_data,
            states_size_in_bytes_,
            random_seed_));
      }
      states_initialized_ = true;
    }
  }
};

class CuDNNDropoutGradientOp final : public CuDNNDropoutOpBase {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  CuDNNDropoutGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : CuDNNDropoutOpBase(operator_def, ws) {
    states_blob_ = ws->GetBlob(GetStatesBlobName(operator_def.input(1)));
    CAFFE_ENFORCE(states_blob_ != nullptr);
    states_ = BlobGetMutableTensor(states_blob_, CUDA);
    CAFFE_ENFORCE(states_ != nullptr);
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& dY = Input(0);
    const auto& mask = Input(1);
    auto* dX = Output(0);
    dX->ResizeLike(dY);
    const int N = dY.numel();
    const T* dY_data = dY.data<T>();
    const std::uint8_t* mask_data = mask.data<std::uint8_t>();
    T* dX_data = dX->mutable_data<T>();
    RestoreDropoutDescriptor();
    SetUpDataDesc(cudnnTypeWrapper<T>::type, N);
    CUDNN_ENFORCE(cudnnDropoutBackward(
        cudnn_wrapper_.inline_cudnn_handle(),
        dropout_desc_,
        data_desc_,
        dY_data,
        data_desc_,
        dX_data,
        const_cast<std::uint8_t*>(mask_data),
        reserve_space_size_in_bytes_));
    return true;
  }

 private:
  void RestoreDropoutDescriptor() {
    if (!states_initialized_) {
      {
        // Need to protect as clashes with NCCL
        std::lock_guard<std::mutex> lock(CUDAContext::mutex());
        std::uint8_t* states_data = states_->mutable_data<std::uint8_t>();
        CUDNN_ENFORCE(cudnnRestoreDropoutDescriptor(
            dropout_desc_,
            cudnn_wrapper_.inline_cudnn_handle(),
            ratio_,
            states_data,
            states_size_in_bytes_,
            random_seed_));
      }
      states_initialized_ = true;
    }
  }
};

} // namespace

REGISTER_CUDNN_OPERATOR(Dropout, CuDNNDropoutOp);
REGISTER_CUDNN_OPERATOR(DropoutGrad, CuDNNDropoutGradientOp);

#endif

}; // namespace caffe2
