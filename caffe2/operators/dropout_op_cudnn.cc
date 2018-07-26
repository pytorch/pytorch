#include "caffe2/core/context_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"

namespace caffe2 {

// cudnnRestoreDropoutDescriptor is needed for correctness and
// doesn't exist prior to cuDNN v7
#if CUDNN_VERSION_MIN(7,0,0)

class CuDNNDropoutOp final : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  CuDNNDropoutOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_),
        ratio_(OperatorBase::GetSingleArgument<float>("ratio", 0.5)),
        is_test_(
            OperatorBase::GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)),
        states_initialized_(false),
        random_seed_(operator_def.device_option().random_seed()) {
    CAFFE_ENFORCE_GE(ratio_, 0);
    CAFFE_ENFORCE_LT(ratio_, 1);
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&data_desc_));

    CUDNN_ENFORCE(cudnnCreateDropoutDescriptor(&dropout_desc_));
    CUDNN_ENFORCE(cudnnDropoutGetStatesSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        reinterpret_cast<size_t*>(&states_size_in_bytes_)));

    if (!is_test_) {
      scratch_blob_ = ws->CreateBlob(scratch_blob_name(operator_def.output(1)));
      CAFFE_ENFORCE(scratch_blob_);
    }
  }

  ~CuDNNDropoutOp() noexcept {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_ENFORCE(cudnnDestroyDropoutDescriptor(dropout_desc_));
  }

  template <typename T, typename M>
  bool DoRunWithType();

  bool RunOnDevice() override;

  static string scratch_blob_name(string mask_blob_name) {
    return "cudnn_dropout_scratch_" + mask_blob_name;
  }

 protected:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t data_desc_;
  cudnnDropoutDescriptor_t dropout_desc_;

  vector<TIndex> cudnn_input_dims_;

  float ratio_;
  bool is_test_;

  Blob* scratch_blob_ = nullptr;

  size_t states_size_in_bytes_, reserve_space_size_in_bytes_;
  // Input: X, Output: Y, mask_and_states

  // track whether states have been initialized - only needs to happen once
  bool states_initialized_;

  // random seed
  unsigned long long random_seed_;
};

class CuDNNDropoutGradientOp final : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  CuDNNDropoutGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_),
        ratio_(OperatorBase::GetSingleArgument<float>("ratio", 0.5)),
        is_test_(
            OperatorBase::GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)),
        states_initialized_(false),
        random_seed_(operator_def.device_option().random_seed()) {
    CAFFE_ENFORCE_GE(ratio_, 0);
    CAFFE_ENFORCE_LT(ratio_, 1);
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&data_desc_));

    CUDNN_ENFORCE(cudnnCreateDropoutDescriptor(&dropout_desc_));
    CUDNN_ENFORCE(cudnnDropoutGetStatesSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        reinterpret_cast<size_t*>(&states_size_in_bytes_)));

    // Share scratch with the forward op
    scratch_blob_ =
        ws->GetBlob(CuDNNDropoutOp::scratch_blob_name(operator_def.input(1)));
    CAFFE_ENFORCE(scratch_blob_);
  }

  ~CuDNNDropoutGradientOp() noexcept {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_ENFORCE(cudnnDestroyDropoutDescriptor(dropout_desc_));
  }

  template <typename T, typename M>
  bool DoRunWithType();

  bool RunOnDevice() override;

 protected:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t data_desc_;
  cudnnDropoutDescriptor_t dropout_desc_;

  vector<TIndex> cudnn_input_dims_;

  Blob* scratch_blob_;

  float ratio_;
  bool is_test_;

  size_t states_size_in_bytes_, reserve_space_size_in_bytes_;
  // Input: dY, mask_and_states, Output: dX

  // only need to initialize states once (size is static)
  bool states_initialized_;

  unsigned long long random_seed_;
};

template <typename T, typename M>
bool CuDNNDropoutOp::DoRunWithType() {
  const auto& X = Input(0);
  auto* Y = Output(0);

  auto size_prod = 1;
  for (auto dim : X.dims()) {
    size_prod *= dim;
  }
  // now actually run the computation
  if (is_test_) {
    if (Y != &X) {
      context_.Copy<T, CUDAContext, CUDAContext>(
          X.size(), X.template data<T>(), Y->template mutable_data<T>());
    }
    return true;
  } else {
    auto* mask = Output(1);
    // Reshape tensor descriptors if necessary
    if (X.dims() != cudnn_input_dims_ && !is_test_) {
      CAFFE_ENFORCE(scratch_blob_);
      Tensor<CUDAContext>* states =
          scratch_blob_->GetMutable<Tensor<CUDAContext>>();
      cudnn_input_dims_ = X.dims();
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          data_desc_,
          GetCudnnTensorFormat(StorageOrder::NCHW),
          cudnnTypeWrapper<T>::type,
          size_prod,
          1,
          1,
          1));

      // get the reserve space we need
      CUDNN_ENFORCE(cudnnDropoutGetReserveSpaceSize(
          data_desc_, &reserve_space_size_in_bytes_));

      mask->Resize(reserve_space_size_in_bytes_);
      states->Resize(states_size_in_bytes_);

      if (!states_initialized_) {
        // set the dropout descriptor (note: need to allocate the states data
        // before acquiring the mutex)
        uint8_t* states_data = states->mutable_data<uint8_t>();
        {
          // Need to protect  as clashes with NCCL
          std::lock_guard<std::mutex> lk(CUDAContext::mutex());
          CUDNN_ENFORCE(cudnnSetDropoutDescriptor(
              dropout_desc_,
              cudnn_wrapper_.inline_cudnn_handle(),
              ratio_,
              states_data,
              states_size_in_bytes_,
              random_seed_
              ));
        }
        states_initialized_ = true;
      }
    }
    CUDNN_ENFORCE(cudnnDropoutForward(
        cudnn_wrapper_.inline_cudnn_handle(),
        dropout_desc_,
        data_desc_,
        X.template data<T>(),
        data_desc_,
        Y->template mutable_data<T>(),
        mask->mutable_data<uint8_t>(),
        reserve_space_size_in_bytes_));
  }
  return true;
}

bool CuDNNDropoutOp::RunOnDevice() {
  // dispatch based on contents of tensor(s)
  const auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);

  if (X.IsType<float>()) {
    return DoRunWithType<float, float>();
  } else if (X.IsType<float16>()) {
    return DoRunWithType<float16, float>();
  }
  return false;
}

template <typename T, typename M>
bool CuDNNDropoutGradientOp::DoRunWithType() {
  const auto& dY = Input(0);
  const auto& mask = Input(1);
  const Tensor<CUDAContext>& states = scratch_blob_->Get<Tensor<CUDAContext>>();
  auto* dX = Output(0);

  auto size_prod = 1;
  for (auto dim : dY.dims()) {
    size_prod *= dim;
  }

  if (!states_initialized_) {
    // set the dropout descriptor
    {
      // Need to protect  as clashes with NCCL
      std::lock_guard<std::mutex> lk(CUDAContext::mutex());
      CUDNN_ENFORCE(cudnnRestoreDropoutDescriptor(
          dropout_desc_,
          cudnn_wrapper_.inline_cudnn_handle(),
          ratio_,
          const_cast<uint8_t*>(states.data<uint8_t>()),
          states_size_in_bytes_,
          random_seed_
          ));
    }
    states_initialized_ = true;
  }

  if (dY.dims() != cudnn_input_dims_) {
    cudnn_input_dims_ = dY.dims();
    CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
        data_desc_,
        GetCudnnTensorFormat(StorageOrder::NCHW),
        cudnnTypeWrapper<T>::type,
        size_prod,
        1,
        1,
        1));

    // get the reserve space we need
    CUDNN_ENFORCE(cudnnDropoutGetReserveSpaceSize(
        data_desc_, &reserve_space_size_in_bytes_));

  }

  // run the computation
  void* mask_data = const_cast<void*>(mask.raw_data());
  CUDNN_ENFORCE(cudnnDropoutBackward(
      cudnn_wrapper_.inline_cudnn_handle(),
      dropout_desc_,
      data_desc_,
      dY.data<T>(),
      data_desc_,
      dX->template mutable_data<T>(),
      mask_data,
      reserve_space_size_in_bytes_));
  return true;
}

bool CuDNNDropoutGradientOp::RunOnDevice() {
  // dispatch based on contents of tensor(s)
  const auto& dY = Input(0);
  auto* dX = Output(0);

  dX->ResizeLike(dY);

  if (dY.IsType<float>()) {
    return DoRunWithType<float, float>();
  } else if (dY.IsType<float16>()) {
    return DoRunWithType<float16, float>();
  }
  return false;
}

namespace {
REGISTER_CUDNN_OPERATOR(Dropout, CuDNNDropoutOp);
REGISTER_CUDNN_OPERATOR(DropoutGrad, CuDNNDropoutGradientOp);
}

#endif

}; // namespace caffe2
