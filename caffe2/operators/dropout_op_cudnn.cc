#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"

namespace caffe2 {

namespace {

// Round up N to nearest multiple of A.
size_t roundUp(size_t n, size_t a) {
  return n + ((-n % a) + a) % a;
}

constexpr size_t alignmentBytes = 128;

// Returns the index into mask_and_states where states_data begins.
// Ensures that states_data is properly aligned.
template <typename T>
size_t firstElementForStates(size_t mask_bytes) {
  return roundUp(mask_bytes, alignmentBytes) / sizeof(T);
}

// Calculate the required size of the tensor which holds the data for both
// "reserveSpace" (mask) and "states" (RNG states).
template <typename T>
vector<size_t> sizeForMaskAndStates(
    size_t mask_bytes, size_t states_bytes) {
  return vector<size_t>{firstElementForStates<T>(mask_bytes) +
    (roundUp(states_bytes, sizeof(T)) / sizeof(T))};
}

}

class CuDNNDropoutOp final : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  CuDNNDropoutOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_),
        ratio_(OperatorBase::GetSingleArgument<float>("ratio", 0.5)),
        is_test_(OperatorBase::GetSingleArgument<int>("is_test", 0)) {
    DCHECK_GE(ratio_, 0);
    DCHECK_LT(ratio_, 1);
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&data_desc_));

    CUDNN_ENFORCE(cudnnCreateDropoutDescriptor(&dropout_desc_));
    CUDNN_ENFORCE(cudnnDropoutGetStatesSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        reinterpret_cast<size_t*>(&states_size_in_bytes_)));
  }

  ~CuDNNDropoutOp() noexcept {
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

  float ratio_;
  bool is_test_;

  size_t states_size_in_bytes_, reserve_space_size_in_bytes_;
  // Input: X, Output: Y, mask_and_states
};

class CuDNNDropoutGradientOp final : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  CuDNNDropoutGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_),
        ratio_(OperatorBase::GetSingleArgument<float>("ratio", 0.5)),
        is_test_(OperatorBase::GetSingleArgument<int>("is_test", 0)) {
    DCHECK_GE(ratio_, 0);
    DCHECK_LT(ratio_, 1);
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&data_desc_));

    CUDNN_ENFORCE(cudnnCreateDropoutDescriptor(&dropout_desc_));
    CUDNN_ENFORCE(cudnnDropoutGetStatesSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        reinterpret_cast<size_t*>(&states_size_in_bytes_)));
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

  float ratio_;
  bool is_test_;

  size_t states_size_in_bytes_, reserve_space_size_in_bytes_;
  // Input: dY, mask_and_states, Output: dX
};

template <typename T, typename M>
bool CuDNNDropoutOp::DoRunWithType() {
  const auto& X = Input(0);
  auto* Y = Output(0);
  auto* mask_and_states = Output(1);

  auto size_prod = 1;
  for (auto dim : X.dims()) {
    size_prod *= dim;
  }

  // Reshape tensor descriptors if necessary
  if (X.dims() != cudnn_input_dims_) {
    VLOG(1) << "Setting descriptors";
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
    // resize the output to hold both mask and states
    mask_and_states->Resize(sizeForMaskAndStates<T>(
          reserve_space_size_in_bytes_, states_size_in_bytes_));
    // get location of states data in mask_and_states
    T* states_data = mask_and_states->template mutable_data<T>() +
      firstElementForStates<T>(reserve_space_size_in_bytes_);
    // set the dropout descriptor
    CUDNN_ENFORCE(cudnnSetDropoutDescriptor(
        dropout_desc_,
        cudnn_wrapper_.inline_cudnn_handle(),
        ratio_,
        states_data,
        states_size_in_bytes_,
        0 // seed
        ));
  }

  // now actually run the computation
  if (is_test_) {
    if (Y != &X) {
      context_.Copy<T, CUDAContext, CUDAContext>(
          X.size(), X.template data<T>(), Y->template mutable_data<T>());
    }
    return true;
  } else {
    CUDNN_ENFORCE(cudnnDropoutForward(
        cudnn_wrapper_.inline_cudnn_handle(),
        dropout_desc_,
        data_desc_,
        X.template data<T>(),
        data_desc_,
        Y->template mutable_data<T>(),
        mask_and_states->raw_mutable_data(),
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
  const auto& mask_and_states = Input(1);
  auto* dX = Output(0);

  auto size_prod = 1;
  for (auto dim : dY.dims()) {
    size_prod *= dim;
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
    // get location of states data in mask_and_states
    T* states_data = const_cast<T*>(
        mask_and_states.template data<T>() +
        firstElementForStates<T>(reserve_space_size_in_bytes_));
    // set the dropout descriptor
    CUDNN_ENFORCE(cudnnSetDropoutDescriptor(
        dropout_desc_,
        cudnn_wrapper_.inline_cudnn_handle(),
        ratio_,
        states_data,
        states_size_in_bytes_,
        0 // seed
        ));
  }

  // run the computation
  void* mask_data = const_cast<void*>(mask_and_states.raw_data());
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
  const auto& states = Input(1);
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

}; // namespace caffe2
