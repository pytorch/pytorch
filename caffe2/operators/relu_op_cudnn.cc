#include "caffe2/core/context_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"

namespace caffe2 {

class CuDNNReluOp final : public Operator<CUDAContext> {
 public:
  CuDNNReluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&data_desc_));
    CUDNN_ENFORCE(cudnnCreateActivationDescriptor(&activ_desc_));
    CUDNN_ENFORCE(cudnnSetActivationDescriptor(
        activ_desc_, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
  }

  ~CuDNNReluOp() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_ENFORCE(cudnnDestroyActivationDescriptor(activ_desc_));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);
    auto* Y = Output(0);

    // Return if X is empty
    if (X.size() == 0) {
      Y->mutable_data<T>();
      return true;
    }

    // See if we need to reshape.
    if (X.dims() != cudnn_input_dims_) {
      VLOG(1) << "Setting descriptors.";
      cudnn_input_dims_ = X.dims();
      int C = 1, H = 1, W = 1;
      if (X.ndim() == 4) {
        // Normal 4-dimensional tensors for images.
        C = (order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(3));
        H = (order_ == StorageOrder::NCHW ? X.dim32(2) : X.dim32(1));
        W = (order_ == StorageOrder::NCHW ? X.dim32(3) : X.dim32(2));
      } else {
        // If X is not 4-dimensional, we will simply use H = 1 and W = 1
        // and wrap everything into C.
        C = X.size() / X.dim32(0);
      }
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          data_desc_,
          GetCudnnTensorFormat(order_),
          cudnnTypeWrapper<T>::type,
          X.dim32(0),
          C,
          H,
          W));
    }
    CUDNN_ENFORCE(cudnnActivationForward(
        cudnn_wrapper_.inline_cudnn_handle(),
        activ_desc_,
        cudnnTypeWrapper<T>::kOne(),
        data_desc_,
        X.template data<T>(),
        cudnnTypeWrapper<T>::kZero(),
        data_desc_,
        Y->template mutable_data<T>()));
    return true;
  }

  bool RunOnDevice() override {
    // dispatch based on contents of tensor(s)
    const auto& X = Input(0);
    auto* Y = Output(0);
    Y->ResizeLike(X);

    if (X.IsType<float>()) {
      return DoRunWithType<float>();
    } else if (X.IsType<float16>()) {
      return DoRunWithType<float16>();
    } else {
      LOG(FATAL) << "Unsupported input types";
    }
    return true;
  }

 protected:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t data_desc_;
  cudnnActivationDescriptor_t activ_desc_;
  vector<TIndex> cudnn_input_dims_;
  StorageOrder order_;
};


// Note: You can see that in CuDNNReluGradientOp, we abused the cudnn interface
// by passing in the output tensor for both bottom and top. This is dependent on
// the assumption that the Relu gradient actually does not rely on the bottom
// data, or it treats input=0 the same way as input<0. This is of course not
// very safe, but we have been running in this way in Caffe for a while so it
// *might* be safe to assume so.
class CuDNNReluGradientOp final : public Operator<CUDAContext> {
 public:
  CuDNNReluGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&data_desc_));
    CUDNN_ENFORCE(cudnnCreateActivationDescriptor(&activ_desc_));
    CUDNN_ENFORCE(cudnnSetActivationDescriptor(
        activ_desc_, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
  }

  ~CuDNNReluGradientOp() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_ENFORCE(cudnnDestroyActivationDescriptor(activ_desc_));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& Y = Input(0);
    const auto& dY = Input(1);
    auto* dX = Output(0);

    // Return if Y is empty
    if (Y.size() == 0) {
      dX->mutable_data<T>();
      return true;
    }

    // See if we need to reshape.
    if (Y.dims() != cudnn_input_dims_) {
      VLOG(1) << "Setting descriptors.";
      cudnn_input_dims_ = Y.dims();
      int C = 1, H = 1, W = 1;
      if (Y.ndim() == 4) {
        // Normal 4-dimensional tensors for images.
        C = (order_ == StorageOrder::NCHW ? Y.dim32(1) : Y.dim32(3));
        H = (order_ == StorageOrder::NCHW ? Y.dim32(2) : Y.dim32(1));
        W = (order_ == StorageOrder::NCHW ? Y.dim32(3) : Y.dim32(2));
      } else {
        // If Y is not 4-dimensional, we will simply use H = 1 and W = 1
        // and wrap everything into C.
        C = Y.size() / Y.dim32(0);
      }
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          data_desc_,
          GetCudnnTensorFormat(order_),
          cudnnTypeWrapper<T>::type,
          Y.dim32(0),
          C,
          H,
          W));
    }
    CUDNN_ENFORCE(cudnnActivationBackward(
        cudnn_wrapper_.inline_cudnn_handle(),
        activ_desc_,
        cudnnTypeWrapper<T>::kOne(),
        data_desc_,
        Y.template data<T>(),
        data_desc_,
        dY.template data<T>(),
        data_desc_,
        // Note: strictly speaking, we should be using the input data in this
        // case, but for the ReLU case we rely on the underlying implementation
        // that only the output is needed to calculate the Relu gradient. This
        // will enable us to do memory optimization for in-place relu. To
        // ensure this is correct, a unit test is provided at
        // caffe2/python/operator_test/relu_op_test.py
        Y.template data<T>(),
        cudnnTypeWrapper<T>::kZero(),
        data_desc_,
        dX->template mutable_data<T>()));
    return true;
  }

  bool RunOnDevice() override {
    const auto& Y = Input(0);
    auto* dX = Output(0);
    dX->ResizeLike(Y);

    if (Y.IsType<float>()) {
      return DoRunWithType<float>();
    } else if (Y.IsType<float16>()) {
      return DoRunWithType<float16>();
    } else {
      LOG(FATAL) << "Unsupported input types";
    }
    return true;
  }

 protected:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t data_desc_;
  cudnnActivationDescriptor_t activ_desc_;
  vector<TIndex> cudnn_input_dims_;
  StorageOrder order_;
  // Input: Y, dY; Output: dX
};

namespace {
REGISTER_CUDNN_OPERATOR(Relu, CuDNNReluOp);
REGISTER_CUDNN_OPERATOR(ReluGradient, CuDNNReluGradientOp);
}  // namespace
}  // namespace caffe2
