#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
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

    // Choose function body for given math type
    TensorProto_DataType math = TensorProto_DataType_FLOAT; // hardcode for now

    switch (math) {
      case TensorProto_DataType_FLOAT:
        body_ = &CuDNNReluOp::DoRunWithMathType<float>;
        break;
      case TensorProto_DataType_FLOAT16:
        body_ = &CuDNNReluOp::DoRunWithMathType<float16>;
        break;
      default:
        CAFFE_THROW("Invalid math type specified");
    }
  }

  ~CuDNNReluOp() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_ENFORCE(cudnnDestroyActivationDescriptor(activ_desc_));
  }

  template <typename M>
  bool DoRunWithMathType() {
    return DispatchHelper<
      TensorTypes<
        float,
        float16>,
      M>::call(this, Input(0));
  }

  template <typename T, typename M>
  bool DoRunWithType() {
    const auto& X = Input(0);
    auto* Y = Output(0);
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

    return (this->*body_)();
  }

 protected:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t data_desc_;
  cudnnActivationDescriptor_t activ_desc_;
  vector<TIndex> cudnn_input_dims_;
  StorageOrder order_;
  bool (CuDNNReluOp::*body_)();
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

    // Choose function body for given math type
    TensorProto_DataType math = TensorProto_DataType_FLOAT; // hardcode for now

    switch (math) {
      case TensorProto_DataType_FLOAT:
        body_ = &CuDNNReluGradientOp::DoRunWithMathType<float>;
        break;
      case TensorProto_DataType_FLOAT16:
        body_ = &CuDNNReluGradientOp::DoRunWithMathType<float16>;
        break;
      default:
        CAFFE_THROW("Invalid math type specified");
    }
  }

  ~CuDNNReluGradientOp() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_ENFORCE(cudnnDestroyActivationDescriptor(activ_desc_));
  }

  template <typename M>
  bool DoRunWithMathType() {
    return DispatchHelper<
      TensorTypes<
        float,
        float16>,
      M>::call(this, Input(0));
  }

  template <typename T, typename M>
  bool DoRunWithType() {
    const auto& Y = Input(0);
    const auto& dY = Input(1);
    auto* dX = Output(0);
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
    const typename cudnnTypeWrapper<T>::ScalingParamType kOne = 1;
    const typename cudnnTypeWrapper<T>::ScalingParamType kZero = 0;
    CUDNN_ENFORCE(cudnnActivationBackward(
        cudnn_wrapper_.inline_cudnn_handle(),
        activ_desc_,
        &kOne,
        data_desc_,
        Y.template data<T>(),
        data_desc_,
        dY.template data<T>(),
        data_desc_,
        Y.template data<T>(),
        &kZero,
        data_desc_,
        dX->template mutable_data<T>()));
    return true;
  }

  bool RunOnDevice() override {
    const auto& Y = Input(0);
    const auto& dY = Input(1);
    auto* dX = Output(0);
    dX->ResizeLike(Y);

    return (this->*body_)();
  }

 protected:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t data_desc_;
  cudnnActivationDescriptor_t activ_desc_;
  vector<TIndex> cudnn_input_dims_;
  StorageOrder order_;
  // Input: Y, dY; Output: dX
 private:
  bool (CuDNNReluGradientOp::*body_)();
};

namespace {
REGISTER_CUDNN_OPERATOR(Relu, CuDNNReluOp);
REGISTER_CUDNN_OPERATOR(ReluGradient, CuDNNReluGradientOp);
}  // namespace
}  // namespace caffe2
