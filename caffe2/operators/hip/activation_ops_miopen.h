#ifndef CAFFE2_OPERATORS_ACTIVATION_OPS_MIOPEN_H_
#define CAFFE2_OPERATORS_ACTIVATION_OPS_MIOPEN_H_

#include "caffe2/core/hip/context_gpu.h"
#include "caffe2/core/hip/miopen_wrapper.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"

namespace caffe2 {

class MIOPENActivationOpBase : public Operator<HIPContext> {
 public:
  USE_OPERATOR_FUNCTIONS(HIPContext);

  MIOPENActivationOpBase(const OperatorDef& operator_def, Workspace* ws)
      : Operator<HIPContext>(operator_def, ws), miopen_wrapper_(&context_) {
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&data_desc_));
    MIOPEN_ENFORCE(miopenCreateActivationDescriptor(&act_desc_));
  }

  virtual ~MIOPENActivationOpBase() {
    MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(data_desc_));
    MIOPEN_ENFORCE(miopenDestroyActivationDescriptor(act_desc_));
  }

 protected:
  MIOPENWrapper miopen_wrapper_;
  miopenTensorDescriptor_t data_desc_;
  miopenActivationDescriptor_t act_desc_;
  vector<int64_t> mio_dims_;
};

template <miopenActivationMode_t kMIOPENActivationMode>
class MIOPENActivationOp final : public MIOPENActivationOpBase {
 public:
  USE_OPERATOR_FUNCTIONS(HIPContext);

  MIOPENActivationOp(const OperatorDef& operator_def, Workspace* ws)
      : MIOPENActivationOpBase(operator_def, ws) {
    MIOPEN_ENFORCE(miopenSetActivationDescriptor(
        act_desc_, kMIOPENActivationMode, 1.0, 1.0, 1.0));
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);
    auto* Y = Output(0);
    Y->ResizeLike(X);
    if (X.size() == 0) {
      Y->template mutable_data<T>();
      return true;
    }
    // See if we need to reshape.
    if (X.sizes() != mio_dims_) {
      VLOG(1) << "Setting descriptors.";
      mio_dims_ = X.sizes().vec();
      int C = 1, H = 1, W = 1;
      if (X.ndim() == 4) {
        // Normal 4-dimensional tensors for images.
        C = X.dim32(1);
        H = X.dim32(2);
        W = X.dim32(3);
      } else {
        // If X is not 4-dimensional, we will simply use H = 1 and W = 1
        // and wrap everything into C.
        C = X.size() / X.dim32(0);
      }
      MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
          data_desc_, miopenTypeWrapper<T>::type, X.dim32(0), C, H, W));
    }
    MIOPEN_ENFORCE(miopenActivationForward(
        this->miopen_wrapper_.inline_miopen_handle(),
        this->act_desc_,
        miopenTypeWrapper<T>::kOne(),
        this->data_desc_,
        X.template data<T>(),
        miopenTypeWrapper<T>::kZero(),
        this->data_desc_,
        Y->template mutable_data<T>()));
    return true;
  }
};

template <miopenActivationMode_t kMIOPENActivationMode>
class MIOPENActivationGradientOp final : public MIOPENActivationOpBase {
 public:
  USE_OPERATOR_FUNCTIONS(HIPContext);

  MIOPENActivationGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : MIOPENActivationOpBase(operator_def, ws) {
    MIOPEN_ENFORCE(miopenSetActivationDescriptor(
        act_desc_, kMIOPENActivationMode, 1.0, 1.0, 1.0));
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& Y = Input(0);
    const auto& dY = Input(1);
    auto* dX = Output(0);
    dX->ResizeLike(Y);
    if (Y.size() == 0) {
      dX->template mutable_data<T>();
      return true;
    }
    // See if we need to reshape.
    if (Y.sizes() != mio_dims_) {
      VLOG(1) << "Setting descriptors.";
      mio_dims_ = Y.sizes().vec();
      int C = 1, H = 1, W = 1;
      if (Y.ndim() == 4) {
        // Normal 4-dimensional tensors for images.
        C = Y.dim32(1);
        H = Y.dim32(2);
        W = Y.dim32(3);
      } else {
        // If Y is not 4-dimensional, we will simply use H = 1 and W = 1
        // and wrap everything into C.
        C = Y.size() / Y.dim32(0);
      }
      MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
          data_desc_, miopenTypeWrapper<T>::type, Y.dim32(0), C, H, W));
    }
    MIOPEN_ENFORCE(miopenActivationBackward(
        this->miopen_wrapper_.inline_miopen_handle(),
        this->act_desc_,
        miopenTypeWrapper<T>::kOne(),
        this->data_desc_,
        Y.template data<T>(),
        this->data_desc_,
        dY.template data<T>(),
        this->data_desc_,
        Y.template data<T>(),
        miopenTypeWrapper<T>::kZero(),
        this->data_desc_,
        dX->template mutable_data<T>()));
    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ACTIVATION_OPS_MIOPEN_H_
