#include "caffe2/mobile/contrib/arm-compute/core/context.h"
#include "caffe2/mobile/contrib/arm-compute/core/operator.h"

#include "caffe2/mobile/contrib/arm-compute/operators/activation_ops.h"
#include "caffe2/operators/relu_op.h"

namespace caffe2 {

template <typename T>
bool GLReluOp<T>::RunOnDevice() {

  auto *Xblob = OperatorBase::Inputs()[0];
  if (first_run_) {
    X_ = GLContext::getGLTensor<T>(Xblob);
  }

  GLTensor<T> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<GLTensor<T>>();

  if (first_run_) {
    first_run_ = false;
    if (Y->get_underlying() != X_->get_underlying())
    {
      Y->ResizeLike(*X_);
    }
    relu_layer_.configure(
        X_->get_underlying(), Y->get_underlying(),
        arm_compute::ActivationLayerInfo(
          arm_compute::ActivationLayerInfo::ActivationFunction::RELU));

  } else if (second_run_) {
    X_->lazy_allocate(Xblob, second_run_, true);
    second_run_ = false;
    // in place activation, do not need to allocate new memory
    if (Y->get_underlying() != X_->get_underlying()) {
      Y->ResizeLike(*X_);
      Y->allocate();
    }
    relu_layer_.run();
  } else {
    X_->lazy_allocate(Xblob, second_run_, true);
    bool need_allocation = false;
    if (Y->get_underlying() != X_->get_underlying()) {
      need_allocation = Y->ResizeLike(*X_, true);
    }
    relu_layer_.configure(
        X_->get_underlying(), Y->get_underlying(),
        arm_compute::ActivationLayerInfo(
          arm_compute::ActivationLayerInfo::ActivationFunction::RELU));
    if (need_allocation) {
      Y->allocate();
    }
    relu_layer_.run();
  }

  return true;
}

REGISTER_GL_OPERATOR(Relu, GLReluOp<DataType>);

template <typename T>
bool GLSigmoidOp<T>::RunOnDevice() {

  auto *Xblob = OperatorBase::Inputs()[0];
  if (first_run_) {
    X_ = GLContext::getGLTensor<T>(Xblob);
  }

  GLTensor<T> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<GLTensor<T>>();
  if (first_run_) {
    first_run_ = false;

    if (Y->get_underlying() != X_->get_underlying())
    {
        Y->ResizeLike(*X_);
    }

    sigmoid_layer_.configure(
      X_->get_underlying(), Y->get_underlying(),
      arm_compute::ActivationLayerInfo(
          arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC));
  } else if (second_run_) {
    X_->lazy_allocate(Xblob, second_run_, true);
    second_run_ = false;
    // in place activation, do not need to allocate new memory
    if (Y->get_underlying() != X_->get_underlying()) {
      Y->ResizeLike(*X_);
      Y->allocate();
    }
    sigmoid_layer_.run();
  } else {
    X_->lazy_allocate(Xblob, second_run_, true);
    bool need_allocation = false;
    if (Y->get_underlying() != X_->get_underlying())
    {
      need_allocation = Y->ResizeLike(*X_, true);
    }
    sigmoid_layer_.configure(
      X_->get_underlying(), Y->get_underlying(),
      arm_compute::ActivationLayerInfo(
          arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC));
    if (need_allocation) {
      Y->allocate();
    }
    sigmoid_layer_.run();
  }

  return true;
}

REGISTER_GL_OPERATOR(Sigmoid, GLSigmoidOp<DataType>);

} // namespace caffe2
