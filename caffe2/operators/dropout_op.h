#ifndef CAFFE2_OPERATORS_DROPOUT_OP_H_
#define CAFFE2_OPERATORS_DROPOUT_OP_H_

#include <cstdlib>
#include <ctime>

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class DropoutOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  DropoutOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(bool, OpSchema::Arg_IsTest, is_test_, false),
        OP_SINGLE_ARG(float, "ratio", ratio_, 0.5f) {
    CAFFE_ENFORCE_GE(ratio_, 0);
    CAFFE_ENFORCE_LT(ratio_, 1);
    // srand for Eigen
    std::srand(operator_def.device_option().random_seed());
  }

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
    Y->ResizeLike(X);
    const int N = X.numel();
    const T* X_data = X.template data<T>();
    T* Y_data = Y->template mutable_data<T>();
    if (is_test_) {
      if (Y != &X) {
        context_.template CopySameDevice<T>(N, X_data, Y_data);
      }
    } else {
      auto* mask = Output(1);
      mask->ResizeLike(X);
      bool* mask_data = mask->template mutable_data<bool>();
      DropoutForward(N, X_data, Y_data, mask_data);
    }
    return true;
  }

 protected:
  void DropoutForward(const int N, const T* X, T* Y, bool* mask);

  const bool is_test_;
  const float ratio_;
  Tensor uniform_{Context::GetDeviceType()};
};

template <typename T, class Context>
class DropoutGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  DropoutGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(float, "ratio", ratio_, 0.5f) {
    CAFFE_ENFORCE_GE(ratio_, 0);
    CAFFE_ENFORCE_LT(ratio_, 1);
  }

  bool RunOnDevice() override {
    const auto& dY = Input(0);
    const auto& mask = Input(1);
    auto* dX = Output(0);
    dX->ResizeLike(dY);
    const int N = dY.numel();
    DropoutBackward(
        N,
        dY.template data<T>(),
        mask.template data<bool>(),
        dX->template mutable_data<T>());
    return true;
  }

 protected:
  void DropoutBackward(const int N, const T* dY, const bool* mask, T* dX);

  const float ratio_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_DROPOUT_OP_H_
