#ifndef CAFFE2_OPERATORS_SELU_OP_H_
#define CAFFE2_OPERATORS_SELU_OP_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class SeluOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit SeluOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {
    alpha_ = this->template GetSingleArgument<T>(
        "alpha", 1.6732632423543772848170429916717f);
    lambda_ = this->template GetSingleArgument<T>(
        "scale", 1.0507009873554804934193349852946f);
    // In the paper "scale" is named "lambda", but "lambda" is a reserved
    // keyword in python
    CAFFE_ENFORCE_GT(lambda_, 1.0);
  }

  bool RunOnDevice() override;

 protected:
  T alpha_;
  T lambda_;
};

template <typename T, class Context>
class SeluGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit SeluGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {
    alpha_ = this->template GetSingleArgument<T>(
        "alpha", 1.6732632423543772848170429916717f);
    lambda_ = this->template GetSingleArgument<T>(
        "scale", 1.0507009873554804934193349852946f);
    CAFFE_ENFORCE_GT(lambda_, 1.0);
  }

  bool RunOnDevice() override;

 protected:
  T alpha_;
  T lambda_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SELU_OP_H_
