#ifndef CAFFE2_OPERATORS_MINMAX_OPS_H_
#define CAFFE2_OPERATORS_MINMAX_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class MaxOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  USE_SIMPLE_CTOR_DTOR(MaxOp)

  bool RunOnDevice() override {
    const auto& X0 = Input(0);
    auto* Y = Output(0);
    Y->ResizeLike(X0);
    const T* X0_data = X0.template data<T>();
    T* Y_data = Y->template mutable_data<T>();
    const int N = X0.numel();
    if (InputSize() == 1) {
      if (Y != &X0) {
        context_.template CopySameDevice<T>(N, X0_data, Y_data);
      }
      return true;
    }
    const auto& X1 = Input(1);
    CAFFE_ENFORCE_EQ(
        X0.sizes(),
        Y->sizes(),
        "Description: Input #1, input dimension:",
        X1.sizes(),
        " should match output dimension: ",
        Y->sizes());
    const T* X1_data = X1.template data<T>();
    math::Max<T, Context>(N, X0_data, X1_data, Y_data, &context_);
    for (int i = 2; i < InputSize(); ++i) {
      const auto& Xi = Input(i);
      CAFFE_ENFORCE_EQ(
          Xi.sizes(),
          Y->sizes(),
          "Description: Input #",
          i,
          ", input dimension:",
          Input(i).sizes(),
          " should match output dimension: ",
          Y->sizes());
      const T* Xi_data = Xi.template data<T>();
      math::Max<T, Context>(N, Y_data, Xi_data, Y_data, &context_);
    }
    return true;
  }
};

template <typename T, class Context>
class MinOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  USE_SIMPLE_CTOR_DTOR(MinOp)

  bool RunOnDevice() override {
    const auto& X0 = Input(0);
    auto* Y = Output(0);
    Y->ResizeLike(X0);
    const T* X0_data = X0.template data<T>();
    T* Y_data = Y->template mutable_data<T>();
    const int N = X0.numel();
    if (InputSize() == 1) {
      if (Y != &X0) {
        context_.template CopySameDevice<T>(N, X0_data, Y_data);
      }
      return true;
    }
    const auto& X1 = Input(1);
    CAFFE_ENFORCE_EQ(
        X0.sizes(),
        Y->sizes(),
        "Description: Input #1, input dimension:",
        X1.sizes(),
        " should match output dimension: ",
        Y->sizes());
    const T* X1_data = X1.template data<T>();
    math::Min<T, Context>(N, X0_data, X1_data, Y_data, &context_);
    for (int i = 2; i < InputSize(); ++i) {
      const auto& Xi = Input(i);
      CAFFE_ENFORCE_EQ(
          Xi.sizes(),
          Y->sizes(),
          "Description: Input #",
          i,
          ", input dimension:",
          Input(i).sizes(),
          " should match output dimension: ",
          Y->sizes());
      const T* Xi_data = Xi.template data<T>();
      math::Min<T, Context>(N, Y_data, Xi_data, Y_data, &context_);
    }
    return true;
  }
};

template <typename T, class Context>
class SelectGradientOpBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SelectGradientOpBase)

  bool RunOnDevice() override;
};

template <typename T, class Context>
class MaxGradientOp final : public SelectGradientOpBase<T, Context> {
 public:
  template <class... Args>
  explicit MaxGradientOp(Args&&... args)
      : SelectGradientOpBase<T, Context>(std::forward<Args>(args)...) {}

  ~MaxGradientOp() = default;
};

template <typename T, class Context>
class MinGradientOp final : public SelectGradientOpBase<T, Context> {
 public:
  template <class... Args>
  explicit MinGradientOp(Args&&... args)
      : SelectGradientOpBase<T, Context>(std::forward<Args>(args)...) {}

  ~MinGradientOp() = default;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_MINMAX_OPS_H_
