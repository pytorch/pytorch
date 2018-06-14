#ifndef CAFFE2_OPERATORS_REDUCE_OPS_H_
#define CAFFE2_OPERATORS_REDUCE_OPS_H_

#include <algorithm>
#include <functional>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
struct Expander {
  template <typename T>
  bool Forward(
      const std::vector<int>& X_dims,
      const std::vector<int>& Y_dims,
      const T* X_data,
      T* Y_data,
      Context* context) const {
    math::Broadcast(
        X_dims.size(),
        X_dims.data(),
        Y_dims.size(),
        Y_dims.data(),
        X_data,
        Y_data,
        context);
    return true;
  }

  template <typename T>
  bool Backward(
      const std::vector<int>& dims,
      const std::vector<int>& axes,
      const T* dY_data,
      T* dX_data,
      Context* context) const {
    math::ReduceSum<T, Context>(
        dims.size(),
        dims.data(),
        axes.size(),
        axes.data(),
        dY_data,
        dX_data,
        context);
    return true;
  }
};

template <typename InputTypes, class Context>
class ExpandOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ExpandOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(0));
  }
 template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);
    const auto& Y_shape_tensor = Input(1);
    std::vector<int64_t> shape_dims(Y_shape_tensor.size());
    context_.template Copy<int64_t, Context, CPUContext>(
        Y_shape_tensor.size(),
        Y_shape_tensor.template data<int64_t>(),
        shape_dims.data());
    auto* Y = Output(0);

	const int ndim = shape_dims.size();
    const std::vector<int> X_dims(X.dims().cbegin(), X.dims().cend());
    std::vector<int> Y_dims;
    Y_dims.reserve(std::max(ndim, X.ndim()));
    // ndim, X.ndim() might equal to 0
    for (int i = ndim - 1, j = X.ndim() - 1; i >= 0 || j >= 0; i--, j--) {
      const int shape_x = (j >= 0 ? X_dims[j] : 1);
      const int shape_y = (i >= 0 ? shape_dims[i] : 1);
      CAFFE_ENFORCE(
          shape_x == 1 || shape_y == 1 || shape_x == shape_y,
          "Dimensions format invalid.");
      Y_dims.push_back(std::max(shape_x, shape_y));
    }
    std::reverse(Y_dims.begin(), Y_dims.end());
    Y->Resize(Y_dims);
    return expander_.template Forward<T>(
        X_dims,
        Y_dims,
        X.template data<T>(),
        Y->template mutable_data<T>(),
        &context_);
  }

 private:
  Expander<Context> expander_{};
};

template <typename InputTypes, class Context>
class ExpandGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ExpandGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& dY = Input(0);
    const auto& X = Input(1);
    auto* dX = Output(0);
    const int ndim = dY.ndim();
    const std::vector<int> dX_dims(X.dims().cbegin(), X.dims().cend());
    const std::vector<int> dY_dims(dY.dims().cbegin(), dY.dims().cend());
    dX->ResizeLike(X);
    std::vector<int> axes;
    const int offset = ndim - X.ndim();
    for (int i = 0; i < ndim; i++) {
      if (i < offset || dX_dims[i - offset] == 1) {
        axes.push_back(i);
      }
    }
    return expander_.template Backward<T>(
        dY_dims,
        axes,
        dY.template data<T>(),
        dX->template mutable_data<T>(),
        &context_);
  }

 private:
  const Expander<Context> expander_{};
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_REDUCE_OPS_H_
