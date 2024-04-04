#ifndef CAFFE2_OPERATORS_SCALE_OP_H_
#define CAFFE2_OPERATORS_SCALE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class ScaleOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit ScaleOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        scale_(this->template GetSingleArgument<float>("scale", 1.0)) {}

  template <typename T>
  bool DoRunWithType() {
    auto& X = Input(0);

    auto* Y = Output(0, X.sizes(), at::dtype<T>());
    math::Scale<float, T, Context>(
        X.numel(),
        scale_,
        X.template data<T>(),
        Y->template mutable_data<T>(),
        &context_);
    return true;
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
  }

 protected:
  float scale_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SCALE_OP_H_
