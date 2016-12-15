#ifndef CAFFE2_OPERATORS_SCALE_OP_H_
#define CAFFE2_OPERATORS_SCALE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class ScaleOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ScaleOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        scale_(OperatorBase::GetSingleArgument<float>("scale", 1.0)) {}
  bool RunOnDevice() override {
    auto& X = Input(0);
    auto* Y = Output(0);
    Y->ResizeLike(X);
    math::Scale<T, Context>(
        X.size(),
        scale_,
        X.template data<T>(),
        Y->template mutable_data<T>(),
        &context_);
    return true;
  }

 protected:
  T scale_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SCALE_OP_H_
