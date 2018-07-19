#ifndef CAFFE2_OPERATORS_ACCUMULATE_OP_H_
#define CAFFE2_OPERATORS_ACCUMULATE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class AccumulateOp final : public Operator<Context> {
 public:
  AccumulateOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        gamma_(static_cast<T>(
            OperatorBase::template GetSingleArgument<float>("gamma", 1.0))) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    if (output->dims() != input.dims()) {
      LOG(INFO) << "Reshaping and initializing output.";
      output->ResizeLike(input);
      math::Set<T, Context>(
          output->size(), 0, output->template mutable_data<T>(), &context_);
    }
    math::Axpby<T, Context>(
        input.size(),
        static_cast<T>(1),
        input.template data<T>(),
        gamma_,
        output->template mutable_data<T>(),
        &context_);
    return true;
  }

 protected:
  T gamma_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ACCUMULATE_OP_H_
