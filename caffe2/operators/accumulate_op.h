#ifndef CAFFE2_OPERATORS_ACCUMULATE_OP_H_
#define CAFFE2_OPERATORS_ACCUMULATE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// Accumulate operator accumulates the input tensor to the output tensor. If the
// output tensor already has the right size, we add to it; otherwise, we first
// initialize the output tensor to all zeros, and then do accumulation. Any
// further calls to the operator, given that no one else fiddles with the output
// in the interim, will do simple accumulations.
template <typename T, class Context>
class AccumulateOp final : public Operator<Context> {
 public:
  AccumulateOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        gamma_(static_cast<T>(
            OperatorBase::template GetSingleArgument<float>("gamma", 1.0))) {}
  USE_OPERATOR_BASE_FUNCTIONS;

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    if (output->dims() != input.dims()) {
      CAFFE_LOG_INFO << "Reshaping and initializing output.";
      output->ReshapeLike(input);
      math::Set<T, Context>(
          output->size(), 0, output->template mutable_data<T>(), &device_context_);
    }
    math::Axpby<T, Context>(
        input.size(), static_cast<T>(1),
        input.template data<T>(),
        gamma_,
        output->template mutable_data<T>(), &device_context_);
    return true;
  }

 protected:
  T gamma_;
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(AccumulateOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_ACCUMULATE_OP_H_
