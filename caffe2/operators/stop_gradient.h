#ifndef CAFFE2_OPERATORS_STOP_GRADIENT_H_
#define CAFFE2_OPERATORS_STOP_GRADIENT_H_

#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class StopGradientOp : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(StopGradientOp)
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override {
    const auto& in = Input(0);
    auto* out = Output(0);
    if (out != &in) {
      out->CopyFrom(in, true /*async*/);
    }
    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_STOP_GRADIENT_H_
