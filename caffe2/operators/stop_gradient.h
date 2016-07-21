#pragma once

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
      out->CopyFrom(in, &context_);
    }
    return true;
  }
};

}  // namespace caffe2
