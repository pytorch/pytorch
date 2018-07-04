
#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class ZeroGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(ZeroGradientOp);

  bool RunOnDevice() override {
    return true;
  }
};

} // namespace caffe2
