#pragma once
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class WeightedMultiSamplingOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  WeightedMultiSamplingOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        num_samples_(
            OperatorBase::GetSingleArgument<int64_t>("num_samples", 0)) {
    CAFFE_ENFORCE_GE(num_samples_, 0);
  }

  bool RunOnDevice() override;

 private:
  const int64_t num_samples_;
};

} // namespace caffe2
