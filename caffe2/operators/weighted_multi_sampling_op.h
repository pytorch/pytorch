#pragma once
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class WeightedMultiSamplingOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit WeightedMultiSamplingOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        num_samples_(
            this->template GetSingleArgument<int64_t>("num_samples", 0)) {
    CAFFE_ENFORCE_GE(num_samples_, 0);
  }

  bool RunOnDevice() override;

 private:
  const int64_t num_samples_;
};

} // namespace caffe2
