// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef CAFFE2_OPERATORS_WEIGHTEDSAMPLE_OP_H_
#define CAFFE2_OPERATORS_WEIGHTEDSAMPLE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class WeightedSampleOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit WeightedSampleOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 private:
  vector<float> cum_mass_;
  Tensor unif_samples_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_WEIGHTEDSAMPLE_OP_H_
