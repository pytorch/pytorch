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
  WeightedSampleOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 private:
  vector<float> cum_mass_;
  Tensor<Context> unif_samples_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_WEIGHTEDSAMPLE_OP_H_
