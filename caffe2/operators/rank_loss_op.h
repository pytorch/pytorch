#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// assumes one batch = one session
template <typename T, class Context>
class PairWiseLossOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(PairWiseLossOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  // Input: X, label
  // Output: Y
};

template <typename T, class Context>
class PairWiseLossGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(PairWiseLossGradientOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  // Input: X, label, dY
  // Ouptut: dX. There is no gradient with respect to the label.
};

} // namespace caffe2
