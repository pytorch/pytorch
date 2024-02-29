#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// support multiple batches of sessions
template <typename T, class Context>
class PairWiseLossOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(PairWiseLossOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 private:
  INPUT_TAGS(XVALUE, LABEL, LENGTHS);
  OUTPUT_TAGS(YVALUE);
};

template <typename T, class Context>
class PairWiseLossGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(PairWiseLossGradientOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 private:
  INPUT_TAGS(XVALUE, LABEL, DYVALUE, LENGTHS);
  OUTPUT_TAGS(DXVALUE);
};

} // namespace caffe2
