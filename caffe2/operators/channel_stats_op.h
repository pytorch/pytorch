#ifndef CAFFE2_OPERATORS_CHANNEL_STATS_OP_H
#define CAFFE2_OPERATORS_CHANNEL_STATS_OP_H

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class ChannelStatsOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ChannelStatsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  ~ChannelStatsOp() {}

  bool RunOnDevice() override {
    return true;
  }

 protected:
  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(SUM, SUMSQ);

  Tensor<Context> sumScratch_;
  Tensor<Context> sumsqScratch_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CHANNEL_STATS_OP_H
