#ifndef CHANNEL_BACKPROP_STATS_OP_H
#define CHANNEL_BACKPROP_STATS_OP_H

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class ChannelBackpropStatsOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ChannelBackpropStatsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  ~ChannelBackpropStatsOp() {}

  bool RunOnDevice() override {
    return true;
  }

 protected:
  INPUT_TAGS(INPUT, SAVED_MEAN, SAVED_INV_STDDEV, OUTPUT_GRAD);
  OUTPUT_TAGS(SCALE_GRAD, BIAS_GRAD);

  Tensor dBiasScratch_{Context::GetDeviceType()};
  Tensor dScaleScratch_{Context::GetDeviceType()};
};

} // namespace caffe2

#endif
