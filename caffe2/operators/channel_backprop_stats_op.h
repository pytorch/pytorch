/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

  Tensor<Context> dBiasScratch_;
  Tensor<Context> dScaleScratch_;
};

} // namespace caffe2

#endif
