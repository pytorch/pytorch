// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef HEATMAP_MAX_KEYPOINT_OP_H_
#define HEATMAP_MAX_KEYPOINT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class HeatmapMaxKeypointOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit HeatmapMaxKeypointOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        should_output_softmax_(this->template GetSingleArgument<bool>(
            "should_output_softmax",
            false)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  bool should_output_softmax_ = false;
};

} // namespace caffe2

#endif // HEATMAP_MAX_KEYPOINT_OP_H_
