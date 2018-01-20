// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef BBOX_TRANSFORM_OP_H_
#define BBOX_TRANSFORM_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class BBoxTransformOp final : public Operator<Context> {
 public:
  BBoxTransformOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        weights_(OperatorBase::GetRepeatedArgument<T>(
            "weights",
            vector<T>{1.0f, 1.0f, 1.0f, 1.0f})) {
    CAFFE_ENFORCE_EQ(
        weights_.size(),
        4,
        "weights size " + caffe2::to_string(weights_.size()) + "must be 4.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  // weights [wx, wy, ww, wh] to apply to the regression target
  vector<T> weights_;
};

} // namespace caffe2

#endif // BBOX_TRANSFORM_OP_H_
