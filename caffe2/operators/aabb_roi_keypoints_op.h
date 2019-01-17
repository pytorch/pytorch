// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef CAFFE2_OPERATORS_AABB_ROI_KEYPOINTS_OP_H_
#define CAFFE2_OPERATORS_AABB_ROI_KEYPOINTS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class AABBRoIKeypointsOp final : public Operator<Context> {
 public:
  AABBRoIKeypointsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<std::string>("order", "NCHW"))) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  StorageOrder order_{StorageOrder::NCHW};
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_AABB_ROI_KEYPOINTS_OP_H_
