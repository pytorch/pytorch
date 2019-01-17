// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef CAFFE2_OPERATORS_QUANTIZED_INT8_AABB_ROI_KEYPOINTS_H_
#define CAFFE2_OPERATORS_QUANTIZED_INT8_AABB_ROI_KEYPOINTS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class Int8AABBRoIKeypointsOp final : public Operator<Context> {
 public:
  Int8AABBRoIKeypointsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<std::string>("order", "NCHW")))
  {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NHWC, "Int8 only supports NHWC order.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  StorageOrder order_{StorageOrder::NCHW};
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_QUANTIZED_INT8_AABB_ROI_KEYPOINTS_H_
