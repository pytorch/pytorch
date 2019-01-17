// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef CAFFE2_OPERATORS_QUANTIZED_INT8_AABB_ROI_NMS_H_
#define CAFFE2_OPERATORS_QUANTIZED_INT8_AABB_ROI_NMS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"

namespace caffe2 {

template <class Context>
class Int8AABBRoINMSOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  Int8AABBRoINMSOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        min_score_(this->template GetSingleArgument<float>("min_score", 0.05)),
        max_iou_(this->template GetSingleArgument<float>("max_iou", 0.3)),
        max_objects_(
            this->template GetSingleArgument<int>("max_objects", 100)) {}

  ~Int8AABBRoINMSOp() {}

  bool RunOnDevice() override;

 protected:
  /* Min score for output bounding boxes */
  float min_score_ = 0.05;
  /* Max allowed IoU between bounding boxes of the same class */
  float max_iou_ = 0.3;
  /* Max number of detected objects per image */
  int max_objects_ = 100;
};

} // namespace caffe2
#endif // CAFFE2_OPERATORS_QUANTIZED_INT8_AABB_ROI_NMS_H_
