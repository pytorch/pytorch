// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef CAFFE2_OPERATORS_QUANTIZED_INT8_AABB_ROI_TRANSFORM_OP_H_
#define CAFFE2_OPERATORS_QUANTIZED_INT8_AABB_ROI_TRANSFORM_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class Int8AABBRoITransformOp final : public Operator<Context> {
 public:
  Int8AABBRoITransformOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_QUANTIZED_INT8_AABB_ROI_TRANSFORM_OP_H_
