// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef CAFFE2_OPERATORS_BATCH_BUCKETIZE_OP_H_
#define CAFFE2_OPERATORS_BATCH_BUCKETIZE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class BatchBucketizeOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit BatchBucketizeOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(FEATURE, INDICES, BOUNDARIES, LENGTHS);
  OUTPUT_TAGS(O);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_BATCH_BUCKETIZE_OP_H_
