// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef CAFFE2_OPERATORS_BUCKETIZE_OP_H_
#define CAFFE2_OPERATORS_BUCKETIZE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class BucketizeOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  BucketizeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        boundaries_(this->template GetRepeatedArgument<float>("boundaries")) {
    CAFFE_ENFORCE(
        std::is_sorted(boundaries_.begin(), boundaries_.end()),
        "The boundaries need to be monotonically increasing");
  }

  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(X);
  OUTPUT_TAGS(INDICES);

 private:
  std::vector<float> boundaries_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_BUCKETIZE_OP_H_
