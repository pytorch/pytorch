// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef CAFFE2_OPERATORS_BUCKETIZE_OP_H_
#define CAFFE2_OPERATORS_BUCKETIZE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/export_caffe2_op_to_c10.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(BucketizeOp);

namespace caffe2 {

template <class Context>
class BucketizeOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit BucketizeOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        boundaries_(this->template GetRepeatedArgument<float>("boundaries")) {
    CAFFE_ENFORCE(
        std::is_sorted(boundaries_.begin(), boundaries_.end()),
        "The boundaries need to be monotonically increasing");

    boundaries_device_.Resize(boundaries_.size());
    context_.template CopyFromCPU<float>(
        boundaries_.size(),
        boundaries_.data(),
        boundaries_device_.mutable_data<float>());
    context_.FinishDeviceComputation();
  }

  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(X);
  OUTPUT_TAGS(INDICES);

 private:
  std::vector<float> boundaries_;
  Tensor boundaries_device_{Context::GetDeviceType()};
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_BUCKETIZE_OP_H_
