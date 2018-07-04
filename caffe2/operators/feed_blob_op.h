#ifndef CAFFE2_OPERATORS_FEED_BLOB_OP_H_
#define CAFFE2_OPERATORS_FEED_BLOB_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class FeedBlobOp : public Operator<Context> {
 public:
  FeedBlobOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    CAFFE_ENFORCE(
        OperatorBase::HasSingleArgumentOfType<string>("value"),
        "value argument must exist and be passed as a string");
    value_ = OperatorBase::GetSingleArgument<string>("value", "");
  }

  bool RunOnDevice() override {
    *OperatorBase::Output<std::string>(0) = value_;
    return true;
  }

 private:
  std::string value_;
};

} // namespace caffe2

#endif
