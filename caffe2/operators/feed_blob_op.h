#ifndef CAFFE2_OPERATORS_FEED_BLOB_OP_H_
#define CAFFE2_OPERATORS_FEED_BLOB_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class FeedBlobOp : public Operator<Context> {
 public:
  template <class... Args>
  explicit FeedBlobOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {
    CAFFE_ENFORCE(
        this->template HasSingleArgumentOfType<string>("value"),
        "value argument must exist and be passed as a string");
    value_ = this->template GetSingleArgument<string>("value", "");
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    *this->template Output<std::string>(0) = value_;
    return true;
  }

 private:
  std::string value_;
};

} // namespace caffe2

#endif
