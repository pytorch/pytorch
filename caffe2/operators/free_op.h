#ifndef CAFFE2_OPERATORS_FREE_OP_H_
#define CAFFE2_OPERATORS_FREE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

// FreeOp frees the content of the output blob. We allow it to take in input
// blobs purely for the reason that it can "wait" on the input blobs to be
// produced by some of the earlier operators before a free is called.
template <class Context>
class FreeOp : public Operator<Context> {
 public:
  template <class... Args>
  explicit FreeOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    for (Blob* output : OperatorBase::Outputs()) {
      output->Reset();
    }
    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FREE_OP_H_
