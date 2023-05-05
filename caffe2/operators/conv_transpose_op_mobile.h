#ifndef CAFFE2_OPERATORS_CONV_TRANSPOSE_MOBILE_OP_H_
#define CAFFE2_OPERATORS_CONV_TRANSPOSE_MOBILE_OP_H_

#include "caffe2/core/common.h"

#ifdef C10_MOBILE

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_transpose_unpool_op_base.h"

namespace caffe2 {

template <typename T, class Context>
class ConvTransposeMobileOp final : public ConvTransposeUnpoolBase<Context> {
 public:
  USE_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS(Context);
  ConvTransposeMobileOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvTransposeUnpoolBase<Context>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        order_ == StorageOrder::NCHW,
        "Only NCHW order is supported right now.");
    OPERATOR_NEEDS_FEATURE(
        this->pad_l() == 0, "operator does not handle row width padding");
    OPERATOR_NEEDS_FEATURE(
        this->pad_r() == 0, "operator does not handle row width padding");
    OPERATOR_NEEDS_FEATURE(this->stride_w() <= 4, "stride width must be <= 4");
  }

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  // We store a numThreads per-worker  tiles of Y, and numThreads per-worker
  // threadBuffer for the gemm output, laid out in that order.
  Tensor threadBuffer_{CPU};

  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

} // namespace caffe2

#endif // C10_MOBILE

#endif // CAFFE2_OPERATORS_CONV_TRANSPOSE_MOBILE_OP_H_
