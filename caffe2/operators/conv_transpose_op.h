#ifndef CAFFE2_OPERATORS_CONV_TRANSPOSE_OP_H_
#define CAFFE2_OPERATORS_CONV_TRANSPOSE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_transpose_unpool_op_base.h"

namespace caffe2 {

template <typename T, class Context>
class ConvTransposeOp final : public ConvTransposeUnpoolBase<Context> {
 public:
  USE_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS(Context);
  template <class... Args>
  explicit ConvTransposeOp(Args&&... args)
      : ConvTransposeUnpoolBase<Context>(std::forward<Args>(args)...) {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  Tensor col_buffer_;
  Tensor bias_multiplier_;
  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

template <typename T, class Context>
class ConvTransposeGradientOp final : public ConvTransposeUnpoolBase<Context> {
 public:
  USE_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS(Context);
  template <class... Args>
  explicit ConvTransposeGradientOp(Args&&... args)
      : ConvTransposeUnpoolBase<Context>(std::forward<Args>(args)...),
        no_bias_(this->template GetSingleArgument<bool>("no_bias", false)) {
    CAFFE_ENFORCE(
        !(no_bias_ && OutputSize() == 3),
        "If bias is not present, you should not have 3 grad output.");
  }

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  Tensor col_buffer_;
  Tensor bias_multiplier_;
  const bool no_bias_;
  // input: X, W, dY
  // output: dW, optionally db and dX
  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(FILTER_GRAD, BIAS_OR_INPUT_GRAD, INPUT_GRAD);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CONV_TRANSPOSE_OP_H_
