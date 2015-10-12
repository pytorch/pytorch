#ifndef CAFFE2_OPERATORS_CONV_OP_H_
#define CAFFE2_OPERATORS_CONV_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

template <typename T, class Context>
class ConvOp final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS;
  ConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws),
        kOne(static_cast<T>(1), &device_context_),
        kZero(static_cast<T>(0), &device_context_) {}
  ~ConvOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  Tensor<Context> col_buffer_;
  Tensor<Context> bias_multiplier_;
  Tensor<Context> kOne;
  Tensor<Context> kZero;
  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
  INPUT_OUTPUT_STATS(3, 3, 1, 1);
  DISABLE_COPY_AND_ASSIGN(ConvOp);
};

template <typename T, class Context>
class ConvGradientOp final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS;
  ConvGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws),
        kOne(static_cast<T>(1), &device_context_),
        kZero(static_cast<T>(0), &device_context_) {}
  ~ConvGradientOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  Tensor<Context> col_buffer_;
  Tensor<Context> bias_multiplier_;
  Tensor<Context> kOne;
  Tensor<Context> kZero;
  // input: X, W, dY
  // output: dW, db, and optionally dX
  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(FILTER_GRAD, BIAS_GRAD, INPUT_GRAD);
  INPUT_OUTPUT_STATS(3, 3, 2, 3);
  DISABLE_COPY_AND_ASSIGN(ConvGradientOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_CONV_OP_H_
