#ifndef CAFFE2_OPERATORS_CONV_OP_H_
#define CAFFE2_OPERATORS_CONV_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

template <typename dtype, class DeviceContext>
class ConvOp final : public ConvPoolOpBase<dtype, DeviceContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS;
  ConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<dtype, DeviceContext>(operator_def, ws),
        kOne(1, &device_context_), kZero(0, &device_context_) {}
  ~ConvOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  Tensor<dtype, DeviceContext> col_buffer_;
  Tensor<dtype, DeviceContext> bias_multiplier_;
  Tensor<dtype, DeviceContext> kOne;
  Tensor<dtype, DeviceContext> kZero;
  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
  INPUT_OUTPUT_STATS(3, 3, 1, 1);
  DISABLE_COPY_AND_ASSIGN(ConvOp);
};

template <typename dtype, class DeviceContext>
class ConvGradientOp final : public ConvPoolOpBase<dtype, DeviceContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS;
  ConvGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<dtype, DeviceContext>(operator_def, ws),
        kOne(1, &device_context_), kZero(0, &device_context_) {}
  ~ConvGradientOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  Tensor<dtype, DeviceContext> col_buffer_;
  Tensor<dtype, DeviceContext> bias_multiplier_;
  Tensor<dtype, DeviceContext> kOne;
  Tensor<dtype, DeviceContext> kZero;
  // input: X, W, dY
  // output: dW, db, and optionally dX
  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(FILTER_GRAD, BIAS_GRAD, INPUT_GRAD);
  INPUT_OUTPUT_STATS(3, 3, 2, 3);
  DISABLE_COPY_AND_ASSIGN(ConvGradientOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_CONV_OP_H_
