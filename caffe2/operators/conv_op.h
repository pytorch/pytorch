#ifndef CAFFE2_OPERATORS_CONV_OP_H_
#define CAFFE2_OPERATORS_CONV_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_op_shared.h"
#include "caffe2/operators/conv_pool_op_base.h"

CAFFE2_DECLARE_bool(caffe2_force_shared_col_buffer);

namespace caffe2 {

template <typename T, class Context>
class ConvOp final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(Context);
  ConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws) {
    // Since this is the default convolution implementation, we will
    // use CAFFE_ENFORCE instead of OPERATOR_NEEDS_FEATURE.
    CAFFE_ENFORCE(
        group_ == 1 || order_ == StorageOrder::NCHW,
        "Group convolution only supports NCHW order right now.");

    // Create shared buffer mutex in the constructor
    // to avoid race-condition in DAGNet.
    if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
      createSharedBuffer<Context>(ws_);
    }
  }
  ~ConvOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  Tensor<Context> col_buffer_;
  Tensor<Context> bias_multiplier_;
  Tensor<Context> img_shape_device_;
  Tensor<Context> col_buffer_shape_device_;
  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

template <typename T, class Context>
class ConvGradientOp final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(Context);
  ConvGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws),
        no_bias_(OperatorBase::GetSingleArgument<int>("no_bias", 0)) {
    CAFFE_ENFORCE(
        !(no_bias_ && OutputSize() == 3),
        "If bias is not present, you should not have 3 grad output.");
    CAFFE_ENFORCE(
        group_ == 1 || order_ == StorageOrder::NCHW,
        "Group convolution only supports NCHW order right now.");
  }
  ~ConvGradientOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  Tensor<Context> col_buffer_;
  Tensor<Context> bias_multiplier_;
  Tensor<Context> img_shape_device_;
  Tensor<Context> col_buffer_shape_device_;
  bool no_bias_;
  // input: X, W, dY
  // output: dW, db, and optionally dX
  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(FILTER_GRAD, BIAS_OR_INPUT_GRAD, INPUT_GRAD);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CONV_OP_H_
