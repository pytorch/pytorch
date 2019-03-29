#ifndef CAFFE2_OPERATORS_CONV_OP_H_
#define CAFFE2_OPERATORS_CONV_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_op_shared.h"
#include "caffe2/operators/conv_pool_op_base.h"

C10_DECLARE_bool(caffe2_force_shared_col_buffer);

namespace caffe2 {

template <typename T, class Context>
class ConvOp final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(Context);
  explicit ConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws) {
    // Since this is the default convolution implementation, we will
    // use CAFFE_ENFORCE instead of OPERATOR_NEEDS_FEATURE.
    CAFFE_ENFORCE(
        (group_ == 1 || order_ == StorageOrder::NCHW ||
         std::is_same<Context, CPUContext>::value),
        "Group convolution only supports NCHW order or CPUContext right now.");

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
  bool Run1x1ConvOnDeviceWithOrderNCHW(
      const int N,
      const int C,
      const int HxW,
      const int M,
      const T* X,
      const T* filter,
      const T* bias,
      T* Y);

  bool Run1x1ConvOnDeviceWithOrderNHWC(
      const int N,
      const int C,
      const int HxW,
      const int M,
      const T* X,
      const T* filter,
      const T* bias,
      T* Y);

  Tensor col_buffer_{Context::GetDeviceType()};
  Tensor bias_multiplier_{Context::GetDeviceType()};
  Tensor img_shape_device_{Context::GetDeviceType()};
  Tensor col_buffer_shape_device_{Context::GetDeviceType()};
  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

template <typename T, class Context>
class ConvGradientOp final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(Context);
  explicit ConvGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws),
        no_bias_(this->template GetSingleArgument<int>("no_bias", 0)) {
    CAFFE_ENFORCE(
        !(no_bias_ && OutputSize() == 3),
        "If bias is not present, you should not have 3 grad output.");
    CAFFE_ENFORCE(
        (group_ == 1 || order_ == StorageOrder::NCHW ||
         std::is_same<Context, CPUContext>::value),
        "Group convolution only supports NCHW order or CPUContext right now.");
  }
  ~ConvGradientOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  Tensor col_buffer_;
  Tensor bias_multiplier_;
  Tensor img_shape_device_{Context::GetDeviceType()};
  Tensor col_buffer_shape_device_{Context::GetDeviceType()};
  bool no_bias_;
  // input: X, W, dY
  // output: dW, db, and optionally dX
  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(FILTER_GRAD, BIAS_OR_INPUT_GRAD, INPUT_GRAD);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CONV_OP_H_
