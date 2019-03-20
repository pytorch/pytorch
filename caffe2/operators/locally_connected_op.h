#ifndef CAFFE2_OPERATORS_LOCALLY_CONNECTED_OP_H_
#define CAFFE2_OPERATORS_LOCALLY_CONNECTED_OP_H_

#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_op_shared.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/locally_connected_op_util.h"

namespace caffe2 {

template <typename T, class Context>
class LocallyConnectedOp final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(Context);

  template <class... Args>
  explicit LocallyConnectedOp(Args&&... args)
      : ConvPoolOpBase<Context>(std::forward<Args>(args)...) {
    // Since this is the default locally connected implementation, we will
    // use CAFFE_ENFORCE instead of OPERATOR_NEEDS_FEATURE.
    CAFFE_ENFORCE(
        group_ == 1 || order_ == StorageOrder::NCHW,
        "Group locally connected only supports NCHW order right now.");
  }

  ~LocallyConnectedOp() = default;

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  void RunOnDeviceWithOrderNCHWImpl(
      const lc_op_util::ShapeParams& shape,
      const T* X_data,
      const T* filter_data,
      const T* bias_data,
      T* Y_data,
      Tensor* column_buffer,
      Tensor* column_transposed_buffer,
      Tensor* output_buffer);

  void RunOnDeviceWithOrderNHWCImpl(
      const lc_op_util::ShapeParams& shape,
      const T* X_data,
      const T* filter_data,
      const T* bias_data,
      T* Y_data,
      Tensor* column_buffer,
      Tensor* column_transposed_buffer,
      Tensor* Y_transposed_buffer);

  Tensor bias_multiplier_{Context::GetDeviceType()};

  // Buffer.
  Tensor column_buffer_{Context::GetDeviceType()};
  Tensor column_transposed_buffer_{Context::GetDeviceType()};
  Tensor Y_transposed_buffer_{Context::GetDeviceType()};

  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

template <typename T, class Context>
class LocallyConnectedGradientOp final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(Context);

  template <class... Args>
  explicit LocallyConnectedGradientOp(Args&&... args)
      : ConvPoolOpBase<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(bool, "no_bias", no_bias_, false) {
    CAFFE_ENFORCE(
        !(no_bias_ && OutputSize() == 3),
        "If bias is not present, you should not have 3 grad output.");
    CAFFE_ENFORCE(
        group_ == 1 || order_ == StorageOrder::NCHW,
        "Group locally connected only supports NCHW order right now.");
  }

  ~LocallyConnectedGradientOp() = default;

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  void RunOnDeviceWithOrderNCHWImpl(
      const lc_op_util::ShapeParams& shape,
      const T* X_data,
      const T* filter_data,
      const T* dY_data,
      T* dfilter_data,
      T* dX_data,
      T* dbias_data,
      Tensor* column_buffer,
      Tensor* column_transposed_buffer,
      Tensor* dY_transposed_buffer);

  void RunOnDeviceWithOrderNHWCImpl(
      const lc_op_util::ShapeParams& shape,
      const T* X_data,
      const T* filter_data,
      const T* dY_data,
      T* dfilter_data,
      T* dX_data,
      T* dbias_data,
      Tensor* column_buffer,
      Tensor* column_transposed_buffer,
      Tensor* dY_transposed_buffer);

  const bool no_bias_;

  Tensor bias_multiplier_{Context::GetDeviceType()};

  // Buffer.
  Tensor column_buffer_{Context::GetDeviceType()};
  Tensor column_transposed_buffer_{Context::GetDeviceType()};
  Tensor dY_transposed_buffer_{Context::GetDeviceType()};

  // input: X, W, dY
  // output: dW, db, and optionally dX
  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(FILTER_GRAD, BIAS_OR_INPUT_GRAD, INPUT_GRAD);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LOCALLY_CONNECTED_OP_H_
