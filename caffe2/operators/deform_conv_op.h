#ifndef CAFFE2_OPERATORS_DEFORM_CONV_OP_H_
#define CAFFE2_OPERATORS_DEFORM_CONV_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_op_shared.h"
#include "caffe2/operators/conv_pool_op_base.h"

C10_DECLARE_bool(caffe2_force_shared_col_buffer);

namespace caffe2 {

template <typename T, class Context>
class DeformConvOpBase : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(Context);
  DeformConvOpBase(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws),
        deformable_group_(
            this->template GetSingleArgument<int>("deformable_group", 1)) {}
  ~DeformConvOpBase() {}

 protected:
  void DeformableIm2col(
      const T* data_im,
      const T* data_offset,
      at::IntArrayRef im_shape,
      at::IntArrayRef col_shape,
      T* data_col);
  void DeformableCol2im(
      const T* data_col,
      const T* data_offset,
      at::IntArrayRef im_shape,
      at::IntArrayRef col_shape,
      T* grad_im);
  void DeformableCol2imCoord(
      const T* data_col,
      const T* data_im,
      const T* data_offset,
      at::IntArrayRef im_shape,
      at::IntArrayRef col_shape,
      T* grad_offset);

 protected:
  int deformable_group_;

#define USE_DEFORMABLE_CONV_BASE_FUNCTIONS(T, Context)   \
  USE_CONV_POOL_BASE_FUNCTIONS(Context);                 \
  using DeformConvOpBase<T, Context>::deformable_group_; \
  using DeformConvOpBase<T, Context>::DeformableIm2col;  \
  using DeformConvOpBase<T, Context>::DeformableCol2im;  \
  using DeformConvOpBase<T, Context>::DeformableCol2imCoord
};

template <typename T, class Context>
class DeformConvOp final : public DeformConvOpBase<T, Context> {
 public:
  USE_DEFORMABLE_CONV_BASE_FUNCTIONS(T, Context);

  DeformConvOp(const OperatorDef& operator_def, Workspace* ws)
      : DeformConvOpBase<T, Context>(operator_def, ws) {
    // Create shared buffer mutex in the constructor
    // to avoid race-condition in DAGNet.
    if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
      createSharedBuffer<Context>(ws_);
    }
  }
  ~DeformConvOp() {}

  bool RunOnDeviceWithOrderNCHW() override;

 private:
  Tensor col_buffer_{Context::GetDeviceType()};
  Tensor bias_multiplier_;
  Tensor img_shape_device_{Context::GetDeviceType()};
  Tensor col_buffer_shape_device_{Context::GetDeviceType()};
  // Input: X, o, W, b
  // Output: Y
  INPUT_TAGS(INPUT, OFFSET, FILTER, BIAS);
};

template <typename T, class Context>
class DeformConvGradientOp final : public DeformConvOpBase<T, Context> {
 public:
  USE_DEFORMABLE_CONV_BASE_FUNCTIONS(T, Context);

  DeformConvGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : DeformConvOpBase<T, Context>(operator_def, ws),
        no_bias_(this->template GetSingleArgument<int>("no_bias", 0)) {
    CAFFE_ENFORCE(
        !(no_bias_ && OutputSize() == 4),
        "If bias is not present, you should not have 4 grad output.");
  }
  ~DeformConvGradientOp() {}

  bool RunOnDeviceWithOrderNCHW() override;

 private:
  Tensor col_buffer_;
  Tensor bias_multiplier_;
  Tensor img_shape_device_{Context::GetDeviceType()};
  Tensor col_buffer_shape_device_{Context::GetDeviceType()};
  bool no_bias_;
  // input: X, W, dY
  // output: dO, dW, db, and optionally dX
  INPUT_TAGS(INPUT, OFFSET, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(OFFSET_GRAD, FILTER_GRAD, BIAS_OR_INPUT_GRAD, INPUT_GRAD);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_DEFORM_CONV_OP_H_
