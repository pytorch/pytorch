// conv_op_impl.h is the templated implementation of the conv_op.h file.
#ifndef CAFFE2_OPERATORS_CONV_OP_IMPL_H_
#define CAFFE2_OPERATORS_CONV_OP_IMPL_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

template <typename T, class Context>
bool ConvOp<T, Context>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto& bias = Input(BIAS);
  auto* Y = Output(0);
  const int N = X.dim(0), C = X.dim(1), H = X.dim(2), W = X.dim(3);
  CAFFE_DCHECK_EQ(filter.ndim(), 4);
  const int M = filter.dim(0);
  CAFFE_DCHECK_EQ(filter.dim(1), C);
  CAFFE_DCHECK_EQ(filter.dim(2), kernel_h_);
  CAFFE_DCHECK_EQ(filter.dim(3), kernel_w_);
  CAFFE_DCHECK_EQ(bias.ndim(), 1);
  CAFFE_DCHECK_EQ(bias.dim(0), M);
  ConvPoolOpBase<Context>::SetOutputSize(X, Y, filter.dim(0));
  // The dimension of each kernel
  const int kernel_dim = C * kernel_h_ * kernel_w_;
  // The offset corresponding to a single input image, and a single output
  // image.
  const int input_offset = C * H * W;
  const int output_offset = Y->size() / Y->dim(0);
  // The output image size is the spatial size of the output.
  const int output_image_size = Y->dim(2) * Y->dim(3);
  // The col buffer is stored in CHW order as well - kernel_dim, and the height
  // and width.
  col_buffer_.Reshape(std::vector<int>{
      C, kernel_h_, kernel_w_, Y->dim(2), Y->dim(3)});
  if (bias_multiplier_.size() != output_image_size) {
    // If the helper bias multiplier is not M, reshape and fill it with one.
    bias_multiplier_.Reshape(std::vector<int>(1, output_image_size));
    math::Set<T, Context>(
        output_image_size, static_cast<T>(1),
        bias_multiplier_.template mutable_data<T>(), &device_context_);
  }
  const T* Xdata = X.template data<T>();
  T* col_buffer_data = col_buffer_.template mutable_data<T>();
  T* Ydata = Y->template mutable_data<T>();
  // Im2col, followed by gemm.
  for (int image_id = 0; image_id < N; ++image_id) {
    math::Im2col<T, Context, StorageOrder::NCHW>(
        Xdata, C, H, W, kernel_h_, kernel_w_,
        pad_t_, pad_l_, pad_b_, pad_r_, stride_h_, stride_w_, col_buffer_data,
        &device_context_);
    // Weight term
    math::Gemm<T, Context>(
        CblasNoTrans, CblasNoTrans, M, output_image_size, kernel_dim,
        kOne.template data<T>(), filter.template data<T>(), col_buffer_data,
        kZero.template data<T>(), Ydata,
        &device_context_);
    // Bias term
    math::Gemm<T, Context>(
        CblasNoTrans, CblasNoTrans, M, output_image_size, 1, kOne.template data<T>(),
        bias.template data<T>(), bias_multiplier_.template data<T>(),
        kOne.template data<T>(), Ydata,
        &device_context_);
    Xdata += input_offset;
    Ydata += output_offset;
  }
  return true;
}

// The implementations.
template <typename T, class Context>
bool ConvOp<T, Context>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto& bias = Input(BIAS);
  auto* Y = Output(0);
  const int N = X.dim(0), H = X.dim(1), W = X.dim(2), C = X.dim(3);
  CAFFE_DCHECK_EQ(filter.ndim(), 4);
  const int M = filter.dim(0);
  CAFFE_DCHECK_EQ(filter.dim(1), kernel_h_);
  CAFFE_DCHECK_EQ(filter.dim(2), kernel_w_);
  CAFFE_DCHECK_EQ(filter.dim(3), C);
  CAFFE_DCHECK_EQ(bias.ndim(), 1);
  CAFFE_DCHECK_EQ(bias.dim(0), M);
  ConvPoolOpBase<Context>::SetOutputSize(X, Y, filter.dim(0));
  // The dimension of each kernel
  const int kernel_dim = kernel_h_ * kernel_w_ * C;
  // The offset corresponding to a single input image, and a single output
  // image.
  const int input_offset = H * W * C;
  const int output_offset = Y->size() / Y->dim(0);
  // The output image size is the spatial size of the output.
  const int output_image_size = Y->dim(1) * Y->dim(2);
  // The col buffer is stored in HWC order as well - kernel_dim, and the height
  // and width.
  const T* Xdata = X.template data<T>();
  T* Ydata = Y->template mutable_data<T>();
  if (bias_multiplier_.size() != output_image_size) {
    // If the helper bias multiplier is not M, reshape and fill it with one.
    bias_multiplier_.Reshape(std::vector<int>(1, output_image_size));
    math::Set<T, Context>(
        output_image_size, static_cast<T>(1),
        bias_multiplier_.template mutable_data<T>(), &device_context_);
  }
  // Specialized path for 1 by 1 convolution
  if (kernel_dim == C && Y->dim(1) == X.dim(1) && Y->dim(2) == X.dim(2)) {
    if (bias_multiplier_.size() != N * H * W) {
      // If the helper bias multiplier is not M, reshape and fill it with one.
      bias_multiplier_.Reshape(std::vector<int>(1, N * H * W));
      math::Set<T, Context>(
          N * H * W, static_cast<T>(1),
          bias_multiplier_.template mutable_data<T>(), &device_context_);
    }
    math::Gemm<T, Context>(
        CblasNoTrans, CblasTrans, N * H * W, M, C, kOne.template data<T>(), Xdata,
        filter.template data<T>(), kZero.template data<T>(), Ydata, &device_context_);
    math::Gemm<T, Context>(
        CblasNoTrans, CblasNoTrans, N * H * W, M, 1, kOne.template data<T>(),
        bias_multiplier_.template data<T>(), bias.template data<T>(), kOne.template data<T>(), Ydata,
        &device_context_);
  } else {
    if (bias_multiplier_.size() != output_image_size) {
      // If the helper bias multiplier is not M, reshape and fill it with one.
      bias_multiplier_.Reshape(std::vector<int>(1, output_image_size));
      math::Set<T, Context>(
          output_image_size, static_cast<T>(1),
          bias_multiplier_.template mutable_data<T>(), &device_context_);
    }
    col_buffer_.Reshape(std::vector<int>{
        Y->dim(1), Y->dim(2), kernel_h_, kernel_w_, C});
    T* col_buffer_data = col_buffer_.template mutable_data<T>();
    // Im2col, followed by gemm.
    for (int image_id = 0; image_id < N; ++image_id) {
      math::Im2col<T, Context, StorageOrder::NHWC>(
          Xdata, C, H, W, kernel_h_, kernel_w_,
          pad_t_, pad_l_, pad_b_, pad_r_, stride_h_, stride_w_, col_buffer_data,
          &device_context_);
      // Weight term
      // Wait, is this right....?
      math::Gemm<T, Context>(
          CblasNoTrans, CblasTrans, output_image_size, M, kernel_dim,
          kOne.template data<T>(), col_buffer_data, filter.template data<T>(), kZero.template data<T>(), Ydata,
          &device_context_);
      // Bias term
      math::Gemm<T, Context>(
          CblasNoTrans, CblasNoTrans, output_image_size, M, 1, kOne.template data<T>(),
          bias_multiplier_.template data<T>(), bias.template data<T>(), kOne.template data<T>(), Ydata,
          &device_context_);
      Xdata += input_offset;
      Ydata += output_offset;
    }
  }
  return true;
}

template <typename T, class Context>
bool ConvGradientOp<T, Context>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto& dY = Input(OUTPUT_GRAD);
  auto* dfilter = Output(FILTER_GRAD);
  auto* dbias = Output(BIAS_GRAD);
  const int N = X.dim(0), C = X.dim(1), H = X.dim(2), W = X.dim(3);
  ConvPoolOpBase<Context>::ComputePads(H, W);
  CAFFE_DCHECK_EQ(filter.ndim(), 4);
  const int M = filter.dim(0);
  CAFFE_DCHECK_EQ(filter.dim(1), C);
  CAFFE_DCHECK_EQ(filter.dim(2), kernel_h_);
  CAFFE_DCHECK_EQ(filter.dim(3), kernel_w_);
  dfilter->ReshapeLike(filter);
  dbias->Reshape(std::vector<int>{M});
  // The dimension of each kernel
  const int kernel_dim = C * kernel_h_ * kernel_w_;
  // The offset corresponding to a single input image, and a single output
  // image.
  const int input_offset = C * H * W;
  const int output_offset = dY.size() / dY.dim(0);
  // The output image size is the spatial size of the output.
  const int output_image_size = dY.dim(2) * dY.dim(3);
  // The col buffer is stored in CHW order as well - kernel_dim, and the height
  // and width.
  col_buffer_.Reshape(std::vector<int>{kernel_dim, output_image_size});
  if (bias_multiplier_.size() != output_image_size) {
    // If the helper bias multiplier is not M, reshape and fill it with one.
    bias_multiplier_.Reshape(std::vector<int>(1, output_image_size));
    math::Set<T, Context>(
        output_image_size, static_cast<T>(1),
        bias_multiplier_.template mutable_data<T>(), &device_context_);
  }
  const T* Xdata = X.template data<T>();
  const T* filter_data = filter.template data<T>();
  const T* dYdata = dY.template data<T>();
  T* col_buffer_data = col_buffer_.template mutable_data<T>();
  T* dfilter_data = dfilter->template mutable_data<T>();
  T* dbias_data = dbias->template mutable_data<T>();
  // Pre-setting the gradients to zero.
  math::Set<T, Context>(dfilter->size(), 0, dfilter_data,
                                  &device_context_);
  math::Set<T, Context>(dbias->size(), 0, dbias_data,
                                  &device_context_);
  for (int image_id = 0; image_id < N; ++image_id) {
    // When we compute the gradient with respect to the filters, we need to do
    // im2col to allow gemm-type computation.
    math::Im2col<T, Context, StorageOrder::NCHW>(
        Xdata, C, H, W, kernel_h_, kernel_w_,
        pad_t_, pad_l_, pad_b_, pad_r_, stride_h_, stride_w_, col_buffer_data,
        &device_context_);
    // Gradient with respect to filter.
    math::Gemm<T, Context>(
        CblasNoTrans, CblasTrans, M, kernel_dim, output_image_size,
        kOne.template data<T>(), dYdata + output_offset * image_id, col_buffer_data,
        kOne.template data<T>(), dfilter_data, &device_context_);
    // Gradient with respect to bias
    math::Gemv<T, Context>(
        CblasNoTrans, M, output_image_size, kOne.template data<T>(),
        dYdata + output_offset * image_id, bias_multiplier_.template data<T>(),
        kOne.template data<T>(), dbias_data, &device_context_);
    Xdata += input_offset;
  }
  if (OutputSize() == 3) {
    // Compute the gradient w.r.t. the input.
    auto *dX = Output(INPUT_GRAD);
    dX->ReshapeLike(X);
    T* dXdata = dX->template mutable_data<T>();
    for (int image_id = 0; image_id < N; ++image_id) {
      // Compute gradient into col_buffer.
      math::Gemm<T, Context>(
          CblasTrans, CblasNoTrans, kernel_dim, output_image_size, M,
          kOne.template data<T>(), filter_data, dYdata + output_offset * image_id,
          kZero.template data<T>(), col_buffer_data, &device_context_);
      math::Col2im<T, Context, StorageOrder::NCHW>(
          col_buffer_data, C, H, W, kernel_h_, kernel_w_,
          pad_t_, pad_l_, pad_b_, pad_r_,
          stride_h_, stride_w_, dXdata, &device_context_);
      dXdata += input_offset;
    }
  }
  return true;
}

template <typename T, class Context>
bool ConvGradientOp<T, Context>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto& dY = Input(OUTPUT_GRAD);
  auto* dfilter = Output(FILTER_GRAD);
  auto* dbias = Output(BIAS_GRAD);
  const int N = X.dim(0), H = X.dim(1), W = X.dim(2), C = X.dim(3);
  ConvPoolOpBase<Context>::ComputePads(H, W);
  CAFFE_DCHECK_EQ(filter.ndim(), 4);
  const int M = filter.dim(0);
  CAFFE_DCHECK_EQ(filter.dim(1), kernel_h_);
  CAFFE_DCHECK_EQ(filter.dim(2), kernel_w_);
  CAFFE_DCHECK_EQ(filter.dim(3), C);
  dfilter->ReshapeLike(filter);
  dbias->Reshape(std::vector<int>{M});
  // The dimension of each kernel
  const int kernel_dim = kernel_h_ * kernel_w_ * C;
  // The offset corresponding to a single input image, and a single output
  // image.
  const int input_offset = H * W * C;
  const int output_offset = dY.size() / dY.dim(0);
  // The output image size is the spatial size of the output.
  const int output_image_size = dY.dim(1) * dY.dim(2);
  // The col buffer is stored in CHW order as well - kernel_dim, and the height
  // and width.
  col_buffer_.Reshape(std::vector<int>{output_image_size, kernel_dim});
  if (bias_multiplier_.size() != output_image_size) {
    // If the helper bias multiplier is not M, reshape and fill it with one.
    bias_multiplier_.Reshape(std::vector<int>(1, output_image_size));
    math::Set<T, Context>(
        output_image_size, static_cast<T>(1),
        bias_multiplier_.template mutable_data<T>(), &device_context_);
  }
  const T* Xdata = X.template data<T>();
  const T* const filter_data = filter.template data<T>();
  const T* const dYdata = dY.template data<T>();
  T* col_buffer_data = col_buffer_.template mutable_data<T>();
  T* dfilter_data = dfilter->template mutable_data<T>();
  T* dbias_data = dbias->template mutable_data<T>();
  // Pre-setting the gradients to zero.
  math::Set<T, Context>(dfilter->size(), 0, dfilter_data,
                                  &device_context_);
  math::Set<T, Context>(dbias->size(), 0, dbias_data,
                                  &device_context_);
  for (int image_id = 0; image_id < N; ++image_id) {
    // When we compute the gradient with respect to the filters, we need to do
    // im2col to allow gemm-type computation.
    math::Im2col<T, Context, StorageOrder::NHWC>(
        Xdata, C, H, W, kernel_h_, kernel_w_,
        pad_t_, pad_l_, pad_b_, pad_r_, stride_h_, stride_w_, col_buffer_data,
        &device_context_);
    // Gradient with respect to filter.
    math::Gemm<T, Context>(
        CblasTrans, CblasNoTrans, M, kernel_dim, output_image_size,
        kOne.template data<T>(), dYdata + output_offset * image_id, col_buffer_data,
        kOne.template data<T>(), dfilter_data, &device_context_);
    // Gradient with respect to bias
    math::Gemv<T, Context>(
        CblasTrans, output_image_size, M, kOne.template data<T>(),
        dYdata + output_offset * image_id, bias_multiplier_.template data<T>(),
        kOne.template data<T>(), dbias_data, &device_context_);
    Xdata += input_offset;
  }
  if (OutputSize() == 3) {
    // Compute the gradient w.r.t. the input.
    auto *dX = Output(INPUT_GRAD);
    dX->ReshapeLike(X);
    T* dXdata = dX->template mutable_data<T>();
    for (int image_id = 0; image_id < N; ++image_id) {
      // Compute gradient into col_buffer.
      math::Gemm<T, Context>(
          CblasNoTrans, CblasNoTrans, output_image_size, kernel_dim, M,
          kOne.template data<T>(), dYdata + output_offset * image_id, filter_data,
          kZero.template data<T>(), col_buffer_data, &device_context_);
      math::Col2im<T, Context, StorageOrder::NHWC>(
          col_buffer_data, C, H, W, kernel_h_, kernel_w_,
          pad_t_, pad_l_, pad_b_, pad_r_,
          stride_h_, stride_w_, dXdata, &device_context_);
      dXdata += input_offset;
    }
  }
  return true;
}
}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_CONV_OP_IMPL_H_
