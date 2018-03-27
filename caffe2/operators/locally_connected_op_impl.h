// locally_connected_impl.h is the templated implementation of the
// locally_connected.h file.

#ifndef CAFFE2_OPERATORS_LOCALLY_CONNECTED_OP_IMPL_H_
#define CAFFE2_OPERATORS_LOCALLY_CONNECTED_OP_IMPL_H_

#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/locally_connected_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
bool LocallyConnectedOp<T, Context>::RunOnDeviceWithOrderNCHW() {
  const auto& X = Input(INPUT);
  const auto& filter = Input(FILTER);
  auto* Y = Output(0);
  const int image_ndim = X.ndim() - 2;
  CAFFE_ENFORCE_EQ(X.ndim() + image_ndim, filter.ndim());
  lc_op_util::ShapeParams shape;
  shape.N = X.dim32(0);
  shape.C = X.dim32(1);
  shape.M = filter.dim32(image_ndim);
  CAFFE_ENFORCE(
      shape.C == filter.dim32(image_ndim + 1) * group_,
      "Locally Connected op: input channels does not match: "
      "# of input channels ",
      shape.C,
      " is not equal to kernel channels * group:",
      filter.dim32(image_ndim + 1),
      "*",
      group_);
  CAFFE_ENFORCE(
      shape.M % group_ == 0,
      "The number of output channels is not divisible by group.");

  ConvPoolOpBase<Context>::SetOutputSize(X, Y, shape.M);
  shape.input_image_size = GetDimsSize(X);
  shape.output_image_size = GetDimsSize(*Y);
  const std::vector<int> output_image_dims = GetDims(*Y);
  for (int i = 0; i < image_ndim; ++i) {
    CAFFE_ENFORCE(output_image_dims[i] == filter.dim32(i));
  }

  int kernel_dims_size = 1;
  for (std::size_t i = 0; i < kernel_.size(); ++i) {
    CAFFE_ENFORCE_EQ(filter.dim32(i + image_ndim + 2), kernel_[i]);
    kernel_dims_size *= kernel_[i];
  }

  shape.input_image_dims = GetDims(X);
  const std::vector<int> X_dims(X.dims().cbegin() + 1, X.dims().cend());
  SetDeviceTensor(X_dims, &X_dims_device_);
  shape.kernel_size = shape.C / group_ * kernel_dims_size;

  SetColumnBufferShape(
      shape.N,
      shape.C,
      shape.kernel_size,
      shape.output_image_size,
      &shape.column_dims,
      &shape.column_transposed_dims);
  SetYTranposedBufferShape(
      shape.N, shape.M, shape.output_image_size, &shape.Y_transposed_dims);

  const T* X_data = X.template data<T>();
  const T* filter_data = filter.template data<T>();
  const T* bias_data = nullptr;
  if (InputSize() == 3) {
    const auto& bias = Input(BIAS);
    CAFFE_ENFORCE(bias.ndim() == image_ndim + 1);
    for (int i = 0; i < image_ndim; ++i) {
      CAFFE_ENFORCE(bias.dim32(i) == output_image_dims[i]);
    }
    CAFFE_ENFORCE(bias.dim32(image_ndim) == shape.M);
    bias_data = bias.template data<T>();
    ConvPoolOpBase<Context>::template SetBiasMultiplier<T>(
        shape.N, &bias_multiplier_);
  }
  T* Y_data = Y->template mutable_data<T>();

  RunOnDeviceWithOrderNCHWImpl(
      shape,
      X_data,
      filter_data,
      bias_data,
      Y_data,
      &column_buffer_,
      &column_transposed_buffer_,
      &Y_transposed_buffer_);

  return true;
}

template <typename T, class Context>
bool LocallyConnectedOp<T, Context>::RunOnDeviceWithOrderNHWC() {
  const auto& X = Input(INPUT);
  const auto& filter = Input(FILTER);
  auto* Y = Output(0);
  CAFFE_ENFORCE_EQ(
      kernel_.size(),
      2,
      "Only 2d locally connected op is supported for NHWC storage type.");
  const int image_ndim = X.ndim() - 2;
  CAFFE_ENFORCE_EQ(X.ndim() + image_ndim, filter.ndim());
  lc_op_util::ShapeParams shape;
  shape.N = X.dim32(0);
  shape.C = X.dim32(3);
  shape.input_image_dims = {X.dim32(1), X.dim32(2)};
  shape.M = filter.dim32(image_ndim);
  CAFFE_ENFORCE(filter.dim32(image_ndim + 1) == kernel_h());
  CAFFE_ENFORCE(filter.dim32(image_ndim + 2) == kernel_w());
  CAFFE_ENFORCE(filter.dim32(image_ndim + 3) == shape.C);
  ConvPoolOpBase<Context>::SetOutputSize(X, Y, shape.M);

  shape.input_image_size = GetDimsSize(X);
  shape.output_image_size = GetDimsSize(*Y);
  const std::vector<int> output_image_dims = GetDims(*Y);
  for (int i = 0; i < image_ndim; ++i) {
    CAFFE_ENFORCE(output_image_dims[i] == filter.dim32(i));
  }

  shape.kernel_size = kernel_h() * kernel_w() * shape.C;
  SetColumnBufferShape(
      shape.N,
      shape.C,
      shape.kernel_size,
      shape.output_image_size,
      &shape.column_dims,
      &shape.column_transposed_dims);
  SetYTranposedBufferShape(
      shape.N, shape.M, shape.output_image_size, &shape.Y_transposed_dims);

  const T* X_data = X.template data<T>();
  const T* filter_data = filter.template data<T>();
  const T* bias_data = nullptr;
  if (InputSize() == 3) {
    const auto& bias = Input(BIAS);
    CAFFE_ENFORCE(bias.ndim() == image_ndim + 1);
    for (int i = 0; i < image_ndim; ++i) {
      CAFFE_ENFORCE(bias.dim32(i) == output_image_dims[i]);
    }
    CAFFE_ENFORCE(bias.dim32(image_ndim) == shape.M);
    bias_data = bias.template data<T>();
    ConvPoolOpBase<Context>::template SetBiasMultiplier<T>(
        shape.N, &bias_multiplier_);
  }
  T* Y_data = Y->template mutable_data<T>();

  RunOnDeviceWithOrderNHWCImpl(
      shape,
      X_data,
      filter_data,
      bias_data,
      Y_data,
      &column_buffer_,
      &column_transposed_buffer_,
      &Y_transposed_buffer_);

  return true;
}

template <typename T, class Context>
void LocallyConnectedOp<T, Context>::RunOnDeviceWithOrderNCHWImpl(
    const lc_op_util::ShapeParams& shape,
    const T* X_data,
    const T* filter_data,
    const T* bias_data,
    T* Y_data,
    Tensor<Context>* column_buffer,
    Tensor<Context>* column_transposed_buffer,
    Tensor<Context>* Y_transposed_buffer) {
  const int input_stride = shape.C / group_ * shape.input_image_size;
  const int column_stride = shape.kernel_size * shape.output_image_size;
  column_buffer->Resize(shape.column_dims);
  column_transposed_buffer->Resize(shape.column_transposed_dims);
  Y_transposed_buffer->Resize(shape.Y_transposed_dims);
  T* column_buffer_data = column_buffer->template mutable_data<T>();
  T* Y_transposed_buffer_data = Y_transposed_buffer->template mutable_data<T>();

  for (int image_id = 0; image_id < shape.N; ++image_id) {
    for (int group_id = 0; group_id < group_; ++group_id) {
      if (kernel_.size() == 2) {
        math::Im2col<T, Context, StorageOrder::NCHW>(
            X_data + group_id * input_stride,
            shape.C / group_,
            shape.input_image_dims[0],
            shape.input_image_dims[1],
            kernel_h(),
            kernel_w(),
            dilation_h(),
            dilation_w(),
            pad_t(),
            pad_l(),
            pad_b(),
            pad_r(),
            stride_h(),
            stride_w(),
            column_buffer_data + group_id * column_stride,
            &context_);
      } else {
        math::Im2colNd<T, Context, StorageOrder::NCHW>(
            X_data + group_id * input_stride,
            X_dims_device_.template data<int>(),
            column_dims_device_.template data<int>() + 1,
            shape.C * shape.input_image_size,
            column_stride,
            kernel_device_.template data<int>(),
            stride_device_.template data<int>(),
            dilation_device_.template data<int>(),
            pads_device_.template data<int>(),
            kernel_.size(),
            column_buffer_data + group_id * column_stride,
            &context_);
      }
    }
    X_data += input_stride * group_;
    column_buffer_data += column_stride * group_;
  }
  math::Transpose(
      shape.column_dims.size(),
      column_dims_device_.template data<int>(),
      column_transposed_dims_device_.template data<int>(),
      column_axes_device_.template data<int>(),
      column_buffer->size(),
      column_buffer->template data<T>(),
      column_transposed_buffer->template mutable_data<T>(),
      &context_);
  math::GemmBatched(
      CblasNoTrans,
      CblasNoTrans,
      shape.output_image_size * group_,
      shape.M / group_,
      shape.N,
      shape.kernel_size,
      1.0f,
      filter_data,
      column_transposed_buffer->template data<T>(),
      0.0f,
      Y_transposed_buffer_data,
      &context_);
  if (bias_data != nullptr) {
    math::Gemm<T, Context>(
        CblasNoTrans,
        CblasNoTrans,
        shape.output_image_size * shape.M,
        shape.N,
        1,
        1.0,
        bias_data,
        bias_multiplier_.template data<T>(),
        1.0,
        Y_transposed_buffer_data,
        &context_);
  }
  math::Transpose(
      shape.Y_transposed_dims.size(),
      Y_transposed_dims_device_.template data<int>(),
      Y_dims_device_.template data<int>(),
      Y_transposed_axes_device_.template data<int>(),
      Y_transposed_buffer->size(),
      Y_transposed_buffer_data,
      Y_data,
      &context_);
}

template <typename T, class Context>
void LocallyConnectedOp<T, Context>::RunOnDeviceWithOrderNHWCImpl(
    const lc_op_util::ShapeParams& shape,
    const T* X_data,
    const T* filter_data,
    const T* bias_data,
    T* Y_data,
    Tensor<Context>* column_buffer,
    Tensor<Context>* column_transposed_buffer,
    Tensor<Context>* Y_transposed_buffer) {
  const int input_stride = shape.C * shape.input_image_size;
  const int column_stride = shape.kernel_size * shape.output_image_size;
  column_buffer->Resize(shape.column_dims);
  column_transposed_buffer->Resize(shape.column_transposed_dims);
  Y_transposed_buffer->Resize(shape.Y_transposed_dims);
  T* column_buffer_data = column_buffer->template mutable_data<T>();
  T* Y_transposed_buffer_data = Y_transposed_buffer->template mutable_data<T>();
  for (int image_id = 0; image_id < shape.N; ++image_id) {
    math::Im2col<T, Context, StorageOrder::NHWC>(
        X_data + image_id * input_stride,
        shape.C,
        shape.input_image_dims[0],
        shape.input_image_dims[1],
        kernel_h(),
        kernel_w(),
        dilation_h(),
        dilation_w(),
        pad_t(),
        pad_l(),
        pad_b(),
        pad_r(),
        stride_h(),
        stride_w(),
        column_buffer_data + image_id * column_stride,
        &context_);
  }
  math::Transpose(
      shape.column_dims.size(),
      column_dims_device_.template data<int>(),
      column_transposed_dims_device_.template data<int>(),
      column_axes_device_.template data<int>(),
      column_buffer->size(),
      column_buffer->template data<T>(),
      column_transposed_buffer->template mutable_data<T>(),
      &context_);
  math::GemmBatched(
      CblasNoTrans,
      CblasTrans,
      shape.output_image_size,
      shape.N,
      shape.M,
      shape.kernel_size,
      1.0f,
      column_transposed_buffer->template data<T>(),
      filter_data,
      0.0f,
      Y_transposed_buffer_data,
      &context_);
  math::Transpose(
      shape.Y_transposed_dims.size(),
      Y_transposed_dims_device_.template data<int>(),
      Y_dims_device_.template data<int>(),
      Y_transposed_axes_device_.template data<int>(),
      Y_transposed_buffer->size(),
      Y_transposed_buffer_data,
      Y_data,
      &context_);
  if (bias_data != nullptr) {
    math::Gemm<T, Context>(
        CblasNoTrans,
        CblasNoTrans,
        shape.N,
        shape.output_image_size * shape.M,
        1,
        1.0f,
        bias_multiplier_.template data<T>(),
        bias_data,
        1.0f,
        Y_data,
        &context_);
  }
}

template <typename T, class Context>
void LocallyConnectedOp<T, Context>::SetColumnBufferShape(
    const int N,
    const int C,
    const int kernel_size,
    const int output_image_size,
    std::vector<int>* column_dims,
    std::vector<int>* column_transposed_dims) {
  std::vector<int> column_axes;
  lc_op_util::SetColumnBufferShapeImpl(
      N,
      kernel_size,
      output_image_size,
      order_,
      column_dims,
      column_transposed_dims,
      &column_axes,
      nullptr);
  SetDeviceTensor(*column_dims, &column_dims_device_);
  SetDeviceTensor(*column_transposed_dims, &column_transposed_dims_device_);
  SetDeviceTensor(column_axes, &column_axes_device_);
}

template <typename T, class Context>
void LocallyConnectedOp<T, Context>::SetYTranposedBufferShape(
    const int N,
    const int M,
    const int output_image_size,
    std::vector<int>* Y_transposed_dims) {
  std::vector<int> Y_dims;
  std::vector<int> Y_transposed_axes;
  lc_op_util::SetYBufferShapeImpl(
      N,
      M,
      output_image_size,
      order_,
      &Y_dims,
      Y_transposed_dims,
      nullptr,
      &Y_transposed_axes);
  SetDeviceTensor(Y_dims, &Y_dims_device_);
  SetDeviceTensor(*Y_transposed_dims, &Y_transposed_dims_device_);
  SetDeviceTensor(Y_transposed_axes, &Y_transposed_axes_device_);
}

template <typename T, class Context>
bool LocallyConnectedGradientOp<T, Context>::RunOnDeviceWithOrderNCHW() {
  const auto& X = Input(INPUT);
  const auto& filter = Input(FILTER);
  const auto& dY = Input(OUTPUT_GRAD);
  auto* dfilter = Output(FILTER_GRAD);
  const int image_ndim = X.ndim() - 2;
  CAFFE_ENFORCE_EQ(X.ndim() + image_ndim, filter.ndim());

  lc_op_util::ShapeParams shape;
  shape.N = X.dim32(0);
  shape.C = X.dim32(1);
  shape.M = filter.dim32(image_ndim);
  CAFFE_ENFORCE(filter.dim32(image_ndim + 1) * group_ == shape.C);
  CAFFE_ENFORCE(shape.M % group_ == 0);

  shape.input_image_dims = GetDims(X);
  shape.input_image_size = GetDimsSize(X);
  const std::vector<int> output_image_dims = GetDims(dY);
  shape.output_image_size = GetDimsSize(dY);
  for (int i = 0; i < image_ndim; ++i) {
    CAFFE_ENFORCE(output_image_dims[i] == filter.dim32(i));
  }
  ConvPoolOpBase<Context>::ComputePads(shape.input_image_dims);

  int kernel_dims_size = 1;
  for (std::size_t i = 0; i < kernel_.size(); ++i) {
    CAFFE_ENFORCE_EQ(filter.dim32(i + image_ndim + 2), kernel_[i]);
    kernel_dims_size *= kernel_[i];
  }

  const std::vector<int> X_dims(X.dims().cbegin() + 1, X.dims().cend());
  SetDeviceTensor(X_dims, &X_dims_device_);
  shape.kernel_size = shape.C / group_ * kernel_dims_size;

  SetColumnBufferShape(
      shape.N,
      shape.C,
      shape.kernel_size,
      shape.output_image_size,
      &shape.column_dims,
      &shape.column_transposed_dims);
  SetDYTranposedBufferShape(
      shape.N, shape.M, shape.output_image_size, &shape.Y_transposed_dims);

  dfilter->ResizeLike(filter);
  const T* X_data = X.template data<T>();
  const T* filter_data = filter.template data<T>();
  const T* dY_data = dY.template data<T>();
  T* dfilter_data = dfilter->template mutable_data<T>();
  T* dX_data = nullptr;
  T* dbias_data = nullptr;
  if (OutputSize() == 3 || (no_bias_ && OutputSize() == 2)) {
    auto* dX = Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
    dX->ResizeLike(X);
    dX_data = dX->template mutable_data<T>();
  }
  if (!no_bias_) {
    auto* dbias = Output(BIAS_OR_INPUT_GRAD);
    std::vector<int> dbias_dims = output_image_dims;
    dbias_dims.push_back(shape.M);
    dbias->Resize(dbias_dims);
    ConvPoolOpBase<Context>::template SetBiasMultiplier<T>(
        shape.N, &bias_multiplier_);
    dbias_data = dbias->template mutable_data<T>();
  }
  RunOnDeviceWithOrderNCHWImpl(
      shape,
      X_data,
      filter_data,
      dY_data,
      dfilter_data,
      dX_data,
      dbias_data,
      &column_buffer_,
      &column_transposed_buffer_,
      &dY_transposed_buffer_);

  return true;
}

template <typename T, class Context>
bool LocallyConnectedGradientOp<T, Context>::RunOnDeviceWithOrderNHWC() {
  const auto& X = Input(INPUT);
  const auto& filter = Input(FILTER);
  const auto& dY = Input(OUTPUT_GRAD);
  auto* dfilter = Output(FILTER_GRAD);
  CAFFE_ENFORCE_EQ(
      kernel_.size(),
      2,
      "Only 2d locally connected op is supported for NHWC storage type.");
  const int image_ndim = X.ndim() - 2;
  CAFFE_ENFORCE_EQ(X.ndim() + image_ndim, filter.ndim());
  lc_op_util::ShapeParams shape;
  shape.N = X.dim32(0);
  shape.C = X.dim32(3);
  shape.input_image_dims = {X.dim32(1), X.dim32(2)};
  shape.M = filter.dim32(image_ndim);
  CAFFE_ENFORCE(filter.dim32(image_ndim + 1) == kernel_h());
  CAFFE_ENFORCE(filter.dim32(image_ndim + 2) == kernel_w());
  CAFFE_ENFORCE(filter.dim32(image_ndim + 3) == shape.C);
  ConvPoolOpBase<Context>::ComputePads(shape.input_image_dims);

  shape.input_image_size = GetDimsSize(X);
  shape.output_image_size = GetDimsSize(dY);
  const std::vector<int> output_image_dims = GetDims(dY);
  for (int i = 0; i < image_ndim; ++i) {
    CAFFE_ENFORCE(output_image_dims[i] == filter.dim32(i));
  }

  shape.kernel_size = kernel_h() * kernel_w() * shape.C;
  SetColumnBufferShape(
      shape.N,
      shape.C,
      shape.kernel_size,
      shape.output_image_size,
      &shape.column_dims,
      &shape.column_transposed_dims);
  SetDYTranposedBufferShape(
      shape.N, shape.M, shape.output_image_size, &shape.Y_transposed_dims);

  dfilter->ResizeLike(filter);
  const T* X_data = X.template data<T>();
  const T* filter_data = filter.template data<T>();
  const T* dY_data = dY.template data<T>();
  T* dfilter_data = dfilter->template mutable_data<T>();
  T* dX_data = nullptr;
  T* dbias_data = nullptr;
  if (OutputSize() == 3 || (no_bias_ && OutputSize() == 2)) {
    auto* dX = Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
    dX->ResizeLike(X);
    dX_data = dX->template mutable_data<T>();
  }
  if (!no_bias_) {
    auto* dbias = Output(BIAS_OR_INPUT_GRAD);
    std::vector<int> dbias_dims = output_image_dims;
    dbias_dims.push_back(shape.M);
    dbias->Resize(dbias_dims);
    ConvPoolOpBase<Context>::template SetBiasMultiplier<T>(
        shape.N, &bias_multiplier_);
    dbias_data = dbias->template mutable_data<T>();
  }
  RunOnDeviceWithOrderNHWCImpl(
      shape,
      X_data,
      filter_data,
      dY_data,
      dfilter_data,
      dX_data,
      dbias_data,
      &column_buffer_,
      &column_transposed_buffer_,
      &dY_transposed_buffer_);

  return true;
}

template <typename T, class Context>
void LocallyConnectedGradientOp<T, Context>::RunOnDeviceWithOrderNCHWImpl(
    const lc_op_util::ShapeParams& shape,
    const T* X_data,
    const T* filter_data,
    const T* dY_data,
    T* dfilter_data,
    T* dX_data,
    T* dbias_data,
    Tensor<Context>* column_buffer,
    Tensor<Context>* column_transposed_buffer,
    Tensor<Context>* dY_transposed_buffer) {
  const int input_stride = shape.C * shape.input_image_size;
  const int column_stride = shape.kernel_size * shape.output_image_size;
  column_buffer->Resize(shape.column_dims);
  column_transposed_buffer->Resize(shape.column_transposed_dims);
  dY_transposed_buffer->Resize(shape.Y_transposed_dims);
  T* column_buffer_data = column_buffer->template mutable_data<T>();
  T* dY_transposed_buffer_data =
      dY_transposed_buffer->template mutable_data<T>();

  for (int image_id = 0; image_id < shape.N; ++image_id) {
    for (int group_id = 0; group_id < group_; ++group_id) {
      if (kernel_.size() == 2) {
        math::Im2col<T, Context, StorageOrder::NCHW>(
            X_data + group_id * input_stride,
            shape.C / group_,
            shape.input_image_dims[0],
            shape.input_image_dims[1],
            kernel_h(),
            kernel_w(),
            dilation_h(),
            dilation_w(),
            pad_t(),
            pad_l(),
            pad_b(),
            pad_r(),
            stride_h(),
            stride_w(),
            column_buffer_data + group_id * column_stride,
            &context_);
      } else {
        math::Im2colNd<T, Context, StorageOrder::NCHW>(
            X_data + group_id * input_stride,
            X_dims_device_.template data<int>(),
            column_dims_device_.template data<int>() + 1,
            shape.C * shape.input_image_size,
            column_stride,
            kernel_device_.template data<int>(),
            stride_device_.template data<int>(),
            dilation_device_.template data<int>(),
            pads_device_.template data<int>(),
            kernel_.size(),
            column_buffer_data + group_id * column_stride,
            &context_);
      }
    }
    X_data += input_stride * group_;
    column_buffer_data += column_stride * group_;
  }
  math::Transpose(
      shape.column_dims.size(),
      column_dims_device_.template data<int>(),
      column_transposed_dims_device_.template data<int>(),
      column_axes_device_.template data<int>(),
      column_buffer->size(),
      column_buffer->template data<T>(),
      column_transposed_buffer->template mutable_data<T>(),
      &context_);

  math::Transpose(
      shape.Y_transposed_dims.size(),
      dY_dims_device_.template data<int>(),
      dY_transposed_dims_device_.template data<int>(),
      dY_axes_device_.template data<int>(),
      dY_transposed_buffer->size(),
      dY_data,
      dY_transposed_buffer_data,
      &context_);

  // Gradient respect to filter.
  math::GemmBatched(
      CblasNoTrans,
      CblasTrans,
      shape.output_image_size * group_,
      shape.M / group_,
      shape.kernel_size,
      shape.N,
      1.0f,
      dY_transposed_buffer_data,
      column_transposed_buffer->template data<T>(),
      0.0f,
      dfilter_data,
      &context_);

  if (dbias_data != nullptr) {
    // Gradient respect to bias.
    math::Gemv<T, Context>(
        CblasNoTrans,
        shape.output_image_size * shape.M,
        shape.N,
        1.0f,
        dY_transposed_buffer_data,
        bias_multiplier_.template data<T>(),
        0.0f,
        dbias_data,
        &context_);
  }

  if (dX_data != nullptr) {
    // Gradient respect to X.
    math::GemmBatched(
        CblasTrans,
        CblasNoTrans,
        shape.output_image_size * group_,
        shape.kernel_size,
        shape.N,
        shape.M / group_,
        1.0f,
        filter_data,
        dY_transposed_buffer_data,
        0.0f,
        column_transposed_buffer->template mutable_data<T>(),
        &context_);
    math::Transpose(
        shape.column_dims.size(),
        column_transposed_dims_device_.template data<int>(),
        column_dims_device_.template data<int>(),
        column_transposed_axes_device_.template data<int>(),
        column_transposed_buffer->size(),
        column_transposed_buffer->template data<T>(),
        column_buffer->template mutable_data<T>(),
        &context_);
    const T* const_column_buffer_data = column_buffer->template data<T>();
    for (int image_id = 0; image_id < shape.N; ++image_id) {
      for (int group_id = 0; group_id < group_; ++group_id) {
        if (kernel_.size() == 2) {
          math::Col2im<T, Context, StorageOrder::NCHW>(
              const_column_buffer_data + group_id * column_stride,
              shape.C / group_,
              shape.input_image_dims[0],
              shape.input_image_dims[1],
              kernel_h(),
              kernel_w(),
              dilation_h(),
              dilation_w(),
              pad_t(),
              pad_l(),
              pad_b(),
              pad_r(),
              stride_h(),
              stride_w(),
              dX_data + group_id * input_stride,
              &context_);
        } else {
          math::Col2imNd<T, Context, StorageOrder::NCHW>(
              const_column_buffer_data + group_id * column_stride,
              X_dims_device_.template data<int>(),
              column_dims_device_.template data<int>() + 1,
              shape.C * shape.input_image_size,
              column_stride,
              kernel_device_.template data<int>(),
              stride_device_.template data<int>(),
              dilation_device_.template data<int>(),
              pads_device_.template data<int>(),
              kernel_.size(),
              dX_data + group_id * input_stride,
              &context_);
        }
      }
      dX_data += input_stride * group_;
      const_column_buffer_data += column_stride * group_;
    }
  }
}

template <typename T, class Context>
void LocallyConnectedGradientOp<T, Context>::RunOnDeviceWithOrderNHWCImpl(
    const lc_op_util::ShapeParams& shape,
    const T* X_data,
    const T* filter_data,
    const T* dY_data,
    T* dfilter_data,
    T* dX_data,
    T* dbias_data,
    Tensor<Context>* column_buffer,
    Tensor<Context>* column_transposed_buffer,
    Tensor<Context>* dY_transposed_buffer) {
  const int input_stride = shape.C * shape.input_image_size;
  const int column_stride = shape.kernel_size * shape.output_image_size;
  column_buffer->Resize(shape.column_dims);
  column_transposed_buffer->Resize(shape.column_transposed_dims);
  dY_transposed_buffer->Resize(shape.Y_transposed_dims);
  T* column_buffer_data = column_buffer->template mutable_data<T>();
  T* dY_transposed_buffer_data =
      dY_transposed_buffer->template mutable_data<T>();
  for (int image_id = 0; image_id < shape.N; ++image_id) {
    math::Im2col<T, Context, StorageOrder::NHWC>(
        X_data + image_id * input_stride,
        shape.C,
        shape.input_image_dims[0],
        shape.input_image_dims[1],
        kernel_h(),
        kernel_w(),
        dilation_h(),
        dilation_w(),
        pad_t(),
        pad_l(),
        pad_b(),
        pad_r(),
        stride_h(),
        stride_w(),
        column_buffer_data + image_id * column_stride,
        &context_);
  }
  math::Transpose(
      shape.column_dims.size(),
      column_dims_device_.template data<int>(),
      column_transposed_dims_device_.template data<int>(),
      column_axes_device_.template data<int>(),
      column_buffer->size(),
      column_buffer->template data<T>(),
      column_transposed_buffer->template mutable_data<T>(),
      &context_);

  math::Transpose(
      shape.Y_transposed_dims.size(),
      dY_dims_device_.template data<int>(),
      dY_transposed_dims_device_.template data<int>(),
      dY_axes_device_.template data<int>(),
      dY_transposed_buffer->size(),
      dY_data,
      dY_transposed_buffer_data,
      &context_);

  // Gradient respect to filter.
  math::GemmBatched(
      CblasTrans,
      CblasNoTrans,
      shape.output_image_size,
      shape.M,
      shape.kernel_size,
      shape.N,
      1.0f,
      dY_transposed_buffer_data,
      column_transposed_buffer->template data<T>(),
      0.0f,
      dfilter_data,
      &context_);

  if (dbias_data != nullptr) {
    // Gradient respect to bias.
    math::Gemv<T, Context>(
        CblasTrans,
        shape.N,
        shape.output_image_size * shape.M,
        1.0f,
        dY_data,
        bias_multiplier_.template data<T>(),
        0.0f,
        dbias_data,
        &context_);
  }

  if (dX_data != nullptr) {
    // Gradient respect to X.
    math::GemmBatched(
        CblasNoTrans,
        CblasNoTrans,
        shape.output_image_size,
        shape.N,
        shape.kernel_size,
        shape.M,
        1.0f,
        dY_transposed_buffer_data,
        filter_data,
        0.0f,
        column_transposed_buffer->template mutable_data<T>(),
        &context_);
    math::Transpose(
        shape.column_dims.size(),
        column_transposed_dims_device_.template data<int>(),
        column_dims_device_.template data<int>(),
        column_transposed_axes_device_.template data<int>(),
        column_transposed_buffer->size(),
        column_transposed_buffer->template data<T>(),
        column_buffer->template mutable_data<T>(),
        &context_);
    const T* const_column_buffer_data = column_buffer->template data<T>();
    for (int image_id = 0; image_id < shape.N; ++image_id) {
      math::Col2im<T, Context, StorageOrder::NHWC>(
          const_column_buffer_data,
          shape.C,
          shape.input_image_dims[0],
          shape.input_image_dims[1],
          kernel_h(),
          kernel_w(),
          dilation_h(),
          dilation_w(),
          pad_t(),
          pad_l(),
          pad_b(),
          pad_r(),
          stride_h(),
          stride_w(),
          dX_data,
          &context_);
      dX_data += input_stride;
      const_column_buffer_data += column_stride;
    }
  }
}

template <typename T, class Context>
void LocallyConnectedGradientOp<T, Context>::SetColumnBufferShape(
    const int N,
    const int C,
    const int kernel_size,
    const int output_image_size,
    std::vector<int>* column_dims,
    std::vector<int>* column_transposed_dims) {
  std::vector<int> column_axes;
  std::vector<int> column_transposed_axes;
  lc_op_util::SetColumnBufferShapeImpl(
      N,
      kernel_size,
      output_image_size,
      order_,
      column_dims,
      column_transposed_dims,
      &column_axes,
      &column_transposed_axes);
  SetDeviceTensor(*column_dims, &column_dims_device_);
  SetDeviceTensor(*column_transposed_dims, &column_transposed_dims_device_);
  SetDeviceTensor(column_axes, &column_axes_device_);
  SetDeviceTensor(column_transposed_axes, &column_transposed_axes_device_);
}

template <typename T, class Context>
void LocallyConnectedGradientOp<T, Context>::SetDYTranposedBufferShape(
    const int N,
    const int M,
    const int output_image_size,
    std::vector<int>* dY_transposed_dims) {
  std::vector<int> dY_dims;
  std::vector<int> dY_axes;
  lc_op_util::SetYBufferShapeImpl(
      N,
      M,
      output_image_size,
      order_,
      &dY_dims,
      dY_transposed_dims,
      &dY_axes,
      nullptr);
  SetDeviceTensor(dY_dims, &dY_dims_device_);
  SetDeviceTensor(*dY_transposed_dims, &dY_transposed_dims_device_);
  SetDeviceTensor(dY_axes, &dY_axes_device_);
}

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LOCALLY_CONNECTED_OP_IMPL_H_
