/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

namespace {

void SetColumnBufferShapeImpl(
    const int N,
    const int /*C*/,
    const int kernel_dim,
    const StorageOrder order,
    const std::vector<int>& output_image_dims,
    std::vector<int>* column_dims,
    std::vector<int>* column_transposed_dims,
    std::vector<int>* column_axes,
    std::vector<int>* column_transposed_axes) {
  const int n_column_dims = output_image_dims.size() + 2;
  column_dims->resize(n_column_dims);
  column_transposed_dims->resize(n_column_dims);
  column_axes->resize(n_column_dims);
  if (order == StorageOrder::NCHW) {
    for (int i = 0; i < n_column_dims - 2; ++i) {
      column_dims->at(i + 2) = output_image_dims[i];
      column_transposed_dims->at(i) = output_image_dims[i];
      column_axes->at(i) = i + 2;
    }
    column_dims->at(0) = N;
    column_dims->at(1) = kernel_dim;
    column_transposed_dims->at(n_column_dims - 2) = kernel_dim;
    column_transposed_dims->at(n_column_dims - 1) = N;
    column_axes->at(n_column_dims - 1) = 0;
    column_axes->at(n_column_dims - 2) = 1;
  } else {
    for (int i = 0; i < n_column_dims - 2; ++i) {
      column_dims->at(i + 1) = output_image_dims[i];
      column_transposed_dims->at(i) = output_image_dims[i];
      column_axes->at(i) = i + 1;
    }
    column_dims->at(0) = N;
    column_dims->at(n_column_dims - 1) = kernel_dim;
    column_transposed_dims->at(n_column_dims - 2) = N;
    column_transposed_dims->at(n_column_dims - 1) = kernel_dim;
    column_axes->at(n_column_dims - 2) = 0;
    column_axes->at(n_column_dims - 1) = n_column_dims - 1;
  }
  if (column_transposed_axes != nullptr) {
    column_transposed_axes->resize(n_column_dims);
    for (int i = 0; i < n_column_dims; ++i) {
      column_transposed_axes->at(column_axes->at(i)) = i;
    }
  }
}

} // namespace

template <typename T, class Context>
bool LocallyConnectedOp<T, Context>::RunOnDeviceWithOrderNCHW() {
  const auto& X = Input(INPUT);
  const auto& filter = Input(FILTER);
  auto* Y = Output(0);
  const int image_ndim = X.ndim() - 2;
  CAFFE_ENFORCE_EQ(X.ndim() + image_ndim, filter.ndim());
  ShapeParams shape;
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
  const std::vector<int> input_dims(X.dims().cbegin() + 1, X.dims().cend());
  SetDeviceTensor(input_dims, &input_dims_device_);
  shape.kernel_dim = shape.C / group_ * kernel_dims_size;

  const std::vector<int> Y_dims(Y->dims().cbegin(), Y->dims().cend());
  SetColumnBufferShape(
      shape.N,
      shape.C,
      shape.kernel_dim,
      output_image_dims,
      &shape.column_dims,
      &shape.column_transposed_dims);
  SetYTranposedBufferShape(Y_dims, &shape.Y_transposed_dims);

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
  ShapeParams shape;
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

  shape.kernel_dim = kernel_h() * kernel_w() * shape.C;
  const std::vector<int> Y_dims(Y->dims().cbegin(), Y->dims().cend());
  SetColumnBufferShape(
      shape.N,
      shape.C,
      shape.kernel_dim,
      output_image_dims,
      &shape.column_dims,
      &shape.column_transposed_dims);
  SetYTranposedBufferShape(Y_dims, &shape.Y_transposed_dims);

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
    const ShapeParams& shape,
    const T* X_data,
    const T* filter_data,
    const T* bias_data,
    T* Y_data,
    Tensor<Context>* column_buffer,
    Tensor<Context>* column_transposed_buffer,
    Tensor<Context>* Y_transposed_buffer) {
  const int input_stride = shape.C / group_ * shape.input_image_size;
  const int column_stride = shape.kernel_dim * shape.output_image_size;
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
            input_dims_device_.template data<int>(),
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
      shape.kernel_dim,
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
    const ShapeParams& shape,
    const T* X_data,
    const T* filter_data,
    const T* bias_data,
    T* Y_data,
    Tensor<Context>* column_buffer,
    Tensor<Context>* column_transposed_buffer,
    Tensor<Context>* Y_transposed_buffer) {
  const int input_stride = shape.C * shape.input_image_size;
  const int column_stride = shape.kernel_dim * shape.output_image_size;
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
      shape.kernel_dim,
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
    const int kernel_dim,
    const std::vector<int>& output_image_dims,
    std::vector<int>* column_dims,
    std::vector<int>* column_transposed_dims) {
  std::vector<int> column_axes;
  SetColumnBufferShapeImpl(
      N,
      C,
      kernel_dim,
      order_,
      output_image_dims,
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
    const std::vector<int>& Y_dims,
    std::vector<int>* Y_transposed_dims) {
  const int n_Y_dims = Y_dims.size();
  Y_transposed_dims->resize(n_Y_dims);
  std::vector<int> Y_transposed_axes(n_Y_dims);
  if (order_ == StorageOrder::NCHW) {
    for (int i = 0; i < n_Y_dims - 2; ++i) {
      Y_transposed_dims->at(i) = Y_dims[i + 2];
      Y_transposed_axes[i + 2] = i;
    }
    Y_transposed_dims->at(n_Y_dims - 2) = Y_dims[1];
    Y_transposed_dims->at(n_Y_dims - 1) = Y_dims[0];
    Y_transposed_axes[1] = n_Y_dims - 2;
    Y_transposed_axes[0] = n_Y_dims - 1;
  } else {
    for (int i = 0; i < n_Y_dims - 2; ++i) {
      Y_transposed_dims->at(i) = Y_dims[i + 1];
      Y_transposed_axes[i + 1] = i;
    }
    Y_transposed_dims->at(n_Y_dims - 2) = Y_dims[0];
    Y_transposed_dims->at(n_Y_dims - 1) = Y_dims[n_Y_dims - 1];
    Y_transposed_axes[0] = n_Y_dims - 2;
    Y_transposed_axes[n_Y_dims - 1] = n_Y_dims - 1;
  }
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

  ShapeParams shape;
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

  const std::vector<int> input_dims(X.dims().cbegin() + 1, X.dims().cend());
  SetDeviceTensor(input_dims, &input_dims_device_);
  shape.kernel_dim = shape.C / group_ * kernel_dims_size;

  const std::vector<int> dY_dims(dY.dims().cbegin(), dY.dims().cend());
  SetColumnBufferShape(
      shape.N,
      shape.C,
      shape.kernel_dim,
      output_image_dims,
      &shape.column_dims,
      &shape.column_transposed_dims);
  SetDYTranposedBufferShape(dY_dims, &shape.dY_transposed_dims);

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
  ShapeParams shape;
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

  shape.kernel_dim = kernel_h() * kernel_w() * shape.C;
  const std::vector<int> dY_dims(dY.dims().cbegin(), dY.dims().cend());
  SetColumnBufferShape(
      shape.N,
      shape.C,
      shape.kernel_dim,
      output_image_dims,
      &shape.column_dims,
      &shape.column_transposed_dims);
  SetDYTranposedBufferShape(dY_dims, &shape.dY_transposed_dims);

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
    const ShapeParams& shape,
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
  const int column_stride = shape.kernel_dim * shape.output_image_size;
  column_buffer->Resize(shape.column_dims);
  column_transposed_buffer->Resize(shape.column_transposed_dims);
  dY_transposed_buffer->Resize(shape.dY_transposed_dims);
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
            input_dims_device_.template data<int>(),
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
      shape.dY_transposed_dims.size(),
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
      shape.kernel_dim,
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
        shape.kernel_dim,
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
              input_dims_device_.template data<int>(),
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
    const ShapeParams& shape,
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
  const int column_stride = shape.kernel_dim * shape.output_image_size;
  column_buffer->Resize(shape.column_dims);
  column_transposed_buffer->Resize(shape.column_transposed_dims);
  dY_transposed_buffer->Resize(shape.dY_transposed_dims);
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
      shape.dY_transposed_dims.size(),
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
      shape.kernel_dim,
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
        shape.kernel_dim,
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
    const int kernel_dim,
    const std::vector<int>& output_image_dims,
    std::vector<int>* column_dims,
    std::vector<int>* column_transposed_dims) {
  std::vector<int> column_axes;
  std::vector<int> column_transposed_axes;
  SetColumnBufferShapeImpl(
      N,
      C,
      kernel_dim,
      order_,
      output_image_dims,
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
    const std::vector<int>& dY_dims,
    std::vector<int>* dY_transposed_dims) {
  const int n_dY_dims = dY_dims.size();
  dY_transposed_dims->resize(n_dY_dims);
  std::vector<int> dY_axes(n_dY_dims);
  if (order_ == StorageOrder::NCHW) {
    for (int i = 0; i < n_dY_dims - 2; ++i) {
      dY_transposed_dims->at(i) = dY_dims[i + 2];
      dY_axes[i] = i + 2;
    }
    dY_transposed_dims->at(n_dY_dims - 2) = dY_dims[1];
    dY_transposed_dims->at(n_dY_dims - 1) = dY_dims[0];
    dY_axes[n_dY_dims - 2] = 1;
    dY_axes[n_dY_dims - 1] = 0;
  } else {
    for (int i = 0; i < n_dY_dims - 2; ++i) {
      dY_transposed_dims->at(i) = dY_dims[i + 1];
      dY_axes[i] = i + 1;
    }
    dY_transposed_dims->at(n_dY_dims - 2) = dY_dims[0];
    dY_transposed_dims->at(n_dY_dims - 1) = dY_dims[n_dY_dims - 1];
    dY_axes[n_dY_dims - 2] = 0;
    dY_axes[n_dY_dims - 1] = n_dY_dims - 1;
  }
  SetDeviceTensor(dY_dims, &dY_dims_device_);
  SetDeviceTensor(*dY_transposed_dims, &dY_transposed_dims_device_);
  SetDeviceTensor(dY_axes, &dY_axes_device_);
}

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LOCALLY_CONNECTED_OP_IMPL_H_
