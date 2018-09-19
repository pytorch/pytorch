// conv_op_impl.h is the templated implementation of the conv_op.h file.
#ifndef CAFFE2_OPERATORS_CONV_OP_IMPL_H_
#define CAFFE2_OPERATORS_CONV_OP_IMPL_H_

#include "caffe2/operators/conv_op.h"

#include <array>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
bool ConvOp<T, Context>::RunOnDeviceWithOrderNCHW() {
  const auto& X = Input(INPUT);
  const auto& filter = Input(FILTER);
  auto* Y = Output(0);
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int G = group_;
  CAFFE_ENFORCE_EQ(X.ndim(), filter.ndim());
  const int M = filter.dim32(0);
  CAFFE_ENFORCE_EQ(
      C,
      filter.dim32(1) * G,
      "Convolution op: input channels does not match: # of input channels ",
      C,
      " is not equal to kernel channels * group: ",
      filter.dim32(1),
      "*",
      G);
  CAFFE_ENFORCE_EQ(
      M % G, 0, "The number of output channels is not divisible by group.");

  int kernel_size = 1;
  for (std::size_t i = 0; i < kernel_.size(); ++i) {
    CAFFE_ENFORCE_EQ(filter.dim32(i + 2), kernel_[i]);
    kernel_size *= kernel_[i];
  }
  ConvPoolOpBase<Context>::SetOutputSize(X, Y, M);
  const vector<int> X_dims = GetDims(X);
  const vector<int> Y_dims = GetDims(*Y);
  const int X_HxW = X.size() / (N * C);
  const int Y_HxW = Y->size() / (N * M);
  const vector<int> img_shape(X.dims().cbegin() + 1, X.dims().cend());
  vector<int> buffer_shape(Y_dims.size() + 1);
  buffer_shape[0] = C * kernel_size;
  std::copy(Y_dims.cbegin(), Y_dims.cend(), buffer_shape.begin() + 1);

  const int buffer_size = C * kernel_size * Y_HxW;

  // The dimension of each kernel
  const int kernel_dim = C / G * kernel_size;
  const int X_stride = C * X_HxW;
  const int Y_stride = M * Y_HxW;
  const int filter_stride = filter.size() / G;

  // The col buffer is stored in CHW order as well - kernel_dim, and the height
  // and width.
  const T* X_data = X.template data<T>();
  const T* filter_data = filter.template data<T>();
  const T* bias_data = nullptr;
  if (InputSize() == 3) {
    const auto& bias = Input(BIAS);
    CAFFE_ENFORCE_EQ(bias.ndim(), 1);
    CAFFE_ENFORCE_EQ(bias.dim32(0), M);
    bias_data = bias.template data<T>();
    ConvPoolOpBase<Context>::template SetBiasMultiplier<T>(
        Y_HxW, &bias_multiplier_);
  }
  T* Y_data = Y->template mutable_data<T>();

  // Shortcut for 1x1 conv.
  if (kernel_size == 1 && !HasPad() && !HasStride()) {
    return Run1x1ConvOnDeviceWithOrderNCHW(
        N, C, X_HxW, M, X_data, filter_data, bias_data, Y_data);
  }

  const auto func = [&](Tensor* col_buffer) {
    col_buffer->Resize(buffer_shape);
    T* col_buffer_data = col_buffer->template mutable_data<T>();
    // Im2Col, followed by gemm.
    for (int image_id = 0; image_id < N; ++image_id) {
      if (kernel_.size() == 2) {
        math::Im2Col<T, Context, StorageOrder::NCHW>(
            C,
            X_dims[0],
            X_dims[1],
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
            X_data,
            col_buffer_data,
            &context_);
      } else {
        math::Im2ColNd<T, Context, StorageOrder::NCHW>(
            kernel_.size(),
            C * X_HxW,
            buffer_size,
            img_shape.data(),
            buffer_shape.data(),
            kernel_.data(),
            stride_.data(),
            dilation_.data(),
            pads_.data(),
            X_data,
            col_buffer_data,
            &context_);
      }
      // Weight term
      if (G == 1) {
        math::Gemm<T, Context>(
            CblasNoTrans,
            CblasNoTrans,
            M,
            Y_HxW,
            kernel_dim,
            1.0f,
            filter_data,
            col_buffer_data,
            0.0f,
            Y_data,
            &context_);
      } else {
        math::GemmStridedBatched<T, Context>(
            CblasNoTrans,
            CblasNoTrans,
            G,
            M / G,
            Y_HxW,
            kernel_dim,
            1.0f,
            filter_data,
            filter_stride,
            col_buffer_data,
            buffer_size / G,
            0.0f,
            Y_data,
            Y_stride / G,
            &context_);
      }
      if (bias_data != nullptr) {
        // Bias term can be carried out outside the group definition
        // to be efficient.
        math::Gemm<T, Context>(
            CblasNoTrans,
            CblasNoTrans,
            M,
            Y_HxW,
            1,
            1.0f,
            bias_data,
            bias_multiplier_.template data<T>(),
            1.0f,
            Y_data,
            &context_);
      }
      X_data += X_stride;
      Y_data += Y_stride;
    }
  };
  if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
    runWithSharedBuffer<Context>(ws_, func);
  } else {
    func(&col_buffer_);
  }
  return true;
}

// The implementations.
template <typename T, class Context>
bool ConvOp<T, Context>::RunOnDeviceWithOrderNHWC() {
  CAFFE_ENFORCE_EQ(
      kernel_.size(),
      2,
      "Only 2d convolution is supported for NHWC storage type");

  const Tensor& X = Input(INPUT);
  auto& filter = Input(FILTER);
  Tensor* Y = Output(0);
  CAFFE_ENFORCE_EQ(X.ndim(), 4);
  const int N = X.dim32(0), H = X.dim32(1), W = X.dim32(2), C = X.dim32(3);

  CAFFE_ENFORCE_EQ(X.ndim(), filter.ndim());
  const int M = filter.dim32(0);
  CAFFE_ENFORCE_EQ(filter.dim32(1), kernel_h());
  CAFFE_ENFORCE_EQ(filter.dim32(2), kernel_w());
  CAFFE_ENFORCE_EQ(filter.dim32(3), C / group_);

  ConvPoolOpBase<Context>::SetOutputSize(X, Y, filter.dim32(0));
  // The dimension of each kernel
  const int kernel_dim = kernel_h() * kernel_w() * (C / group_);
  // The offset corresponding to a single input image, and a single output
  // image.
  const int input_offset = H * W * C;
  const int output_offset = Y->size() / Y->dim32(0);
  // The output image size is the spatial size of the output.
  const int output_image_size = Y->dim32(1) * Y->dim32(2);
  // The col buffer is stored in HWC order as well - kernel_dim, and the height
  // and width.
  const T* X_data = X.template data<T>();
  const T* filter_data = filter.template data<T>();
  const T* bias_data = nullptr;
  T* Y_data = Y->template mutable_data<T>();
  if (InputSize() == 3) {
    const auto& bias = Input(BIAS);
    CAFFE_ENFORCE_EQ(bias.ndim(), 1);
    CAFFE_ENFORCE_EQ(bias.dim32(0), M);
    bias_data = bias.template data<T>();
  }
  // Specialized path for 1 by 1 convolution with stride 1, pad 0 - we
  // can skip im2col.
  if (kernel_dim == (C / group_) && !HasPad() && !HasStride()) {
    const int HxW = X.size() / (N * C);
    if (bias_data != nullptr) {
      ConvPoolOpBase<Context>::template SetBiasMultiplier<T>(
          N * HxW, &bias_multiplier_);
    }
    return Run1x1ConvOnDeviceWithOrderNHWC(
        N, C, HxW, M, X_data, filter_data, bias_data, Y_data);
  }

  if (bias_data != nullptr) {
    ConvPoolOpBase<Context>::template SetBiasMultiplier<T>(
        output_image_size, &bias_multiplier_);
  }
  auto f = [&](Tensor* col_buffer) {
    col_buffer->Resize(
        vector<TIndex>{Y->dim32(1), Y->dim32(2), kernel_h(), kernel_w(), C});
    T* col_buffer_data = col_buffer->template mutable_data<T>();
    // Im2Col, followed by gemm.
    for (int image_id = 0; image_id < N; ++image_id) {
      math::Im2Col<T, Context, StorageOrder::NHWC>(
          C,
          H,
          W,
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
          X_data,
          col_buffer_data,
          &context_,
          group_);
      // Weight term
      for (int group_id = 0; group_id < group_; ++group_id) {
        math::GemmEx<T, Context>(
            CblasNoTrans,
            CblasTrans,
            output_image_size,
            M / group_,
            kernel_dim,
            1,
            col_buffer_data + group_id * kernel_dim,
            group_ * kernel_dim,
            filter_data + group_id * (M / group_) * kernel_dim,
            kernel_dim,
            0,
            Y_data + group_id * (M / group_),
            M,
            &context_);
      }
      if (bias_data != nullptr) {
        // Bias term
        math::Gemm<T, Context>(
            CblasNoTrans,
            CblasNoTrans,
            output_image_size,
            M,
            1,
            1,
            bias_multiplier_.template data<T>(),
            bias_data,
            1,
            Y_data,
            &context_);
      }
      X_data += input_offset;
      Y_data += output_offset;
    }
  };
  if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
    runWithSharedBuffer<Context>(ws_, f);
  } else {
    f(&col_buffer_);
  }
  return true;
}

template <typename T, class Context>
bool ConvOp<T, Context>::Run1x1ConvOnDeviceWithOrderNCHW(
    const int N,
    const int C,
    const int HxW,
    const int M,
    const T* X,
    const T* filter,
    const T* bias,
    T* Y) {
  const int G = group_;
  if (G == 1) {
    math::GemmStridedBatched<T, Context>(
        CblasNoTrans,
        CblasNoTrans,
        N,
        M,
        HxW,
        C,
        1.0f,
        filter,
        0,
        X,
        C * HxW,
        0.0f,
        Y,
        M * HxW,
        &context_);
  } else {
    const int batch_size = N * G;
    const int D_X = C / G;
    const int D_Y = M / G;
    const int X_stride = D_X * HxW;
    const int W_stride = D_Y * D_X;
    const int Y_stride = D_Y * HxW;
    std::vector<const T*> X_ptr(N * G);
    std::vector<const T*> W_ptr(N * G);
    std::vector<T*> Y_ptr(N * G);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < G; ++j) {
        const int index = i * G + j;
        X_ptr[index] = X + index * X_stride;
        W_ptr[index] = filter + j * W_stride;
        Y_ptr[index] = Y + index * Y_stride;
      }
    }
    math::GemmBatched<T, Context>(
        CblasNoTrans,
        CblasNoTrans,
        batch_size,
        D_Y,
        HxW,
        D_X,
        1.0f,
        W_ptr.data(),
        X_ptr.data(),
        0.0f,
        Y_ptr.data(),
        &context_);
  }
  if (bias != nullptr) {
    const T* bias_multiplier_data = bias_multiplier_.template data<T>();
    math::GemmStridedBatched<T, Context>(
        CblasNoTrans,
        CblasNoTrans,
        N,
        M,
        HxW,
        1,
        1.0f,
        bias,
        0,
        bias_multiplier_data,
        0,
        1.0f,
        Y,
        M * HxW,
        &context_);
  }
  return true;
}

template <typename T, class Context>
bool ConvOp<T, Context>::Run1x1ConvOnDeviceWithOrderNHWC(
    const int N,
    const int C,
    const int HxW,
    const int M,
    const T* X,
    const T* filter,
    const T* bias,
    T* Y) {
  const int G = group_;
  const int kernel_dim = C / G;
  for (int group_id = 0; group_id < group_; ++group_id) {
    math::GemmEx<T, Context>(
        CblasNoTrans,
        CblasTrans,
        N * HxW,
        M / group_,
        kernel_dim,
        1.0f,
        X + group_id * kernel_dim,
        C,
        filter + group_id * (M / group_) * kernel_dim,
        kernel_dim,
        0.0f,
        Y + group_id * (M / group_),
        M,
        &context_);
  }
  if (bias != nullptr) {
    const T* bias_multiplier_data = bias_multiplier_.template data<T>();
    math::Gemm<T, Context>(
        CblasNoTrans,
        CblasNoTrans,
        N * HxW,
        M,
        1,
        1.0f,
        bias_multiplier_data,
        bias,
        1.0f,
        Y,
        &context_);
  }
  return true;
}

template <typename T, class Context>
bool ConvGradientOp<T, Context>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto& dY = Input(OUTPUT_GRAD);
  auto* dfilter = Output(FILTER_GRAD);
  const int N = X.dim32(0), C = X.dim32(1);

  const vector<int> input_dims = this->GetDims(X);
  const int input_image_size = this->GetDimsSize(X);

  const vector<int> output_dims = this->GetDims(dY);
  // The output image size is the spatial size of the output.
  const int output_image_size = this->GetDimsSize(dY);

  ConvPoolOpBase<Context>::ComputePads(input_dims);
  CAFFE_ENFORCE_EQ(X.ndim(), filter.ndim());
  const int M = filter.dim32(0);
  CAFFE_ENFORCE(filter.dim32(1) * group_ == C);

  int kernel_dims_size = 1;
  for (int i = 0; i < kernel_.size(); ++i) {
    CAFFE_ENFORCE(filter.dim32(i + 2) == kernel_[i]);
    kernel_dims_size *= kernel_[i];
  }

  CAFFE_ENFORCE(M % group_ == 0);
  dfilter->ResizeLike(filter);
  // The dimension of each kernel
  const int kernel_dim = C / group_ * kernel_dims_size;
  // The offset corresponding to a single input image, and a single output
  // image.
  const int input_offset = C / group_ * input_image_size;
  const int output_offset = dY.size() / dY.dim32(0) / group_;
  const int filter_offset = filter.size() / group_;
  // The col buffer is stored in CHW order as well - kernel_dim, and the height
  // and width.

  vector<int> img_shape;
  img_shape.assign(X.dims().begin() + 1, X.dims().end());
  vector<int> col_buffer_shape;
  col_buffer_shape.push_back(C / group_ * kernel_dims_size);
  col_buffer_shape.insert(
      col_buffer_shape.end(), output_dims.begin(), output_dims.end());
  col_buffer_.Resize(col_buffer_shape);

  if (kernel_.size() != 2) {
    SetDeviceTensor(img_shape, &img_shape_device_);
    SetDeviceTensor(col_buffer_shape, &col_buffer_shape_device_);
  }

  const int col_buffer_size =
      (C / group_) * kernel_dims_size * output_image_size;
  const T* Xdata = X.template data<T>();
  const T* filter_data = filter.template data<T>();
  const T* dYdata = dY.template data<T>();
  T* col_buffer_data = col_buffer_.template mutable_data<T>();
  T* dfilter_data = dfilter->template mutable_data<T>();

  // Pre-setting the gradients to zero.
  math::Set<T, Context>(dfilter->size(), 0, dfilter_data, &context_);

  T* dbias_data = nullptr;
  if (!no_bias_) {
    auto* dbias = Output(BIAS_OR_INPUT_GRAD);
    dbias->Resize(M);
    if (bias_multiplier_.size() != output_image_size) {
      // If the helper bias multiplier is not M, reshape and fill it with one.
      bias_multiplier_.Resize(vector<TIndex>(1, output_image_size));
      math::Set<T, Context>(
          output_image_size,
          static_cast<T>(1),
          bias_multiplier_.template mutable_data<T>(),
          &context_);
    }
    dbias_data = dbias->template mutable_data<T>();
    math::Set<T, Context>(dbias->size(), 0, dbias_data, &context_);
  }

  for (int image_id = 0; image_id < N; ++image_id) {
    for (int group_id = 0; group_id < group_; ++group_id) {
      // When we compute the gradient with respect to the filters, we need to do
      // im2col to allow gemm-type computation.
      if (kernel_.size() == 2) {
        math::Im2Col<T, Context, StorageOrder::NCHW>(
            C / group_,
            input_dims[0],
            input_dims[1],
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
            Xdata + group_id * input_offset,
            col_buffer_data,
            &context_);
      } else {
        math::Im2ColNd<T, Context, StorageOrder::NCHW>(
            kernel_.size(),
            C * input_image_size,
            col_buffer_size,
            img_shape.data(),
            col_buffer_shape.data(),
            kernel_.data(),
            stride_.data(),
            dilation_.data(),
            pads_.data(),
            Xdata + group_id * input_offset,
            col_buffer_data,
            &context_);
      }
      // Gradient with respect to filter.
      math::Gemm<T, Context>(
          CblasNoTrans,
          CblasTrans,
          M / group_,
          kernel_dim,
          output_image_size,
          1,
          dYdata + group_id * output_offset,
          col_buffer_data,
          1,
          dfilter_data + group_id * filter_offset,
          &context_);
    }
    if (!no_bias_) {
      // Gradient with respect to bias can be computed independent from group.
      math::Gemv<T, Context>(
          CblasNoTrans,
          M,
          output_image_size,
          1,
          dYdata,
          bias_multiplier_.template data<T>(),
          1,
          dbias_data,
          &context_);
    }
    Xdata += input_offset * group_;
    dYdata += output_offset * group_;
  }
  if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
    // Compute the gradient w.r.t. the input.
    auto* dX = Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
    dX->ResizeLike(X);
    T* dXdata = dX->template mutable_data<T>();
    dYdata = dY.template data<T>();
    for (int image_id = 0; image_id < N; ++image_id) {
      for (int group_id = 0; group_id < group_; ++group_id) {
        // Compute gradient into col_buffer.
        math::Gemm<T, Context>(
            CblasTrans,
            CblasNoTrans,
            kernel_dim,
            output_image_size,
            M / group_,
            1,
            filter_data + group_id * filter_offset,
            dYdata,
            0,
            col_buffer_data,
            &context_);
        if (kernel_.size() == 2) {
          math::Col2Im<T, Context, StorageOrder::NCHW>(
              C / group_,
              input_dims[0],
              input_dims[1],
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
              col_buffer_data,
              dXdata,
              &context_);
        } else {
          math::Col2ImNd<T, Context, StorageOrder::NCHW>(
              kernel_.size(),
              C * input_image_size,
              col_buffer_size,
              img_shape.data(),
              col_buffer_shape.data(),
              kernel_.data(),
              stride_.data(),
              dilation_.data(),
              pads_.data(),
              col_buffer_data,
              dXdata,
              &context_);
        }
        dXdata += input_offset;
        dYdata += output_offset;
      }
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

  const int N = X.dim32(0), H = X.dim32(1), W = X.dim32(2), C = X.dim32(3);
  ConvPoolOpBase<Context>::ComputePads({H, W});
  CAFFE_ENFORCE_EQ(filter.ndim(), 4);
  const int M = filter.dim32(0);
  CAFFE_ENFORCE_EQ(filter.dim32(1), kernel_h());
  CAFFE_ENFORCE_EQ(filter.dim32(2), kernel_w());
  CAFFE_ENFORCE_EQ(filter.dim32(3), C / group_);
  dfilter->ResizeLike(filter);

  // The dimension of each kernel
  const int kernel_dim = kernel_h() * kernel_w() * (C / group_);
  // The offset corresponding to a single input image, and a single output
  // image.
  const int input_offset = H * W * C;
  const int output_offset = dY.size() / dY.dim32(0);
  // The output image size is the spatial size of the output.
  const int output_image_size = dY.dim32(1) * dY.dim32(2);
  // The col buffer is stored in CHW order as well - kernel_dim, and the height
  // and width.
  col_buffer_.Resize(output_image_size, group_ * kernel_dim);

  const T* Xdata = X.template data<T>();
  const T* const filter_data = filter.template data<T>();
  const T* const dYdata = dY.template data<T>();
  T* col_buffer_data = col_buffer_.template mutable_data<T>();
  T* dfilter_data = dfilter->template mutable_data<T>();

  // Pre-setting the gradients to zero.
  math::Set<T, Context>(dfilter->size(), 0, dfilter_data, &context_);

  T* dbias_data = nullptr;
  if (!no_bias_) {
    auto* dbias = Output(BIAS_OR_INPUT_GRAD);
    dbias->Resize(M);
    dbias_data = dbias->template mutable_data<T>();
    math::Set<T, Context>(dbias->size(), 0, dbias_data, &context_);
    if (bias_multiplier_.size() != output_image_size) {
      // If the helper bias multiplier is not M, reshape and fill it with one.
      bias_multiplier_.Resize(vector<TIndex>(1, output_image_size));
      math::Set<T, Context>(
          output_image_size,
          static_cast<T>(1),
          bias_multiplier_.template mutable_data<T>(),
          &context_);
    }
  }

  for (int image_id = 0; image_id < N; ++image_id) {
    // When we compute the gradient with respect to the filters, we need to do
    // im2col to allow gemm-type computation.
    math::Im2Col<T, Context, StorageOrder::NHWC>(
        C,
        H,
        W,
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
        Xdata,
        col_buffer_data,
        &context_,
        group_);
    // Gradient with respect to filter.
    for (int group_id = 0; group_id < group_; ++group_id) {
      math::GemmEx<T, Context>(
          CblasTrans,
          CblasNoTrans,
          M / group_,
          kernel_dim,
          output_image_size,
          1,
          dYdata + output_offset * image_id + group_id * (M / group_),
          M,
          col_buffer_data + group_id * kernel_dim,
          group_ * kernel_dim,
          1,
          dfilter_data + group_id * (M / group_) * kernel_dim,
          kernel_dim,
          &context_);
    }
    if (!no_bias_) {
      // Gradient with respect to bias
      math::Gemv<T, Context>(
          CblasTrans,
          output_image_size,
          M,
          1,
          dYdata + output_offset * image_id,
          bias_multiplier_.template data<T>(),
          1,
          dbias_data,
          &context_);
    }
    Xdata += input_offset;
  }

  if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
    // Compute the gradient w.r.t. the input.
    auto* dX = Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
    dX->ResizeLike(X);
    T* dXdata = dX->template mutable_data<T>();
    for (int image_id = 0; image_id < N; ++image_id) {
      // Compute gradient into col_buffer.
      for (int group_id = 0; group_id < group_; ++group_id) {
        math::GemmEx<T, Context>(
            CblasNoTrans,
            CblasNoTrans,
            output_image_size,
            kernel_dim,
            M / group_,
            1,
            dYdata + output_offset * image_id + group_id * (M / group_),
            M,
            filter_data + group_id * (M / group_) * kernel_dim,
            kernel_dim,
            0,
            col_buffer_data + group_id * kernel_dim,
            group_ * kernel_dim,
            &context_);
      }
      math::Col2Im<T, Context, StorageOrder::NHWC>(
          C,
          H,
          W,
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
          col_buffer_data,
          dXdata,
          &context_,
          group_);
      dXdata += input_offset;
    }
  }
  return true;
}
} // namespace caffe2

#endif // CAFFE2_OPERATORS_CONV_OP_IMPL_H_
