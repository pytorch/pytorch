// conv_op_impl.h is the templated implementation of the conv_op.h file.
#ifndef CAFFE2_OPERATORS_CONV_OP_IMPL_H_
#define CAFFE2_OPERATORS_CONV_OP_IMPL_H_

#include "caffe2/core/context.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
bool ConvOp<T, Context>::RunOnDeviceWithOrderNCHW() {
  const Tensor<Context>& X = Input(INPUT);
  auto& filter = Input(FILTER);
  Tensor<Context>* Y = Output(0);
  const int N = X.dim32(0), C = X.dim32(1);
  CAFFE_ENFORCE_EQ(X.ndim(), filter.ndim());
  const int M = filter.dim32(0);
  CAFFE_ENFORCE(
      C == filter.dim32(1) * group_,
      "Convolution op: input channels does not match: # of input channels ",
      C,
      " is not equal to kernel channels * group:",
      filter.dim32(1),
      "*",
      group_);
  CAFFE_ENFORCE(
      M % group_ == 0,
      "The number of output channels is not divisible by group.");

  int kernel_dims_size = 1;
  for (int i = 0; i < kernel_.size(); ++i) {
    CAFFE_ENFORCE(filter.dim32(i + 2) == kernel_[i]);
    kernel_dims_size *= kernel_[i];
  }

  ConvPoolOpBase<Context>::SetOutputSize(X, Y, filter.dim32(0));

  const vector<int> input_dims = GetDims(X);
  const vector<int> output_dims = GetDims(*Y);
  const int input_image_size = this->GetDimsSize(X);
  const int output_image_size = this->GetDimsSize(*Y);

  vector<int> img_shape;
  img_shape.assign(X.dims().begin() + 1, X.dims().end());

  vector<int> buffer_shape;
  buffer_shape.push_back(C / group_ * kernel_dims_size);
  buffer_shape.insert(
      buffer_shape.end(), output_dims.begin(), output_dims.end());

  if (kernel_.size() != 2) {
    SetDeviceTensor(img_shape, &img_shape_device_);
    SetDeviceTensor(buffer_shape, &col_buffer_shape_device_);
  }

  const int col_buffer_size =
      (C / group_) * kernel_dims_size * output_image_size;

  // The dimension of each kernel
  const int kernel_dim = C / group_ * kernel_dims_size;
  // The offset corresponding to a single input image, and a single output
  // image.
  const int input_offset = C / group_ * input_image_size;
  const int output_offset = Y->size() / Y->dim32(0) / group_;
  const int filter_offset = filter.size() / group_;

  // The col buffer is stored in CHW order as well - kernel_dim, and the height
  // and width.
  const T* Xdata = X.template data<T>();
  if (InputSize() == 3) {
    const auto& bias = Input(BIAS);
    CAFFE_ENFORCE(bias.ndim() == 1);
    CAFFE_ENFORCE(bias.dim32(0) == M);
    ConvPoolOpBase<Context>::template SetBiasMultiplier<T>(
        output_image_size, &bias_multiplier_);
  }
  T* Ydata = Y->template mutable_data<T>();

  auto f = [&](Tensor<Context>* col_buffer) {
    col_buffer->Resize(buffer_shape);
    T* col_buffer_data = col_buffer->template mutable_data<T>();
    // Im2col, followed by gemm.
    for (int image_id = 0; image_id < N; ++image_id) {
      for (int group_id = 0; group_id < group_; ++group_id) {
        if (kernel_.size() == 2) {
          math::Im2col<T, Context, StorageOrder::NCHW>(
              Xdata + group_id * input_offset,
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
              &context_);
        } else {
          math::Im2colNd<T, Context, StorageOrder::NCHW>(
              Xdata + group_id * input_offset,
              img_shape_device_.template data<int>(),
              col_buffer_shape_device_.template data<int>(),
              C * input_image_size,
              col_buffer_size,
              kernel_device_.template data<int>(),
              stride_device_.template data<int>(),
              dilation_device_.template data<int>(),
              pads_device_.template data<int>(),
              kernel_.size(),
              col_buffer_data,
              &context_);
        }
        // Weight term
        math::Gemm<T, Context>(
            CblasNoTrans,
            CblasNoTrans,
            M / group_,
            output_image_size,
            kernel_dim,
            1,
            filter.template data<T>() + group_id * filter_offset,
            col_buffer_data,
            0,
            Ydata + group_id * output_offset,
            &context_);
      }
      if (InputSize() == 3) {
        // Bias term can be carried out outside the group definition
        // to be efficient.
        auto* bias_data = Input(BIAS).template data<T>();
        math::Gemm<T, Context>(
            CblasNoTrans,
            CblasNoTrans,
            M,
            output_image_size,
            1,
            1,
            bias_data,
            bias_multiplier_.template data<T>(),
            1,
            Ydata,
            &context_);
      }
      Xdata += input_offset * group_;
      Ydata += output_offset * group_;
    }
  };

  if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
    runWithSharedBuffer<Context>(ws_, f);
  } else {
    f(&col_buffer_);
  }
  return true;
}

// The implementations.
template <typename T, class Context>
bool ConvOp<T, Context>::RunOnDeviceWithOrderNHWC() {
  const Tensor<Context>& X = Input(INPUT);
  auto& filter = Input(FILTER);
  Tensor<Context>* Y = Output(0);
  const int N = X.dim32(0), H = X.dim32(1), W = X.dim32(2), C = X.dim32(3);

  CAFFE_ENFORCE_EQ(
      kernel_.size(),
      2,
      "Only 2d convolution is supported for NHWC storage type");

  CAFFE_ENFORCE(X.ndim(), filter.ndim());
  const int M = filter.dim32(0);
  CAFFE_ENFORCE(filter.dim32(1) == kernel_h());
  CAFFE_ENFORCE(filter.dim32(2) == kernel_w());
  CAFFE_ENFORCE(filter.dim32(3) == C);

  ConvPoolOpBase<Context>::SetOutputSize(X, Y, filter.dim32(0));
  // The dimension of each kernel
  const int kernel_dim = kernel_h() * kernel_w() * C;
  // The offset corresponding to a single input image, and a single output
  // image.
  const int input_offset = H * W * C;
  const int output_offset = Y->size() / Y->dim32(0);
  // The output image size is the spatial size of the output.
  const int output_image_size = Y->dim32(1) * Y->dim32(2);
  // The col buffer is stored in HWC order as well - kernel_dim, and the height
  // and width.
  const T* Xdata = X.template data<T>();
  T* Ydata = Y->template mutable_data<T>();
  // Specialized path for 1 by 1 convolution with stride 1, pad 0 - we
  // can skip im2col.
  if (kernel_dim == C && Y->dim32(1) == X.dim32(1) &&
      Y->dim32(2) == X.dim32(2) && stride_h() == 1 && stride_w() == 1 &&
      pad_t() == 0 && pad_b() == 0 && pad_l() == 0 && pad_r() == 0) {
    math::Gemm<T, Context>(
        CblasNoTrans,
        CblasTrans,
        N * H * W,
        M,
        C,
        1,
        Xdata,
        filter.template data<T>(),
        0,
        Ydata,
        &context_);
    if (InputSize() == 3) {
      auto& bias = Input(BIAS);
      CAFFE_ENFORCE(1 == bias.ndim());
      CAFFE_ENFORCE(bias.dim32(0) == M);
      if (bias_multiplier_.size() != N * H * W) {
        // If the helper bias multiplier is not M, reshape and fill it with one.
        bias_multiplier_.Resize(vector<TIndex>(1, N * H * W));
        math::Set<T, Context>(
            N * H * W,
            static_cast<T>(1),
            bias_multiplier_.template mutable_data<T>(),
            &context_);
      }
      math::Gemm<T, Context>(
          CblasNoTrans,
          CblasNoTrans,
          N * H * W,
          M,
          1,
          1,
          bias_multiplier_.template data<T>(),
          bias.template data<T>(),
          1,
          Ydata,
          &context_);
    }
  } else {
    if (InputSize() == 3) {
      const auto& bias = Input(BIAS);
      CAFFE_ENFORCE(1 == bias.ndim());
      CAFFE_ENFORCE(bias.dim32(0) == M);
      ConvPoolOpBase<Context>::template SetBiasMultiplier<T>(
          output_image_size, &bias_multiplier_);
    }
    auto f = [&](Tensor<Context>* col_buffer) {
      col_buffer->Resize(
          vector<TIndex>{Y->dim32(1), Y->dim32(2), kernel_h(), kernel_w(), C});
      T* col_buffer_data = col_buffer->template mutable_data<T>();
      // Im2col, followed by gemm.
      for (int image_id = 0; image_id < N; ++image_id) {
        math::Im2col<T, Context, StorageOrder::NHWC>(
            Xdata,
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
            &context_);
        // Weight term
        math::Gemm<T, Context>(
            CblasNoTrans,
            CblasTrans,
            output_image_size,
            M,
            kernel_dim,
            1,
            col_buffer_data,
            filter.template data<T>(),
            0,
            Ydata,
            &context_);
        if (InputSize() == 3) {
          // Bias term
          math::Gemm<T, Context>(
              CblasNoTrans,
              CblasNoTrans,
              output_image_size,
              M,
              1,
              1,
              bias_multiplier_.template data<T>(),
              Input(BIAS).template data<T>(),
              1,
              Ydata,
              &context_);
        }
        Xdata += input_offset;
        Ydata += output_offset;
      }
    };
    if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
      runWithSharedBuffer<Context>(ws_, f);
    } else {
      f(&col_buffer_);
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
        math::Im2col<T, Context, StorageOrder::NCHW>(
            Xdata + group_id * input_offset,
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
            &context_);
      } else {
        math::Im2colNd<T, Context, StorageOrder::NCHW>(
            Xdata + group_id * input_offset,
            img_shape_device_.template data<int>(),
            col_buffer_shape_device_.template data<int>(),
            C * input_image_size,
            col_buffer_size,
            kernel_device_.template data<int>(),
            stride_device_.template data<int>(),
            dilation_device_.template data<int>(),
            pads_device_.template data<int>(),
            kernel_.size(),
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
          math::Col2im<T, Context, StorageOrder::NCHW>(
              col_buffer_data,
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
              dXdata,
              &context_);
        } else {
          math::Col2imNd<T, Context, StorageOrder::NCHW>(
              col_buffer_data,
              img_shape_device_.template data<int>(),
              col_buffer_shape_device_.template data<int>(),
              C * input_image_size,
              col_buffer_size,
              kernel_device_.template data<int>(),
              stride_device_.template data<int>(),
              dilation_device_.template data<int>(),
              pads_device_.template data<int>(),
              kernel_.size(),
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
  CAFFE_ENFORCE(4 == filter.ndim());
  const int M = filter.dim32(0);
  CAFFE_ENFORCE(filter.dim32(1) == kernel_h());
  CAFFE_ENFORCE(filter.dim32(2) == kernel_w());
  CAFFE_ENFORCE(filter.dim32(3) == C);
  dfilter->ResizeLike(filter);

  // The dimension of each kernel
  const int kernel_dim = kernel_h() * kernel_w() * C;
  // The offset corresponding to a single input image, and a single output
  // image.
  const int input_offset = H * W * C;
  const int output_offset = dY.size() / dY.dim32(0);
  // The output image size is the spatial size of the output.
  const int output_image_size = dY.dim32(1) * dY.dim32(2);
  // The col buffer is stored in CHW order as well - kernel_dim, and the height
  // and width.
  col_buffer_.Resize(output_image_size, kernel_dim);

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
    math::Im2col<T, Context, StorageOrder::NHWC>(
        Xdata,
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
        &context_);
    // Gradient with respect to filter.
    math::Gemm<T, Context>(
        CblasTrans,
        CblasNoTrans,
        M,
        kernel_dim,
        output_image_size,
        1,
        dYdata + output_offset * image_id,
        col_buffer_data,
        1,
        dfilter_data,
        &context_);
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
      math::Gemm<T, Context>(
          CblasNoTrans,
          CblasNoTrans,
          output_image_size,
          kernel_dim,
          M,
          1,
          dYdata + output_offset * image_id,
          filter_data,
          0,
          col_buffer_data,
          &context_);
      math::Col2im<T, Context, StorageOrder::NHWC>(
          col_buffer_data,
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
          dXdata,
          &context_);
      dXdata += input_offset;
    }
  }
  return true;
}
} // namespace caffe2

#endif // CAFFE2_OPERATORS_CONV_OP_IMPL_H_
