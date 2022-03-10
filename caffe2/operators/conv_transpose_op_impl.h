// conv_transpose_op_impl.h is the templated implementation of the
// conv_transpose_op.h file.
#ifndef CAFFE2_OPERATORS_CONV_TRANSPOSE_OP_IMPL_H_
#define CAFFE2_OPERATORS_CONV_TRANSPOSE_OP_IMPL_H_

#include "caffe2/operators/conv_transpose_op.h"

#include <array>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_op_shared.h"
#include "caffe2/operators/conv_transpose_unpool_op_base.h"
#include "caffe2/utils/math.h"

C10_DECLARE_bool(caffe2_force_shared_col_buffer);

namespace caffe2 {

template <typename T, class Context>
bool ConvTransposeOp<T, Context>::RunOnDeviceWithOrderNCHW() {
  const auto& X = Input(INPUT);
  const auto& filter = Input(FILTER);
  CAFFE_ENFORCE_EQ(X.dim(), 4, "Input must be 4D tensor");
  CAFFE_ENFORCE_EQ(filter.dim(), 4, "filter must be 4D tensor");
  const int N = X.dim32(0);
  const int M = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);
  const int G = group_;
  CAFFE_ENFORCE_EQ(M, filter.dim32(0));
  CAFFE_ENFORCE_EQ(
      M % G, 0, "The number of input channels is not divisible by group.");
  const int C = filter.dim32(1) * G;
  CAFFE_ENFORCE_EQ(
      filter.dim32(2),
      kernel_h(),
      "filter height must be equal to kernel height");
  CAFFE_ENFORCE_EQ(
      filter.dim32(3),
      this->kernel_w(),
      "filter width must be equal to kernel width");
  const std::vector<std::int64_t> Y_dims =
      ConvTransposeUnpoolBase<Context>::GetOutputSize(X, C);
  auto* Y = Output(0, Y_dims, at::dtype<T>());
  if (X.numel() == 0) {
    VLOG(2) << "Number of elements is 0 in ConvTrasposeOp";
    return true;
  }

  const int K_HxW = kernel_h() * kernel_w();
  const int kernel_dim = C / G * K_HxW;
  const int X_HxW = H * W;
  const int Y_HxW = Y->dim32(2) * Y->dim32(3);

  const T* X_data = X.template data<T>();
  const T* filter_data = filter.template data<T>();
  const T* bias_data = nullptr;
  if (InputSize() == 3) {
    auto& bias = Input(BIAS);
    CAFFE_ENFORCE_EQ(bias.dim(), 1, "bias must be 1D tensor");
    CAFFE_ENFORCE_EQ(
        bias.dim32(0),
        C,
        "bias dimension must be equal to output channel number");
    bias_data = bias.template data<T>();
  }
  T* Y_data = Y->template mutable_data<T>();

  const std::vector<std::int64_t> buffer_shape = {
      C, kernel_h(), kernel_w(), H, W};

  const auto func = [&](Tensor* col_buffer) {
    ReinitializeTensor(
        col_buffer,
        buffer_shape,
        at::dtype<T>().device(Context::GetDeviceType()));
    T* col_buffer_data = col_buffer->template mutable_data<T>();
    for (const auto image_id : c10::irange(N)) {
      // Weight term
      if (G == 1) {
        math::Gemm<T, Context>(
            CblasTrans,
            CblasNoTrans,
            kernel_dim,
            X_HxW,
            M,
            1.0f,
            filter_data,
            X_data + image_id * M * X_HxW,
            0.0f,
            col_buffer_data,
            &context_);
      } else {
        math::GemmStridedBatched<T, Context>(
            CblasTrans,
            CblasNoTrans,
            G,
            kernel_dim,
            X_HxW,
            M / G,
            1.0f,
            filter_data,
            M / G * kernel_dim,
            X_data + image_id * M * X_HxW,
            M / G * X_HxW,
            0.0f,
            col_buffer_data,
            col_buffer->numel() / G,
            &context_);
      }

      // Col2Im
      math::Col2Im<T, Context, StorageOrder::NCHW>(
          C,
          Y->dim32(2),
          Y->dim32(3),
          kernel_h(),
          kernel_w(),
          1,
          1,
          pad_t(),
          pad_l(),
          pad_b(),
          pad_r(),
          stride_h(),
          stride_w(),
          col_buffer_data,
          Y_data + image_id * C * Y_HxW,
          &context_);

      if (bias_data != nullptr) {
        // Bias term
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
        math::BiasCHW<T, Context>(
            bias_data,
            nullptr,
            C,
            Y_HxW,
            Y_data + image_id * C * Y_HxW,
            &context_);
#endif // !defined(__ARM_NEON__) && !defined(__ARM_NEON)
      }
    }
    if (bias_data != nullptr) {
#if !defined(__ARM_NEON__) && !defined(__ARM_NEON)
      // Bias term
      const std::array<int, 3> Y_dims = {N, C, Y_HxW};
      const std::array<int, 3> b_dims = {1, C, 1};
      math::Add<T, Context>(
          3,
          Y_dims.data(),
          3,
          b_dims.data(),
          Y_data,
          bias_data,
          Y_data,
          &context_);
#endif
    }
  };

  if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
    runWithSharedBuffer<Context>(ws_, func);
  } else {
    func(&col_buffer_);
  }
  return true;
}

template <typename T, class Context>
bool ConvTransposeOp<T, Context>::RunOnDeviceWithOrderNHWC() {
  const auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  CAFFE_ENFORCE_EQ(filter.dim(), 4, "filter must be 4D tensor");
  const int N = X.dim32(0);
  const int H = X.dim32(1);
  const int W = X.dim32(2);
  const int M = X.dim32(3);
  const int G = group_;
  CAFFE_ENFORCE_EQ(
      filter.dim32(0),
      M,
      "filter number must be equal to input channel number");
  CAFFE_ENFORCE_EQ(
      M % G, 0, "The number of input channels is not divisible by group.");
  const int C = filter.dim32(3) * G;
  CAFFE_ENFORCE_EQ(
      filter.dim32(1),
      kernel_h(),
      "filter height must be equal to kernel height");
  CAFFE_ENFORCE_EQ(
      filter.dim32(2),
      kernel_w(),
      "filter width must be equal to kernel width");

  const std::vector<std::int64_t> Y_dims =
      ConvTransposeUnpoolBase<Context>::GetOutputSize(X, C);
  auto* Y = Output(0, Y_dims, at::dtype<T>());
  if (X.numel() == 0) {
    VLOG(2) << "Number of elements is 0 in ConvTrasposeOp";
    return true;
  }

  const int K_HxW = kernel_h() * kernel_w();
  const int kernel_dim = C / G * K_HxW;
  const int X_HxW = H * W;
  const int Y_HxW = Y->dim32(1) * Y->dim32(2);

  const T* X_data = X.template data<T>();
  const T* filter_data = filter.template data<T>();
  const T* bias_data = nullptr;
  if (InputSize() == 3) {
    auto& bias = Input(BIAS);
    CAFFE_ENFORCE_EQ(bias.dim(), 1, "bias must be 1D tensor");
    CAFFE_ENFORCE_EQ(
        bias.dim32(0),
        C,
        "bias dimension must be equal to output channel number");
    bias_data = bias.template data<T>();
  }
  T* Y_data = Y->template mutable_data<T>();

  const std::vector<std::int64_t> buffer_shape = {
      G, H, W, kernel_h(), kernel_w(), C / G};
  const auto func = [&](Tensor* /*col_buffer*/) {
    ReinitializeTensor(
        &col_buffer_,
        buffer_shape,
        at::dtype<T>().device(Context::GetDeviceType()));
    T* col_buffer_data = col_buffer_.template mutable_data<T>();
    for (const auto image_id : c10::irange(N)) {
      // Weight term
      if (G == 1) {
        math::Gemm<T, Context>(
            CblasNoTrans,
            CblasNoTrans,
            X_HxW,
            kernel_dim,
            M,
            1.0f,
            X_data + image_id * M * X_HxW,
            filter_data,
            0.0f,
            col_buffer_data,
            &context_);
      } else {
        for (const auto group_id : c10::irange(G)) {
          math::GemmEx<T, Context>(
              CblasNoTrans,
              CblasNoTrans,
              X_HxW,
              kernel_dim,
              M / G,
              1.0f,
              X_data + image_id * M * X_HxW + group_id * M / G,
              M,
              filter_data + group_id * M / G * kernel_dim,
              kernel_dim,
              0.0f,
              col_buffer_data + group_id * kernel_dim,
              G * kernel_dim,
              &context_);
        }
      }
      // Col2Im
      math::Col2Im<T, Context, StorageOrder::NHWC>(
          C,
          Y->dim32(1),
          Y->dim32(2),
          kernel_h(),
          kernel_w(),
          1,
          1,
          pad_t(),
          pad_l(),
          pad_b(),
          pad_r(),
          stride_h(),
          stride_w(),
          col_buffer_data,
          Y_data + image_id * C * Y_HxW,
          &context_,
          G);
    }
    if (bias_data != nullptr) {
      // Bias term
      const std::array<int, 2> Y_dims = {N * Y_HxW, C};
      const std::array<int, 2> b_dims = {1, C};
      math::Add<T, Context>(
          2,
          Y_dims.data(),
          2,
          b_dims.data(),
          Y_data,
          bias_data,
          Y_data,
          &context_);
    }
  };

  if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
    runWithSharedBuffer<Context>(ws_, func);
  } else {
    func(&col_buffer_);
  }
  return true;
}

template <typename T, class Context>
bool ConvTransposeGradientOp<T, Context>::RunOnDeviceWithOrderNCHW() {
  const auto& X = Input(INPUT);
  const auto& filter = Input(FILTER);
  const auto& dY = Input(OUTPUT_GRAD);
  CAFFE_ENFORCE_EQ(filter.dim(), 4);
  const int N = X.dim32(0);
  const int M = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);
  const int G = group_;

  CAFFE_ENFORCE_EQ(M, filter.dim32(0));
  CAFFE_ENFORCE_EQ(
      M % G, 0, "The number of input channels is not divisible by group.");
  const int C = filter.dim32(1) * G;
  CAFFE_ENFORCE_EQ(C, dY.dim32(1));
  CAFFE_ENFORCE_EQ(
      filter.dim32(2),
      kernel_h(),
      "filter height must be equal to kernel height");
  CAFFE_ENFORCE_EQ(
      filter.dim32(3),
      this->kernel_w(),
      "filter width must be equal to kernel width");

  const int K_HxW = kernel_h() * kernel_w();
  const int kernel_dim = C / G * K_HxW;
  const int X_HxW = H * W;
  const int Y_HxW = dY.dim32(2) * dY.dim32(3);
  auto* dfilter = Output(FILTER_GRAD, filter.sizes(), at::dtype<T>());

  const T* X_data = X.template data<T>();
  const T* filter_data = filter.template data<T>();
  const T* dY_data = dY.template data<T>();
  T* dfilter_data = dfilter->template mutable_data<T>();
  T* dbias_data = nullptr;
  T* dX_data = nullptr;
  if (!no_bias_) {
    auto* dbias = Output(BIAS_OR_INPUT_GRAD, {C}, at::dtype<T>());
    dbias_data = dbias->template mutable_data<T>();
  }
  const bool compute_dX =
      (OutputSize() == 3) || (no_bias_ && (OutputSize() == 2));
  if (compute_dX) {
    auto* dX = Output(
        no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD, X.sizes(), at::dtype<T>());
    dX_data = dX->template mutable_data<T>();
  }
  math::Set<T, Context>(filter.numel(), T(0), dfilter_data, &context_);

  if (X.numel() == 0) {
    VLOG(2) << "Number of elements is 0 in ConvTrasposeOp";
    if (dbias_data != nullptr) {
      math::Set<T, Context>(C, T(0), dbias_data, &context_);
    }
    return true;
  }

  ReinitializeTensor(
      &col_buffer_,
      std::vector<std::int64_t>{C, kernel_h(), kernel_w(), H, W},
      at::dtype<T>().device(Context::GetDeviceType()));
  T* col_buffer_data = col_buffer_.template mutable_data<T>();

  for (const auto image_id : c10::irange(N)) {
    // gradient w.r.t. filters. Im2Col followed by Gemm
    // Im2Col.
    math::Im2Col<T, Context, StorageOrder::NCHW>(
        C,
        dY.dim32(2),
        dY.dim32(3),
        kernel_h(),
        kernel_w(),
        1,
        1,
        pad_t(),
        pad_l(),
        pad_b(),
        pad_r(),
        stride_h(),
        stride_w(),
        dY_data + image_id * C * Y_HxW,
        col_buffer_data,
        &context_);
    // Gemm
    if (G == 1) {
      math::Gemm<T, Context>(
          CblasNoTrans,
          CblasTrans,
          M,
          kernel_dim,
          X_HxW,
          1.0f,
          X_data + image_id * M * X_HxW,
          col_buffer_data,
          1.0f,
          dfilter_data,
          &context_);
    } else {
      math::GemmStridedBatched<T, Context>(
          CblasNoTrans,
          CblasTrans,
          G,
          M / G,
          kernel_dim,
          X_HxW,
          1.0f,
          X_data + image_id * M * X_HxW,
          M / G * X_HxW,
          col_buffer_data,
          col_buffer_.numel() / G,
          1.0f,
          dfilter_data,
          M / G * kernel_dim,
          &context_);
    }

    if (dX_data != nullptr) {
      // Compute gradients w.r.t. the input
      if (G == 1) {
        math::Gemm<T, Context>(
            CblasNoTrans,
            CblasNoTrans,
            M,
            X_HxW,
            kernel_dim,
            1.0f,
            filter_data,
            col_buffer_data,
            0.0f,
            dX_data + image_id * M * X_HxW,
            &context_);
      } else {
        math::GemmStridedBatched<T, Context>(
            CblasNoTrans,
            CblasNoTrans,
            G,
            M / G,
            X_HxW,
            kernel_dim,
            1.0f,
            filter_data,
            M / G * kernel_dim,
            col_buffer_data,
            col_buffer_.numel() / G,
            0.0f,
            dX_data + image_id * M * X_HxW,
            M / G * X_HxW,
            &context_);
      }
    }
  }

  if (dbias_data != nullptr) {
    // gradient w.r.t. bias
    const std::array<int, 3> Y_dims = {N, C, Y_HxW};
    const std::array<int, 3> b_dims = {1, C, 1};
    math::ReduceSum<T, Context>(
        3, Y_dims.data(), b_dims.data(), T(1), dY_data, dbias_data, &context_);
  }

  return true;
}

template <typename T, class Context>
bool ConvTransposeGradientOp<T, Context>::RunOnDeviceWithOrderNHWC() {
  const auto& X = Input(INPUT);
  const auto& filter = Input(FILTER);
  const auto& dY = Input(OUTPUT_GRAD);
  CAFFE_ENFORCE_EQ(filter.dim(), 4);
  const int N = X.dim32(0);
  const int H = X.dim32(1);
  const int W = X.dim32(2);
  const int M = X.dim32(3);
  const int G = group_;
  CAFFE_ENFORCE_EQ(M, filter.dim32(0));
  CAFFE_ENFORCE_EQ(
      M % G, 0, "The number of input channels is not divisible by group.");
  const int C = filter.dim32(3) * G;
  CAFFE_ENFORCE_EQ(C, dY.dim32(3));
  CAFFE_ENFORCE_EQ(
      filter.dim32(1),
      kernel_h(),
      "filter height must be equal to kernel height");
  CAFFE_ENFORCE_EQ(
      filter.dim32(2),
      this->kernel_w(),
      "filter width must be equal to kernel width");
  CAFFE_ENFORCE_EQ(dY.dim32(3), C);

  const int K_HxW = kernel_h() * kernel_w();
  const int kernel_dim = C / G * K_HxW;
  const int X_HxW = H * W;
  const int Y_HxW = dY.dim32(1) * dY.dim32(2);
  auto* dfilter = Output(FILTER_GRAD, filter.sizes(), at::dtype<T>());

  const T* X_data = X.template data<T>();
  const T* filter_data = filter.template data<T>();
  const T* dY_data = dY.template data<T>();
  T* dfilter_data = dfilter->template mutable_data<T>();
  T* dbias_data = nullptr;
  T* dX_data = nullptr;
  if (!no_bias_) {
    auto* dbias = Output(BIAS_OR_INPUT_GRAD, {C}, at::dtype<T>());
    dbias_data = dbias->template mutable_data<T>();
  }
  const bool compute_dX =
      (OutputSize() == 3) || (no_bias_ && (OutputSize() == 2));
  if (compute_dX) {
    auto* dX = Output(
        no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD, X.sizes(), at::dtype<T>());
    dX_data = dX->template mutable_data<T>();
  }
  math::Set<T, Context>(filter.numel(), T(0), dfilter_data, &context_);

  if (X.numel() == 0) {
    VLOG(2) << "Number of elements is 0 in ConvTrasposeOp";
    if (dbias_data != nullptr) {
      math::Set<T, Context>(C, T(0), dbias_data, &context_);
    }
    return true;
  }

  ReinitializeTensor(
      &col_buffer_,
      std::vector<std::int64_t>{C, kernel_h(), kernel_w(), H, W},
      at::dtype<T>().device(Context::GetDeviceType()));
  T* col_buffer_data = col_buffer_.template mutable_data<T>();

  for (const auto image_id : c10::irange(N)) {
    // gradient w.r.t. filters. Im2Col followed by Gemm
    // Im2Col.
    math::Im2Col<T, Context, StorageOrder::NHWC>(
        C,
        dY.dim32(1),
        dY.dim32(2),
        kernel_h(),
        kernel_w(),
        1,
        1,
        pad_t(),
        pad_l(),
        pad_b(),
        pad_r(),
        stride_h(),
        stride_w(),
        dY_data + image_id * C * Y_HxW,
        col_buffer_data,
        &context_,
        G);
    // Gemm
    if (G == 1) {
      math::Gemm<T, Context>(
          CblasTrans,
          CblasNoTrans,
          M,
          kernel_dim,
          X_HxW,
          1.0f,
          X_data + image_id * M * X_HxW,
          col_buffer_data,
          1.0f,
          dfilter_data,
          &context_);
    } else {
      for (const auto group_id : c10::irange(G)) {
        math::GemmEx<T, Context>(
            CblasTrans,
            CblasNoTrans,
            M / G,
            kernel_dim,
            X_HxW,
            1.0f,
            X_data + image_id * M * X_HxW + group_id * M / G,
            M,
            col_buffer_data + group_id * kernel_dim,
            G * kernel_dim,
            1.0f,
            dfilter_data + group_id * M / G * kernel_dim,
            kernel_dim,
            &context_);
      }
    }

    if (dX_data != nullptr) {
      // Compute gradients w.r.t. the input
      if (G == 1) {
        math::Gemm<T, Context>(
            CblasNoTrans,
            CblasTrans,
            X_HxW,
            M,
            kernel_dim,
            1.0f,
            col_buffer_data,
            filter_data,
            0.0f,
            dX_data + image_id * M * X_HxW,
            &context_);
      } else {
        for (const auto group_id : c10::irange(G)) {
          math::GemmEx<T, Context>(
              CblasNoTrans,
              CblasTrans,
              X_HxW,
              M / G,
              kernel_dim,
              1.0f,
              col_buffer_data + group_id * kernel_dim,
              G * kernel_dim,
              filter_data + group_id * M / G * kernel_dim,
              kernel_dim,
              0.0f,
              dX_data + image_id * M * X_HxW + group_id * M / G,
              M,
              &context_);
        }
      }
    }
  }

  if (dbias_data != nullptr) {
    const std::array<int, 2> Y_dims = {N * Y_HxW, C};
    const std::array<int, 2> b_dims = {1, C};
    math::ReduceSum<T, Context>(
        2, Y_dims.data(), b_dims.data(), T(1), dY_data, dbias_data, &context_);
  }

  return true;
}

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CONV_TRANSPOSE_OP_IMPL_H_
