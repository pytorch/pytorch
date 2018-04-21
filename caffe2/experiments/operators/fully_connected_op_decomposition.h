#ifndef CAFFE2_OPERATORS_FULLY_CONNECTED_OP_DECOMPOSITION_H_
#define CAFFE2_OPERATORS_FULLY_CONNECTED_OP_DECOMPOSITION_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
/*
 * Although a FC_decomp is just like 2 small FC,
 * it is better to have it as one op for future analysis.
 * And if we have 2 FC with bias, it is not right.
 * TODO(wyiming): decompose the layer into 2 matrices
 * W(N * K) = U(N * middle) * trans(V(K * middle))
 * */
// This is Caffe's InnerProductOp, with a name that fits its purpose better.
template <typename T, class Context, class Engine=DefaultEngine>
class FullyConnectedOpDecomp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FullyConnectedOpDecomp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  ~FullyConnectedOpDecomp() {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    const auto& U = Input(1);
    const auto& V = Input(2);
    const auto& b = Input(3);
    auto* Y = Output(0);
    //auto* buffer_ptr = Output(1);
    // Size M * middle;
    //auto& multi_buffer_ = *buffer_ptr;
    CAFFE_ENFORCE_GE(X.ndim(), 1);
    CAFFE_ENFORCE_GE(U.ndim(), 2);
    CAFFE_ENFORCE_GE(V.ndim(), 2);
    if (X.ndim() > 2 || U.ndim() > 2 || V.ndim() > 2) {
      VLOG(1) << "Using legacy support for arbitrary input and weight "
                       "dimensions.";
    }
    CAFFE_ENFORCE_EQ(b.ndim(), 1);
    // batch size
    int M = X.ndim() > 1 ? X.dim32(0) : 1;
    // Feature dimension
    int K = X.size() / M;
    // number of outputs.
    int N = U.dim32(0);
    int middle = U.dim32(0);
    CAFFE_ENFORCE_EQ(K, V.dim32(0));
    CAFFE_ENFORCE_EQ(N, b.dim32(0));
    if (X.ndim() > 1) {
      Y->Resize(M, N);
      multi_buffer_.Resize(M, middle);
    } else {
      Y->Resize(N);
      multi_buffer_.Resize(middle);
    }
  // The col buffer is stored in CHW order as well - kernel_dim, and the height
  // and width.
    //  multi_buffer_.Resize(M, middle);
    T* multi_buffer_data = multi_buffer_.template mutable_data<T>();
    //  X * V * tans(U)
    math::Gemm<T, Context, Engine>(
        CblasNoTrans, CblasNoTrans, M, middle, K, 1, X.template data<T>(),
        V.template data<T>(), 0, multi_buffer_data,
        &context_);
    math::Gemm<T, Context, Engine>(
        CblasNoTrans, CblasTrans, M, N, middle, 1, multi_buffer_data,
        U.template data<T>(), 0, Y->template mutable_data<T>(),
        &context_);
    // Add bias term
    if (bias_multiplier_.size() != M) {
      // If the helper bias multiplier is not M, reshape and fill it with one.
      bias_multiplier_.Resize(M);
      math::Set<T, Context>(
          M, static_cast<T>(1), bias_multiplier_.template mutable_data<T>(),
          &context_);
    }
    math::Gemm<T, Context, Engine>(
        CblasNoTrans, CblasNoTrans, M, N, 1, 1,
        bias_multiplier_.template data<T>(), b.template data<T>(), 1,
        Y->template mutable_data<T>(), &context_);
    return true;
  }

 protected:
  Tensor<Context> bias_multiplier_;
  Tensor<Context> multi_buffer_;
};

template <typename T, class Context, class Engine=DefaultEngine>
class FullyConnectedDecompGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FullyConnectedDecompGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  ~FullyConnectedDecompGradientOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    const auto& U = Input(1);
    const auto& V = Input(2);
    const auto& dY = Input(3);
    DCHECK_GE(X.ndim(), 1);
    DCHECK_GE(U.ndim(), 2);
    DCHECK_GE(V.ndim(), 2);
    DCHECK_LE(dY.ndim(), 2);
    // batch size
    int M = X.ndim() > 1 ? X.dim32(0) : 1;
    // Feature dimension
    int K = X.size() / M;
    // number of outputs.
    int N = U.dim32(0);
    int middle = U.dim32(1);
    DCHECK_EQ(K, V.dim32(0));
    if (dY.ndim() > 1) {
      DCHECK_EQ(M, dY.dim32(0));
      DCHECK_EQ(N, dY.dim32(1));
    } else {
      DCHECK_EQ(X.ndim(), 1);
      DCHECK_EQ(N, dY.size());
    }
    auto* dU = Output(0);
    auto* dV = Output(1);
    auto* db = Output(2);
    dU->ResizeLike(U);
    dV->ResizeLike(V);
    db->Resize(N);

    // Compute dU
    // first compute X * V
    du_buffer_.Resize(N, middle);
    T* du_buffer_data = du_buffer_.template mutable_data<T>();
    math::Gemm<T, Context, Engine>(
        CblasNoTrans, CblasNoTrans, M, middle, K, 1,
        X.template data<T>(), V.template data<T>(),
        0, du_buffer_data,
        &context_);
    math::Gemm<T, Context, Engine>(
        CblasTrans, CblasNoTrans, N, middle, M, 1,
        dY.template data<T>(), du_buffer_data,
        0, dU->template mutable_data<T>(),
        &context_);
    // Compute dV
    // first compute dY * U
    dv_buffer_.Resize(M, middle);
    T* dv_buffer_data = dv_buffer_.template mutable_data<T>();
    math::Gemm<T, Context, Engine>(
        CblasNoTrans, CblasNoTrans, M, middle, N, 1,
        dY.template data<T>(), U.template data<T>(),
        0, dv_buffer_data,
        &context_);
    math::Gemm<T, Context, Engine>(
        CblasTrans, CblasNoTrans, K, middle, M, 1,
        dY.template data<T>(), du_buffer_data,
        0, dV->template mutable_data<T>(),
        &context_);
    if (bias_multiplier_.size() != M) {
      // If the helper bias multiplier is not M, reshape and fill it with one.
      bias_multiplier_.Resize(M);
      math::Set<T, Context>(
          M, static_cast<T>(1),
          bias_multiplier_.template mutable_data<T>(),
          &context_);
    }
    // Compute dB
    math::Gemv<T, Context>(
        CblasTrans, M, N, 1, dY.template data<T>(),
        bias_multiplier_.template data<T>(), 0,
        db->template mutable_data<T>(),
        &context_);
    // Compute dX if necessary.
    if (OutputSize() == 4) {
      auto* dX = Output(3);
      dX->ResizeLike(X);
      dx_buffer_.Resize(M, middle);
      T* dx_buffer_data = dx_buffer_.template mutable_data<T>();
      math::Gemm<T, Context, Engine>(
          CblasNoTrans, CblasNoTrans, M, middle, N, 1,
          dY.template data<T>(), U.template data<T>(),
          0, dx_buffer_data,
          &context_);
      math::Gemm<T, Context, Engine>(
          CblasNoTrans, CblasTrans, M, K, middle, 1,
          dx_buffer_data, V.template data<T>(),
          0, dX->template mutable_data<T>(),
          &context_);
    }

    return true;
  }

 protected:
  Tensor<Context> bias_multiplier_;
  Tensor<Context> du_buffer_;
  Tensor<Context> dv_buffer_;
  Tensor<Context> dx_buffer_;
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_FULLY_CONNECTED_OP_H_
