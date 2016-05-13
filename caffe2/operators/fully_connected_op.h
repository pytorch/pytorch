#ifndef CAFFE2_OPERATORS_FULLY_CONNECTED_OP_H_
#define CAFFE2_OPERATORS_FULLY_CONNECTED_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// This is Caffe's InnerProductOp, with a name that fits its purpose better.
template <typename T, class Context, class Engine=DefaultEngine>
class FullyConnectedOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FullyConnectedOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  ~FullyConnectedOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    const auto& W = Input(1);
    const auto& b = Input(2);
    auto* Y = Output(0);
    CAFFE_CHECK_GE(X.ndim(), 2);
    CAFFE_CHECK_GE(W.ndim(), 2);
    if (X.ndim() > 2 || W.ndim() > 2) {
      CAFFE_VLOG(1) << "Using legacy support for arbitrary input and weight "
                       "dimensions.";
    }
    CAFFE_CHECK_EQ(b.ndim(), 1);
    // batch size
    int M = X.dim32(0);
    // Feature dimension
    int K = X.size() / X.dim32(0);
    // number of outputs.
    int N = W.dim32(0);
    CAFFE_CHECK_EQ(K, W.size() / W.dim32(0));
    CAFFE_CHECK_EQ(N, b.dim32(0));
    Y->Reshape(vector<TIndex>{M, N});
    // W * x
    math::Gemm<T, Context, Engine>(
        CblasNoTrans, CblasTrans, M, N, K, 1, X.template data<T>(),
        W.template data<T>(), 0, Y->template mutable_data<T>(),
        &context_);
    // Add bias term
    if (bias_multiplier_.size() != M) {
      // If the helper bias multiplier is not M, reshape and fill it with one.
      bias_multiplier_.Reshape(vector<TIndex>{M});
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
  DISABLE_COPY_AND_ASSIGN(FullyConnectedOp);
};

template <typename T, class Context, class Engine=DefaultEngine>
class FullyConnectedGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FullyConnectedGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  ~FullyConnectedGradientOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    const auto& W = Input(1);
    const auto& dY = Input(2);
    CAFFE_DCHECK_GE(X.ndim(), 2);
    CAFFE_DCHECK_GE(W.ndim(), 2);
    CAFFE_DCHECK_EQ(dY.ndim(), 2);
    // batch size
    int M = X.dim32(0);
    // Feature dimension
    int K = X.size() / X.dim32(0);
    // number of outputs.
    int N = W.dim32(0);
    CAFFE_DCHECK_EQ(K, W.size() / W.dim32(0));
    CAFFE_DCHECK_EQ(M, dY.dim32(0));
    CAFFE_DCHECK_EQ(N, dY.dim32(1));
    auto* dW = Output(0);
    auto* db = Output(1);
    dW->ReshapeLike(W);
    db->Reshape(vector<TIndex>{N});

    // Compute dW
    math::Gemm<T, Context, Engine>(
        CblasTrans, CblasNoTrans, N, K, M, 1,
        dY.template data<T>(), X.template data<T>(),
        0, dW->template mutable_data<T>(),
        &context_);
    if (bias_multiplier_.size() != M) {
      // If the helper bias multiplier is not M, reshape and fill it with one.
      bias_multiplier_.Reshape(vector<TIndex>{M});
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
    if (OutputSize() == 3) {
      auto* dX = Output(2);
      dX->ReshapeLike(X);
      math::Gemm<T, Context, Engine>(
          CblasNoTrans, CblasNoTrans, M, K, N, 1,
          dY.template data<T>(), W.template data<T>(),
          0, dX->template mutable_data<T>(),
          &context_);
    }

    return true;
  }

 protected:
  Tensor<Context> bias_multiplier_;

  DISABLE_COPY_AND_ASSIGN(FullyConnectedGradientOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_FULLY_CONNECTED_OP_H_
