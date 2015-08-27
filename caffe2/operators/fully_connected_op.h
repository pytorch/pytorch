#ifndef CAFFE2_OPERATORS_FULLY_CONNECTED_OP_H_
#define CAFFE2_OPERATORS_FULLY_CONNECTED_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// This is Caffe's InnerProductOp, with a name that fits its purpose better.
template <typename dtype, class DeviceContext>
class FullyConnectedOp final : public Operator<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  FullyConnectedOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<dtype, DeviceContext>(operator_def, ws),
        kOne(static_cast<dtype>(1), &device_context_),
        kZero(static_cast<dtype>(0), &device_context_) {}
  ~FullyConnectedOp() {}

  bool RunOnDevice() {
    const auto& X = Input(0);
    const auto& W = Input(1);
    const auto& b = Input(2);
    auto* Y = Output(0);
    DCHECK_GE(X.ndim(), 2);
    DCHECK_GE(W.ndim(), 2);
    if (X.ndim() > 2 || W.ndim() > 2) {
      VLOG(1) << "Using legacy support for arbitrary input and weight "
                << "dimensions.";
    }
    DCHECK_EQ(b.ndim(), 1);
    // batch size
    int M = X.dim(0);
    // Feature dimension
    int K = X.size() / X.dim(0);
    // number of outputs.
    int N = W.dim(0);
    DCHECK_EQ(K, W.size() / W.dim(0));
    DCHECK_EQ(N, b.dim(0));
    Y->Reshape(vector<int>{M, N});
    // W * x
    math::Gemm<dtype, DeviceContext>(
        CblasNoTrans, CblasTrans, M, N, K, kOne.data(), X.data(),
        W.data(), kZero.data(), Y->mutable_data(), &device_context_);
    // Add bias term
    if (bias_multiplier_.size() != M) {
      // If the helper bias multiplier is not M, reshape and fill it with one.
      bias_multiplier_.Reshape(std::vector<int>{M});
      math::Set<dtype, DeviceContext>(
          M, static_cast<dtype>(1), bias_multiplier_.mutable_data(),
          &device_context_);
    }
    math::Gemm<dtype, DeviceContext>(
        CblasNoTrans, CblasNoTrans, M, N, 1, kOne.data(),
        bias_multiplier_.data(), b.data(), kOne.data(),
        Y->mutable_data(), &device_context_);
    return true;
  }

 protected:
  Tensor<dtype, DeviceContext> bias_multiplier_;
  Tensor<dtype, DeviceContext> kOne;
  Tensor<dtype, DeviceContext> kZero;
  // We force this Op to have 3 inputs, since that is almost always the case in
  // deep networks.
  INPUT_OUTPUT_STATS(3, 3, 1, 1);
  DISABLE_COPY_AND_ASSIGN(FullyConnectedOp);
};

template <typename dtype, class DeviceContext>
class FullyConnectedGradientOp : public Operator<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  FullyConnectedGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<dtype, DeviceContext>(operator_def, ws),
        kOne(static_cast<dtype>(1), &device_context_),
        kZero(static_cast<dtype>(0), &device_context_) {}
  ~FullyConnectedGradientOp() {}

  bool RunOnDevice() {
    const auto& X = Input(0);
    const auto& W = Input(1);
    const auto& b = Input(2);
    const auto& dY = Input(3);
    auto* dW = Output(0);
    auto* db = Output(1);
    dW->ReshapeLike(W);
    db->ReshapeLike(b);
    DCHECK_GE(X.ndim(), 2);
    DCHECK_GE(W.ndim(), 2);
    DCHECK_EQ(b.ndim(), 1);
    DCHECK_EQ(dY.ndim(), 2);
    // batch size
    int M = X.dim(0);
    // Feature dimension
    int K = X.size() / X.dim(0);
    // number of outputs.
    int N = W.dim(0);
    DCHECK_EQ(K, W.size() / W.dim(0));
    DCHECK_EQ(N, b.dim(0));
    DCHECK_EQ(M, dY.dim(0));
    DCHECK_EQ(N, dY.dim(1));

    // Compute dW
    math::Gemm<dtype, DeviceContext>(
        CblasTrans, CblasNoTrans, N, K, M, kOne.data(), dY.data(),
        X.data(), kZero.data(), dW->mutable_data(), &device_context_);
    if (bias_multiplier_.size() != M) {
      // If the helper bias multiplier is not M, reshape and fill it with one.
      bias_multiplier_.Reshape(std::vector<int>{M});
      math::Set<dtype, DeviceContext>(
          M, static_cast<dtype>(1), bias_multiplier_.mutable_data(),
          &device_context_);
    }
    // Compute dB
    math::Gemv<dtype, DeviceContext>(
        CblasTrans, M, N, kOne.data(), dY.data(),
        bias_multiplier_.data(), kZero.data(), db->mutable_data(),
        &device_context_);
    // Compute dX if necessary.
    if (OutputSize() == 3) {
      auto* dX = Output(2);
      dX->ReshapeLike(X);
      math::Gemm<dtype, DeviceContext>(
          CblasNoTrans, CblasNoTrans, M, K, N, kOne.data(),
          dY.data(), W.data(), kZero.data(), dX->mutable_data(),
          &device_context_);
    }

    return true;
  }

 protected:
  Tensor<dtype, DeviceContext> bias_multiplier_;
  Tensor<dtype, DeviceContext> kOne;
  Tensor<dtype, DeviceContext> kZero;

  // input: X, W, b, dY
  // output: dW, db, and optionally dX.
  INPUT_OUTPUT_STATS(4, 4, 2, 3);
  DISABLE_COPY_AND_ASSIGN(FullyConnectedGradientOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_FULLY_CONNECTED_OP_H_
