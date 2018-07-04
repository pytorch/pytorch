#pragma once
#include "caffe2/utils/math.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

template <typename Context>
class ChannelShuffleOp final : public ConvPoolOpBase<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);
  ChannelShuffleOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws) {}

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
    Y->ResizeLike(X);
    const auto C = X.dim32(1);
    const auto G = this->group_;
    CAFFE_ENFORCE(C % G == 0, "");
    const auto K = C / G;
    const auto S = X.dim32(2) * X.dim32(3);
    for (auto n = 0; n < X.dim32(0); ++n) {
      for (auto g = 0; g < G; ++g) {
        // Scatter the group g block (of size KxS) to output channels
        // g + 0 * G, g + 1 * G, g + 2 * G, g + G * (K - 1) etc.
        math::CopyMatrix<Context>(
            X.itemsize(),
            K,
            S,
            X.template data<float>() + g * K * S + n * C * S,
            S,
            Y->template mutable_data<float>() + g * S + n * C * S,
            G * S,
            &context_,
            X.meta().copy());
      }
    }
    return true;
  }

  bool RunOnDeviceWithOrderNHWC() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
    Y->ResizeLike(X);
    const auto C = X.dim32(3);
    const auto G = this->group_;
    CAFFE_ENFORCE(C % G == 0, "");
    const auto K = C / G;
    std::array<int, 2> dims = {G, K};
    std::array<int, 2> axes = {1, 0};
    for (auto i = 0; i < X.size(); i += C) {
      // Transpose each C = GxK matrix
      math::Transpose(
          2,
          dims.data(),
          axes.data(),
          X.template data<float>() + i,
          Y->template mutable_data<float>() + i,
          &context_);
    }
    return true;
  }
};

template <typename Context>
class ChannelShuffleGradientOp final : public ConvPoolOpBase<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);
  ChannelShuffleGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws) {}

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& dY = Input(0);
    auto* dX = Output(0);
    dX->ResizeLike(dY);
    const auto C = dY.dim32(1);
    const auto G = this->group_;
    CAFFE_ENFORCE(C % G == 0, "");
    const auto K = C / G;
    const auto S = dY.dim32(2) * dY.dim32(3);
    for (auto n = 0; n < dY.dim32(0); ++n) {
      for (auto g = 0; g < G; ++g) {
        // Gather the group g block (of size KxS) from output channels
        // g + 0 * G, g + 1 * G, g + 2 * G, g + G * (K - 1) etc.
        math::CopyMatrix<Context>(
            dY.itemsize(),
            K,
            S,
            dY.template data<float>() + g * S + n * C * S,
            G * S,
            dX->template mutable_data<float>() + g * K * S + n * C * S,
            S,
            &context_,
            dY.meta().copy());
      }
    }
    return true;
  }

  bool RunOnDeviceWithOrderNHWC() override {
    const auto& dY = Input(0);
    auto* dX = Output(0);
    dX->ResizeLike(dY);
    const auto C = dY.dim32(3);
    const auto G = this->group_;
    CAFFE_ENFORCE(C % G == 0, "");
    const auto K = C / G;
    std::array<int, 2> dims = {K, G};
    std::array<int, 2> axes = {1, 0};
    for (auto i = 0; i < dY.size(); i += C) {
      // Transpose each C = KxG matrix
      math::Transpose(
          2,
          dims.data(),
          axes.data(),
          dY.template data<float>() + i,
          dX->template mutable_data<float>() + i,
          &context_);
    }
    return true;
  }
};
} // namespace caffe2
