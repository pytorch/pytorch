#ifndef CAFFE2_OPERATORS_POW_OP_H_
#define CAFFE2_OPERATORS_POW_OP_H_

#include <algorithm>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/elementwise_ops.h"
#include "caffe2/operators/elementwise_ops_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename InputTypes, class Context>
class PowOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  PowOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(bool, "broadcast", legacy_broadcast_, false),
        OP_SINGLE_ARG(int, "axis", axis_, -1),
        OP_SINGLE_ARG(std::string, "axis_str", axis_str_, ""),
        OP_SINGLE_ARG(std::string, "order", order_, "NCHW") {
    if (HasArgument("exponent")) {
      CAFFE_ENFORCE_EQ(
          InputSize(),
          1,
          "If exponent argument exists, there should be only one Input.");
      exponent_ = OperatorBase::GetSingleArgument<float>("exponent", 1.0f);
    } else {
      CAFFE_ENFORCE_EQ(
          InputSize(),
          2,
          "If exponent argument doesn't exist, there should be two Inputs.");
      if (legacy_broadcast_) {
        if (axis_ != -1) {
          // Get axis from an explicit axis argument.
          CAFFE_ENFORCE_EQ(
              axis_str_.size(),
              0,
              "Args axis and axis_str cannot be used simultaneously.");
        } else if (axis_str_.size()) {
          // Get the axis index semantically.
          CAFFE_ENFORCE_EQ(
              axis_str_.size(), 1, "Unsupported axis string", axis_str_);
          const size_t semantic_axis_ = order_.find(axis_str_);
          CAFFE_ENFORCE_NE(
              semantic_axis_,
              string::npos,
              "Unrecognizable axis string ",
              axis_str_,
              " from order string ",
              order_);
          axis_ = semantic_axis_;
        } else {
          CAFFE_ENFORCE(
              axis_ == -1 && axis_str_.empty(),
              "Do not specify axis or axis_str if broadcast is not enabled.");
        }
      }
    }
  }

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    if (InputSize() == 1) {
      const auto& X = Input(0);
      auto* Y = Output(0);
      Y->ResizeLike(X);
      return ComputeUnaryPow<T>(
          X.size(),
          static_cast<T>(exponent_),
          X.template data<T>(),
          Y->template mutable_data<T>());
    }

    const auto& A = Input(0);
    const auto& B = Input(1);
    auto* C = Output(0);
    CAFFE_ENFORCE_NE(
        C, &A, "In-place is not allowed when Pow take two Inputs.");
    CAFFE_ENFORCE_NE(
        C, &B, "In-place is not allowed when Pow take two Inputs.");
    const T* A_data = A.template data<T>();
    const T* B_data = B.template data<T>();
    std::vector<int> A_dims;
    std::vector<int> B_dims;
    if (legacy_broadcast_) {
      C->ResizeLike(A);
      size_t pre, n, post;
      std::tie(pre, n, post) =
          elementwise_ops_utils::ComputeLegacyBroadcastSizes(A, B, axis_);
      A_dims = {
          static_cast<int>(pre), static_cast<int>(n), static_cast<int>(post)};
      B_dims = {static_cast<int>(n), 1};
    } else {
      std::copy(A.dims().cbegin(), A.dims().cend(), std::back_inserter(A_dims));
      std::copy(B.dims().cbegin(), B.dims().cend(), std::back_inserter(B_dims));
      const std::vector<int> C_dims =
          elementwise_ops_utils::ComputeBinaryBroadcastForwardDims(
              A_dims, B_dims);
      C->Resize(C_dims);
    }
    if (B.size() == 1) {
      T exponent;
      context_.template Copy<T, Context, CPUContext>(1, B_data, &exponent);
      return ComputeUnaryPow<T>(
          A.size(), exponent, A_data, C->template mutable_data<T>());
    }
    return ComputeBinaryPow<T>(
        A_dims, B_dims, A_data, B_data, C->template mutable_data<T>());
  }

 private:
  template <typename T>
  bool ComputeUnaryPow(const int N, const T& exponent, const T* X, T* Y) {
    if (exponent == T(0)) {
      math::Set<T, Context>(N, T(1), Y, &context_);
    } else if (exponent == T(1)) {
      if (Y != X) {
        context_.template Copy<T, Context, Context>(N, X, Y);
      }
    } else if (exponent == T(2)) {
      math::Sqr<T, Context>(N, X, Y, &context_);
    } else if (exponent == T(-1)) {
      math::Inv<T, Context>(N, X, Y, &context_);
    } else if (exponent == T(0.5)) {
      math::Sqrt<T, Context>(N, X, Y, &context_);
    } else if (exponent == T(-0.5)) {
      math::Rsqrt<T, Context>(N, X, Y, &context_);
    } else {
      math::Powx<T, Context>(N, X, exponent, Y, &context_);
    }
    return true;
  }

  template <typename T>
  bool ComputeBinaryPow(
      const std::vector<int>& A_dims,
      const std::vector<int>& B_dims,
      const T* A,
      const T* B,
      T* C) {
    math::Pow<T, Context>(
        A_dims.size(),
        A_dims.data(),
        B_dims.size(),
        B_dims.data(),
        A,
        B,
        C,
        &context_);
    return true;
  }

  float exponent_;

  const bool legacy_broadcast_;
  int axis_;
  const std::string axis_str_;
  const std::string order_;
};

template <typename InputTypes, class Context>
class PowGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  PowGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(bool, "broadcast", legacy_broadcast_, false),
        OP_SINGLE_ARG(int, "axis", axis_, -1),
        OP_SINGLE_ARG(std::string, "axis_str", axis_str_, ""),
        OP_SINGLE_ARG(std::string, "order", order_, "NCHW") {
    if (HasArgument("exponent")) {
      exponent_ = OperatorBase::GetSingleArgument<float>("exponent", 1.0f);
      is_unary_ = true;
    } else if (legacy_broadcast_) {
      if (axis_ != -1) {
        // Get axis from an explicit axis argument.
        CAFFE_ENFORCE_EQ(
            axis_str_.size(),
            0,
            "Args axis and axis_str cannot be used simultaneously.");
      } else if (axis_str_.size()) {
        // Get the axis index semantically.
        CAFFE_ENFORCE_EQ(
            axis_str_.size(), 1, "Unsupported axis string", axis_str_);
        const size_t semantic_axis_ = order_.find(axis_str_);
        CAFFE_ENFORCE_NE(
            semantic_axis_,
            string::npos,
            "Unrecognizable axis string ",
            axis_str_,
            " from order string ",
            order_);
        axis_ = semantic_axis_;
      } else {
        CAFFE_ENFORCE(
            axis_ == -1 && axis_str_.empty(),
            "Do not specify axis or axis_str if broadcast is not enabled.");
      }
    }
  }

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(1));
  }

  template <typename T>
  bool DoRunWithType() {
    if (is_unary_) {
      const auto& dY = Input(0);
      auto* dX = Output(0);
      const int N = dY.size();
      const T* dY_data = dY.template data<T>();
      const T* X_data = nullptr;
      const T* Y_data = nullptr;
      if (dX == &dY) {
        LOG(WARNING) << "In-place Pow gradient, possible loss of precision";
        const auto& Y = Input(1);
        Y_data = Y.template data<T>();
      } else {
        const auto& X = Input(1);
        const auto& Y = Input(2);
        X_data = X.template data<T>();
        Y_data = Y.template data<T>();
        dX->ResizeLike(X);
      }
      T* dX_data = dX->template mutable_data<T>();
      return ComputeUnaryPowGradient<T>(
          N, static_cast<T>(exponent_), dY_data, X_data, Y_data, dX_data);
    }

    const auto& dC = Input(0);
    const auto& A = Input(1);
    const auto& B = Input(2);
    const auto& C = Input(3);
    auto* dA = Output(0);
    auto* dB = Output(1);
    dA->ResizeLike(A);
    dB->ResizeLike(B);
    const T* dC_data = dC.template data<T>();
    const T* A_data = A.template data<T>();
    const T* B_data = B.template data<T>();
    const T* C_data = C.template data<T>();
    T* dA_data = dA->template mutable_data<T>();
    T* dB_data = dB->template mutable_data<T>();

    if (B.size() == 1) {
      const int N = A.size();
      T exponent;
      context_.template Copy<T, Context, CPUContext>(1, B_data, &exponent);
      ComputeUnaryPowGradient<T>(N, exponent, dC_data, A_data, C_data, dA_data);
      ComputeSinglePowBGradient<T>(N, dC_data, A_data, C_data, dB_data);
      return true;
    }

    vector<int> A_dims;
    vector<int> B_dims;
    if (legacy_broadcast_) {
      size_t pre, n, post;
      std::tie(pre, n, post) =
          elementwise_ops_utils::ComputeLegacyBroadcastSizes(A, B, axis_);
      A_dims = {
          static_cast<int>(pre), static_cast<int>(n), static_cast<int>(post)};
      B_dims = {static_cast<int>(n), 1};
    } else {
      std::copy(A.dims().cbegin(), A.dims().cend(), std::back_inserter(A_dims));
      std::copy(B.dims().cbegin(), B.dims().cend(), std::back_inserter(B_dims));
    }
    return ComputeBinaryPowGradient<T>(
        A_dims, B_dims, dC_data, A_data, B_data, C_data, dA_data, dB_data);
  }

 private:
  template <typename T>
  bool ComputeUnaryPowGradient(
      const int N,
      const T& exponent,
      const T* dY,
      const T* X,
      const T* Y,
      T* dX);

  template <typename T>
  bool ComputeSinglePowBGradient(
      const int N,
      const T* dC,
      const T* A,
      const T* C,
      T* dB);

  template <typename T>
  bool ComputeBinaryPowGradient(
      const std::vector<int>& A_dims,
      const std::vector<int>& B_dims,
      const T* dC,
      const T* A,
      const T* B,
      const T* C,
      T* dA,
      T* dB);

  float exponent_;
  bool is_unary_ = false;

  Tensor<Context> buff_;
  Tensor<Context> scratch_;

  const bool legacy_broadcast_;
  int axis_;
  const std::string axis_str_;
  const std::string order_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_POW_OP_H_
