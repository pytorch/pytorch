#ifndef CAFFE2_OPERATORS_DISTANCE_OP_H_
#define CAFFE2_OPERATORS_DISTANCE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class SquaredL2DistanceOp : public Operator<Context> {
 public:
  SquaredL2DistanceOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  // Input: X, Y; Output: Distance
};

template <typename T, class Context>
class SquaredL2DistanceGradientOp final
    : public Operator<Context> {
 public:
  SquaredL2DistanceGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto& Y = Input(1);
    auto& dDistance = Input(2);
    auto* dX = Output(0);
    auto* dY = Output(1);
    int N = X.ndim() > 0 ? X.dim32(0) : 1;
    int D = X.size() / N;
    CAFFE_ENFORCE(X.ndim() == Y.ndim());
    for (int i = 0; i < X.ndim(); ++i) {
      CAFFE_ENFORCE(X.dim32(i) == Y.dim32(i));
    }
    CAFFE_ENFORCE(dDistance.ndim() == 1);
    CAFFE_ENFORCE(dDistance.dim32(0) == N);
    dX->ResizeLike(X);
    dY->ResizeLike(Y);
    math::Sub<T, Context>(
        X.size(), X.template data<T>(), Y.template data<T>(),
        dX->template mutable_data<T>(), &context_);
    for (int i = 0; i < N; ++i) {
      math::Scale<T, Context>(
          D, dDistance.template data<T>() + i, dX->template data<T>() + i * D,
          dX->template mutable_data<T>() + i * D, &context_);
    }
    // The gradient of the other side is basically the negative.
    math::Scale<T, Context>(
        X.size(), -1, dX->template data<T>(),
        dY->template mutable_data<T>(),
        &context_);
    return true;
  }

 protected:
  // Input: X, Y, dDistance; Output: dX, dY
};

template <typename T, class Context>
class DotProductOp : public Operator<Context> {
 public:
  DotProductOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(X_IN, Y_IN);
  OUTPUT_TAGS(DOT_OUT);
};

template <typename T, class Context>
class DotProductGradientOp final : public Operator<Context> {
 public:
  DotProductGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto& X = Input(X_IN);
    auto& Y = Input(Y_IN);
    auto& dDot = Input(DER_DOT_IN);
    auto* dX = Output(DER_X_OUT);
    auto* dY = Output(DER_Y_OUT);
    int N = X.ndim() > 0 ? X.dim32(0) : 1;
    int D = X.size() / N;
    CAFFE_ENFORCE(X.ndim() == Y.ndim());
    for (int i = 0; i < X.ndim(); ++i) {
      CAFFE_ENFORCE(X.dim32(i) == Y.dim32(i));
    }
    CAFFE_ENFORCE(dDot.ndim() == 1);
    CAFFE_ENFORCE(dDot.dim32(0) == N);
    dX->ResizeLike(X);
    dY->ResizeLike(Y);

    const auto* X_data = X.template data<T>();
    const auto* Y_data = Y.template data<T>();
    const auto* dDot_data = dDot.template data<T>();
    auto* dX_data = dX->template mutable_data<T>();
    auto* dY_data = dY->template mutable_data<T>();
    for (int i = 0; i < N; ++i) { // TODO: multithreading
      auto offset = i * D;
      math::Scale<T, Context>(
          D, dDot_data[i], X_data + offset, dY_data + offset, &context_);
      math::Scale<T, Context>(
          D, dDot_data[i], Y_data + offset, dX_data + offset, &context_);
    }

    return true;
  }

 protected:
  INPUT_TAGS(X_IN, Y_IN, DER_DOT_IN);
  OUTPUT_TAGS(DER_X_OUT, DER_Y_OUT);
};

template <typename T, class Context>
class CosineSimilarityOp : public Operator<Context> {
 public:
  CosineSimilarityOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(X_IN, Y_IN);
  OUTPUT_TAGS(COS_OUT);
};

template <typename T, class Context>
class CosineSimilarityGradientOp final : public Operator<Context> {
 public:
  CosineSimilarityGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto& X = Input(X_IN);
    auto& Y = Input(Y_IN);
    auto& dCos = Input(DER_COS_IN);
    auto* dX = Output(DER_X_OUT);
    auto* dY = Output(DER_Y_OUT);
    int N = X.ndim() > 0 ? X.dim32(0) : 1;
    int D = X.size() / N;
    CAFFE_ENFORCE(X.ndim() == Y.ndim());
    for (int i = 0; i < X.ndim(); ++i) {
      CAFFE_ENFORCE(X.dim32(i) == Y.dim32(i));
    }
    CAFFE_ENFORCE(dCos.ndim() == 1);
    CAFFE_ENFORCE(dCos.dim32(0) == N);
    dX->ResizeLike(X);
    dY->ResizeLike(Y);

    const auto* X_data = X.template data<T>();
    const auto* Y_data = Y.template data<T>();
    const auto* dCos_data = dCos.template data<T>();
    auto* dX_data = dX->template mutable_data<T>();
    auto* dY_data = dY->template mutable_data<T>();
    T XN, YN, XY;
    const T kEps = 1e-12;
    for (int i = 0; i < N; ++i) { // TODO: multithreading
      auto offset = i * D;

      // TODO: cache these result from the forward pass
      // ||x||
      math::Dot<T, CPUContext>(
          D, X_data + offset, X_data + offset, &XN, &context_);
      XN = std::sqrt(std::max(XN, kEps));
      // ||y||
      math::Dot<T, CPUContext>(
          D, Y_data + offset, Y_data + offset, &YN, &context_);
      YN = std::sqrt(std::max(YN, kEps));
      // ||x|| * || y ||
      T XYN = XN * YN;
      // x^Ty
      math::Dot<T, CPUContext>(
          D, X_data + offset, Y_data + offset, &XY, &context_);

      math::Scale<T, Context>(
          D, dCos_data[i] / XYN, Y_data + offset, dX_data + offset, &context_);
      math::Axpy(
          D,
          -dCos_data[i] * XY / (XN * XN * XYN),
          X_data + offset,
          dX_data + offset,
          &context_);

      math::Scale<T, Context>(
          D, dCos_data[i] / XYN, X_data + offset, dY_data + offset, &context_);
      math::Axpy(
          D,
          -dCos_data[i] * XY / (YN * YN * XYN),
          Y_data + offset,
          dY_data + offset,
          &context_);
    }

    return true;
  }

 protected:
  INPUT_TAGS(X_IN, Y_IN, DER_COS_IN);
  OUTPUT_TAGS(DER_X_OUT, DER_Y_OUT);
};

}  // namespace caffe2

#endif // CAFFE2_OPERATORS_DISTANCE_OP_H_
