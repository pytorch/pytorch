#ifndef CAFFE2_OPERATORS_DISTANCE_OP_H_
#define CAFFE2_OPERATORS_DISTANCE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class SquaredL2DistanceOp : public Operator<Context> {
 public:
  template <class... Args>
  explicit SquaredL2DistanceOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  // Input: X, Y; Output: Distance
};

template <typename T, class Context>
class SquaredL2DistanceGradientOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit SquaredL2DistanceGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto& Y = Input(1);
    auto& dDistance = Input(2);

    int N = X.dim() > 0 ? X.dim32(0) : 1;
    int D = N > 0 ? X.numel() / N : 0;
    CAFFE_ENFORCE(X.dim() == Y.dim());
    for (int i = 0; i < X.dim(); ++i) {
      CAFFE_ENFORCE(X.dim32(i) == Y.dim32(i));
    }
    CAFFE_ENFORCE(dDistance.dim() == 1);
    CAFFE_ENFORCE(dDistance.dim32(0) == N);
    auto* dX = Output(0, X.sizes(), at::dtype<T>());
    auto* dY = Output(1, Y.sizes(), at::dtype<T>());
    math::Sub<T, Context>(
        X.numel(),
        X.template data<T>(),
        Y.template data<T>(),
        dX->template mutable_data<T>(),
        &context_);
    for (int i = 0; i < N; ++i) {
      math::Scale<T, T, Context>(
          D,
          dDistance.template data<T>() + i,
          dX->template data<T>() + i * D,
          dX->template mutable_data<T>() + i * D,
          &context_);
    }
    // The gradient of the other side is basically the negative.
    math::Scale<T, T, Context>(
        X.numel(),
        -1,
        dX->template data<T>(),
        dY->template mutable_data<T>(),
        &context_);
    return true;
  }

 protected:
  // Input: X, Y, dDistance; Output: dX, dY
};

template <typename T, class Context>
class L1DistanceOp : public Operator<Context> {
 public:
  template <class... Args>
  explicit L1DistanceOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  // Input: X, Y; Output: Distance
};

template <typename T, class Context>
class L1DistanceGradientOp : public Operator<Context> {
 public:
  template <class... Args>
  explicit L1DistanceGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  // Input: X, Y, dDistance; Output: dX, dY
};

template <typename T, class Context>
class DotProductOp : public Operator<Context> {
 public:
  template <class... Args>
  explicit DotProductOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(X_IN, Y_IN);
  OUTPUT_TAGS(DOT_OUT);
};

template <typename T, class Context>
class DotProductGradientOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit DotProductGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(X_IN, Y_IN, DER_DOT_IN);
  OUTPUT_TAGS(DER_X_OUT, DER_Y_OUT);
};

template <typename T, class Context>
class DotProductWithPaddingOp : public Operator<Context> {
 public:
  template <class... Args>
  explicit DotProductWithPaddingOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        pad_value_(this->template GetSingleArgument<float>("pad_value", 0.0)),
        replicate_(this->template GetSingleArgument<bool>("replicate", false)) {
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  float pad_value_;
  bool replicate_;
  INPUT_TAGS(X_IN, Y_IN);
  OUTPUT_TAGS(DOT_OUT);
};

template <typename T, class Context>
class CosineSimilarityOp : public Operator<Context> {
 public:
  template <class... Args>
  explicit CosineSimilarityOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(X_IN, Y_IN);
  OUTPUT_TAGS(COS_OUT);

 private:
  Tensor aux_;
};

template <typename T, class Context>
class CosineSimilarityGradientOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit CosineSimilarityGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(X_IN, Y_IN, DER_COS_IN);
  OUTPUT_TAGS(DER_X_OUT, DER_Y_OUT);

 private:
  Tensor aux_;
};

template <typename T, class Context>
class DotProductWithPaddingGradientOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit DotProductWithPaddingGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        pad_value_(this->template GetSingleArgument<float>("pad_value", 0.0)),
        replicate_(this->template GetSingleArgument<bool>("replicate", false)) {
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto& X = Input(X_IN);
    auto& Y = Input(Y_IN);
    auto& dDot = Input(DER_DOT_IN);

    int N, D, DX, DY, restD;
    if (X.numel() > 0) {
      N = X.dim() > 0 ? X.dim32(0) : 1;
      DX = X.numel() / N;
      DY = Y.numel() / N;
    } else {
      N = 0;
      DX = 0;
      DY = 0;
    }
    CAFFE_ENFORCE(!replicate_ || DX % DY == 0 || DY % DX == 0);
    D = std::min(DX, DY);
    restD = std::max(DX, DY) - D;
    CAFFE_ENFORCE_EQ(X.dim(), Y.dim());
    CAFFE_ENFORCE_EQ(X.dim32(0), Y.dim32(0));
    CAFFE_ENFORCE_EQ(dDot.dim(), 1);
    CAFFE_ENFORCE_EQ(dDot.dim32(0), N);
    auto* dX = Output(DER_X_OUT, X.sizes(), at::dtype<T>());
    auto* dY = Output(DER_Y_OUT, Y.sizes(), at::dtype<T>());

    const auto* X_data = X.template data<T>();
    const auto* Y_data = Y.template data<T>();
    const auto* dDot_data = dDot.template data<T>();
    auto* dX_data = dX->template mutable_data<T>();
    auto* dY_data = dY->template mutable_data<T>();
    for (int i = 0; i < N; ++i) { // TODO: multithreading
      auto offsetX = i * DX;
      auto offsetY = i * DY;
      if (replicate_) {
        // L_ for longer vector and S_ for shorter vector
        const T *L_data, *S_data;
        T *dL_data, *dS_data;
        int DL, DS;
        if (DX > DY) {
          L_data = X_data + offsetX;
          S_data = Y_data + offsetY;
          dL_data = dX_data + offsetX;
          dS_data = dY_data + offsetY;
          DL = DX;
          DS = DY;
        } else {
          L_data = Y_data + offsetY;
          S_data = X_data + offsetX;
          dL_data = dY_data + offsetY;
          dS_data = dX_data + offsetX;
          DL = DY;
          DS = DX;
        }

        // TODO: get rid of temp memory use
        std::vector<T> tmp_data(DS);
        math::Set<T, Context>(DS, 0.0, dS_data, &context_);
        for (int j = 0; j < DL / DS; j++) {
          math::Scale<T, T, Context>(
              DS, dDot_data[i], S_data, dL_data + j * DS, &context_);
          math::Scale<T, T, Context>(
              DS, dDot_data[i], L_data + j * DS, tmp_data.data(), &context_);
          math::Axpy<float, T, Context>(
              DS, 1.0, tmp_data.data(), dS_data, &context_);
        }
      } else {
        math::Scale<T, T, Context>(
            D, dDot_data[i], X_data + offsetX, dY_data + offsetY, &context_);
        math::Scale<T, T, Context>(
            D, dDot_data[i], Y_data + offsetY, dX_data + offsetX, &context_);
      }

      if (!replicate_ && DX != DY) {
        T* rest_data;
        if (DX > DY) {
          rest_data = dX_data + offsetX + D;
        } else {
          rest_data = dY_data + offsetY + D;
        }
        auto pad_gradient = dDot_data[i] * pad_value_;
        math::Set<T, Context>(restD, pad_gradient, rest_data, &context_);
      }
    }

    return true;
  }

 protected:
  float pad_value_;
  bool replicate_;
  INPUT_TAGS(X_IN, Y_IN, DER_DOT_IN);
  OUTPUT_TAGS(DER_X_OUT, DER_Y_OUT);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_DISTANCE_OP_H_
