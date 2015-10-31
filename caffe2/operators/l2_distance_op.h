#ifndef CAFFE2_OPERATORS_L2_DISTANCE_OP_H_
#define CAFFE2_OPERATORS_L2_DISTANCE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class SquaredL2DistanceOp : public Operator<Context> {
 public:
  SquaredL2DistanceOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_BASE_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  // Input: X, Y; Output: Distance
  INPUT_OUTPUT_STATS(2, 2, 1, 1);
  DISABLE_COPY_AND_ASSIGN(SquaredL2DistanceOp);
};

template <typename T, class Context>
class SquaredL2DistanceGradientOp final
    : public Operator<Context> {
 public:
  SquaredL2DistanceGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_BASE_FUNCTIONS;

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto& Y = Input(1);
    auto& dDistance = Input(2);
    auto* dX = Output(0);
    auto* dY = Output(1);
    CAFFE_DCHECK_EQ(X.ndim(), 2);
    int N = X.dim(0);
    int D = X.dim(1);
    CAFFE_DCHECK_EQ(Y.ndim(), 2);
    CAFFE_DCHECK_EQ(Y.dim(0), N);
    CAFFE_DCHECK_EQ(Y.dim(1), D);
    CAFFE_DCHECK_EQ(dDistance.ndim(), 1);
    CAFFE_DCHECK_EQ(dDistance.dim(0), N);
    dX->ReshapeLike(X);
    dY->ReshapeLike(Y);
    math::Sub<T, Context>(
        X.size(), X.template data<T>(), Y.template data<T>(),
        dX->template mutable_data<T>(), &device_context_);
    for (int i = 0; i < N; ++i) {
      math::Scale<T, Context>(
          D, dDistance.template data<T>() + i, dX->template data<T>() + i * D,
          dX->template mutable_data<T>() + i * D, &device_context_);
    }
    // The gradient of the other side is basically the negative.
    math::Scale<T, Context>(
        X.size(), -1, dX->template data<T>(),
        dY->template mutable_data<T>(),
        &device_context_);
    return true;
  }

 protected:
  // Input: X, Y, dDistance; Output: dX, dY
  INPUT_OUTPUT_STATS(3, 3, 2, 2);
  DISABLE_COPY_AND_ASSIGN(SquaredL2DistanceGradientOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_L2_DISTANCE_OP_H_
