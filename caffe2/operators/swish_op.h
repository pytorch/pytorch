#ifndef CAFFE2_OPERATORS_SWISH_OP_H_
#define CAFFE2_OPERATORS_SWISH_OP_H_

#include "caffe2/operators/elementwise_ops.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
struct SwishFunctor {
  template <typename T>
  bool operator()(const int N, const T* X, T* Y, Context* context) const;
};

template <class Context>
class SwishGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(SwishGradientOp)
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <typename T>
  bool DoRunWithType();

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(X));
  }

 protected:
  INPUT_TAGS(X, Y, DY);
  OUTPUT_TAGS(DX);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SWISH_OP_H_
