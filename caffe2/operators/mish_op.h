#ifndef CAFFE2_OPERATORS_MISH_OP_H_
#define CAFFE2_OPERATORS_MISH_OP_H_

#include "caffe2/operators/elementwise_ops.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
struct MishFunctor {
  template <typename T>
  bool operator()(const int N, const T* X, T* Y, Context* context) const;
};

template <class Context>
class MishGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(MishGradientOp)
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

#endif // CAFFE2_OPERATORS_Mish_OP_H_
