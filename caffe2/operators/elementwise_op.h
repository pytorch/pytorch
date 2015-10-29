#ifndef CAFFE2_OPERATORS_ELEMENTWISE_OP_H_
#define CAFFE2_OPERATORS_ELEMENTWISE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

template <typename T, class Context, class Functor>
class BinaryElementwiseOp : public Operator<Context> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(BinaryElementwiseOp);

  bool RunOnDevice() {
    auto& input0 = Input(0);
    auto& input1 = Input(1);
    auto* output = Output(0);
    CAFFE_CHECK_EQ(input0.size(), input1.size());
    output->ReshapeLike(input0);
    Functor()(input0.size(), input0.template data<T>(),
              input1.template data<T>(),
              output->template mutable_data<T>(), &device_context_);
    return true;
  }

  INPUT_OUTPUT_STATS(2, 2, 1, 1);
  IN_PLACE_ALLOWED({0, 0}, {1, 0});
  DISABLE_COPY_AND_ASSIGN(BinaryElementwiseOp);
};


#define CAFFE2_BINARY_FUNCTOR_WRAPPER(name)                                    \
template <typename T, class Context>                                           \
struct name##Functor {                                                         \
  inline void operator()(const int n, const T* x, const T* y,                  \
                         T* output, Context* device_context) {                 \
    math::name<T, Context>(n, x, y, output, device_context);                   \
  }                                                                            \
};                                                                             \
template <typename T, class DC>                                                \
using name##Op =                                                               \
    BinaryElementwiseOp<T, DC, name##Functor<T, DC> >


CAFFE2_BINARY_FUNCTOR_WRAPPER(Add);
CAFFE2_BINARY_FUNCTOR_WRAPPER(Sub);
CAFFE2_BINARY_FUNCTOR_WRAPPER(Mul);
CAFFE2_BINARY_FUNCTOR_WRAPPER(Div);
#undef CAFFE2_BINARY_FUNCTOR_WRAPPER

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_ELEMENTWISE_OP_H_
