#ifndef CAFFE2_OPERATORS_ELEMENTWISE_OP_H_
#define CAFFE2_OPERATORS_ELEMENTWISE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "glog/logging.h"

namespace caffe2 {

template <typename dtype, class DeviceContext, class Functor>
class BinaryElementwiseOp : public Operator<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(BinaryElementwiseOp);

  bool RunOnDevice() final {
    auto& input0 = Input(0);
    auto& input1 = Input(1);
    auto* output = Output(0);
    CHECK_EQ(input0.size(), input1.size());
    output->ReshapeLike(input0);
    Functor()(input0.size(), input0.data(), input1.data(),
              output->mutable_data(), &device_context_);
    return true;
  }

  INPUT_OUTPUT_STATS(2, 2, 1, 1);
  DISABLE_COPY_AND_ASSIGN(BinaryElementwiseOp);
};


#define CAFFE2_BINARY_FUNCTOR_WRAPPER(name)                                    \
template <typename dtype, class DeviceContext>                                 \
struct name##Functor {                                                          \
  inline void operator()(const int n, const dtype* x, const dtype* y,          \
                         dtype* output, DeviceContext* device_context) {       \
    math::name<dtype, DeviceContext>(n, x, y, output, device_context);         \
  }                                                                            \
};                                                                             \
template <typename dtype, class DC>                                            \
using name##Op =                                                               \
    BinaryElementwiseOp<dtype, DC, name##Functor<dtype, DC> >


CAFFE2_BINARY_FUNCTOR_WRAPPER(Add);
CAFFE2_BINARY_FUNCTOR_WRAPPER(Sub);
CAFFE2_BINARY_FUNCTOR_WRAPPER(Mul);
CAFFE2_BINARY_FUNCTOR_WRAPPER(Div);
#undef CAFFE2_BINARY_FUNCTOR_WRAPPER

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_ELEMENTWISE_OP_H_
