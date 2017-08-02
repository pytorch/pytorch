#ifndef CAFFE2_OPERATORS_STRING_OPS_H_
#define CAFFE2_OPERATORS_STRING_OPS_H_

#include "caffe2/core/operator.h"
#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

/**
 * ForEach is a unary functor that forwards each element of the input array
 * into the elementwise Functor provided, and gathers the results of each
 * call into the resulting array. Use it as an adaptor if you want to create
 * a UnaryElementwiseOp that acts on each element of the tensor per function
 * call -- this is resonable for complex types where vectorization wouldn't
 * be much of a gain, performance-wise.
 */
template <typename Functor>
struct ForEach {
  explicit ForEach(OperatorBase& op) : functor(op) {}

  template <typename In, typename Out, typename Context>
  void operator()(int n, const In* in, Out* out, Context* /*c*/) {
    for (int i = 0; i < n; ++i) {
      out[i] = functor(in[i]);
    }
  }
  Functor functor;
};

template <typename ScalarFunctor, typename TypeMap = FixedType<std::string>>
using StringElementwiseOp = UnaryElementwiseWithArgsOp<
    TensorTypes<std::string>,
    CPUContext,
    ForEach<ScalarFunctor>,
    TypeMap>;

template <class Context>
class StringJoinOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  StringJoinOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        delimiter_(
            OperatorBase::GetSingleArgument<std::string>("delimiter", ",")),
        axis_(OperatorBase::GetSingleArgument<int>("axis", 0)) {
    CAFFE_ENFORCE(axis_ == 0 || axis_ == 1);
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<
        float,
        double,
        int8_t,
        uint8_t,
        int16_t,
        uint16_t,
        int32_t,
        int64_t,
        std::string,
        bool>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType();

 protected:
  std::string delimiter_;
  int axis_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_STRING_OPS_H_
