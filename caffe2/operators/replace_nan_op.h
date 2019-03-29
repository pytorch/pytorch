#ifndef CAFFE_OPERATORS_REPLACE_NAN_OP_H_
#define CAFFE_OPERATORS_REPLACE_NAN_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class ReplaceNaNOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit ReplaceNaNOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
  }

  template <typename T>
  void ReplaceNaN(const T& value, const int64_t size, const T* X, T* Y);

  template <typename T>
  bool DoRunWithType() {
    T value = this->template GetSingleArgument<T>("value", 0);

    auto& input = Input(0);

    auto* output = Output(0, input.sizes(), at::dtype<T>());

    const T* input_data = input.template data<T>();
    T* output_data = output->template mutable_data<T>();

    ReplaceNaN<T>(value, input.numel(), input_data, output_data);

    return true;
  }
};

} // namespace caffe2

#endif // CAFFE_OPERATORS_REPLACE_NAN_OP_H_
