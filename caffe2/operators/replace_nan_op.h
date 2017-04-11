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
  ReplaceNaNOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    T value = OperatorBase::GetSingleArgument<T>("value", 0);

    auto& input = Input(0);
    auto* output = Output(0);
    output->ResizeLike(input);

    const T* input_data = input.template data<T>();
    T* output_data = output->template mutable_data<T>();
    for (TIndex i = 0; i < input.size(); i++) {
      if (std::isnan(input_data[i])) {
        output_data[i] = value;
      } else {
        output_data[i] = input_data[i];
      }
    }

    return true;
  }
};

} // namespace caffe2

#endif // CAFFE_OPERATORS_REPLACE_NAN_OP_H_
