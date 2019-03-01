#ifndef CAFFE_OPERATORS_ENFORCE_FINITE_OP_H_
#define CAFFE_OPERATORS_ENFORCE_FINITE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class EnforceFiniteOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit EnforceFiniteOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType();

 private:
  Tensor buffer_{CPU};

  template <typename T>
  void EnforceOnCPU(const Tensor& input) {
    const T* input_data = input.template data<T>();
    auto size = input.numel();

    for (auto i = 0; i < size; i++) {
      CAFFE_ENFORCE(
          std::isfinite(input_data[i]),
          "Index ",
          i,
          " is not finite (e.g., NaN, Inf): ",
          input_data[i]);
    }
  }
};

} // namespace caffe2

#endif // CAFFE_OPERATORS_ENFORCE_FINITE_OP_H_
