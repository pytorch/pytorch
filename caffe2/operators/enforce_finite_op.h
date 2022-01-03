#ifndef CAFFE_OPERATORS_ENFORCE_FINITE_OP_H_
#define CAFFE_OPERATORS_ENFORCE_FINITE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "c10/util/irange.h"

namespace caffe2 {

namespace detail {
TORCH_API void LogBlobFiniteness(Workspace *ws);
}

template <class Context>
class EnforceFiniteOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit EnforceFiniteOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws), ws_(ws) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType();

 private:
  Workspace* ws_;
  Tensor buffer_{CPU};

  template <typename T>
  void EnforceOnCPU(const Tensor& input) {
    const T* input_data = input.template data<T>();
    auto size = input.numel();

    for (const auto i : c10::irange(size)) {
      auto isfinite = std::isfinite(input_data[i]);
      if (!isfinite) {
        LogBlobFiniteness();
      }
      CAFFE_ENFORCE_FINITE(
        isfinite,
          "Index ",
          i,
          " is not finite (e.g., NaN, Inf): ",
          input_data[i]);
    }
  }

  // LogBlobFiniteness sums every tensor in the workspace and logs whether it's finite or not.
  void LogBlobFiniteness() {
    caffe2::detail::LogBlobFiniteness(ws_);
  }
};

} // namespace caffe2

#endif // CAFFE_OPERATORS_ENFORCE_FINITE_OP_H_
