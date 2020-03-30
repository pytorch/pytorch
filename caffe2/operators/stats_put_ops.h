#include <limits>
#include "caffe2/core/operator.h"
#include "caffe2/core/stats.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"

namespace caffe2 {

template <typename T>
struct TemplatePutOp : public Operator<CPUContext> {
  explicit TemplatePutOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws),
        given_name_(GetSingleArgument<std::string>(
            "stat_name",
            operator_def.input().Get(0))),
        magnitude_expand_(GetSingleArgument<int64_t>("magnitude_expand", 1)),
        bound_(GetSingleArgument<bool>("bound", false)),
        has_default_(HasSingleArgumentOfType<float>("default_value")),
        default_value_(GetSingleArgument<float>("default_value", 0.0)),
        stat_(given_name_) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<
        int,
        float,
        uint8_t,
        int8_t,
        uint16_t,
        int16_t,
        int64_t,
        at::Half,
        double>>::call(this, Input(0));
  }

  template <typename V>
  bool DoRunWithType() {
    V input = default_value_;

    // If we receive an empty tensor
    if (Input(0).template data<V>()) {
      input = *Input(0).template data<V>();
    } else if (!has_default_) {
      CAFFE_THROW(
          "Default value must be provided when receiving empty tensors for ",
          given_name_);
    }

    int64_t bound_value =
        std::numeric_limits<int64_t>::max() / magnitude_expand_;

    int64_t int_value;
    if (bound_) {
      if (isNan(input)) {
        int_value = 0;
      } else if (input <= -bound_value) {
        int_value = std::numeric_limits<int64_t>::min();
      } else if (input >= bound_value) {
        int_value = std::numeric_limits<int64_t>::max();
      } else {
        int_value = input * magnitude_expand_;
      }
    } else {
      CAFFE_ENFORCE(
          std::abs(static_cast<int64_t>(input)) < bound_value,
          "Input value is too large for the given magnitude expansion!");
      CAFFE_ENFORCE(!isNan(input), "Input value cannot be NaN!");
      int_value = input * magnitude_expand_;
    }

    CAFFE_EVENT(stat_, stat_value, int_value);

    return true;
  }

 private:
  const std::string given_name_;
  const int64_t magnitude_expand_;
  const bool bound_;
  const bool has_default_;
  const float default_value_;
  T stat_;

  template <typename V>
  bool isNan(V input) {
    /*
    Checks if given number of is NaN, while being permissive with different
    implementations of the standard libraries between operating systems.

    Uses the preperties of NaN, defined by IEEE.
    https://www.gnu.org/software/libc/manual/html_node/Infinity-and-NaN.html
    */
    return input != input;
  }
};
} // namespace caffe2
