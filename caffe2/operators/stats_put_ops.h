#include <limits>
#include "caffe2/core/operator.h"
#include "caffe2/core/stats.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"

namespace caffe2 {

template <typename T>
struct TemplatePutOp : public Operator<CPUContext> {
  TemplatePutOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws),
        given_name_(GetSingleArgument<std::string>(
            "stat_name",
            operator_def.input().Get(0))),
        magnitude_expand_(GetSingleArgument<int64_t>("magnitude_expand", 1)),
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
    auto input = *Input(0).template data<V>();

    CAFFE_ENFORCE(
        static_cast<int64_t>(input + 1) <
            std::numeric_limits<int64_t>::max() / magnitude_expand_,
        "Input value is too large for the given magnitude expansion!");

    int64_t int_value = input * magnitude_expand_;

    CAFFE_EVENT(stat_, stat_value, int_value);

    return true;
  }

 private:
  const std::string given_name_;
  const int64_t magnitude_expand_;
  T stat_;
};
} // namespace caffe2
