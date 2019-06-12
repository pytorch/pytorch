#ifndef CAFFE2_OPERATORS_BOOLEAN_MASK_OPS_H_
#define CAFFE2_OPERATORS_BOOLEAN_MASK_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/conversions.h"

namespace caffe2 {

template <class Context>
class BooleanMaskOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit BooleanMaskOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override;
};

template <class Context>
class BooleanMaskOpGradient final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  BooleanMaskOpGradient(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  /* Calculating the gradient of the Boolean Mask operator
   * requires access to the original mask that's passed in,
   * and the gradient to backpropagate.
   */
  bool RunOnDevice() override {
    return DispatchHelper<
        TensorTypes<bool, std::int32_t, std::int64_t, float, double>>::
        call(this, Input(1));
  }

  template <typename T>
  bool DoRunWithType();
};

template <class Context>
class SequenceMaskOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  explicit SequenceMaskOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(this->template GetSingleArgument<int>("axis", 1)),
        radius_(this->template GetSingleArgument<int>("radius", 10)),
        grad_(this->template GetSingleArgument<bool>("grad", false)),
        fill_val_(this->template GetSingleArgument<float>(
            "fill_val",
            -1.0f * std::numeric_limits<float>::infinity())) {
    // Mode argument is required
    mode_ = GetArgument(operator_def, "mode").s();
    // batch argument is optional, but if not given, we don't want a default val
    if (HasArgument("batch")) {
      batch_ = GetArgument(operator_def, "batch").i();
    }

    if (HasArgument("repeat_from_axis")) {
      CAFFE_ENFORCE(
          mode_ == "sequence",
          "repeat_from_axis currently only supported in sequence mode.");
      CAFFE_ENFORCE(
          !HasArgument("batch"),
          "repeat_from_axis and batch not currently supported together.");
      repeat_from_ =
          this->template GetSingleArgument<int>("repeat_from_axis", -1);
    }
  }

  bool RunOnDevice() override;

  template <typename T>
  bool DoRunWithType();

 private:
  int axis_;
  int radius_;
  std::string mode_;
  bool grad_;
  float fill_val_;
  int batch_;
  int repeat_from_;
};

} // namespace caffe2

#endif
