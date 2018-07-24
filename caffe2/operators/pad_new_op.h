#ifndef CAFFE2_OPERATORS_PAD_NEW_OP_H_
#define CAFFE2_OPERATORS_PAD_NEW_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/pad_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class PadOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  PadOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        pads_(OperatorBase::GetRepeatedArgument<int>("pads")),
        mode_(StringToPadMode(
            OperatorBase::GetSingleArgument<string>("mode", "constant"))),
        value_(static_cast<T>(
            OperatorBase::GetSingleArgument<float>("value", 0.0))) {
    for (int i = 0; i < pads_.size(); i++) {
      CAFFE_ENFORCE(pads_[i] >= 0, "pads value must be non-negative");
    }
  }
  ~PadOp() {}

  bool RunOnDevice() override;

  static std::vector<TensorShape> PadTensorInference(
      const OperatorDef& def,
      const vector<TensorShape>& in);

 private:
  vector<int> pads_;
  PadMode mode_;
  T value_;
  // Input: X
  // Output: Y
};

template <typename T, class Context>
class PadGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  PadGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        pads_(OperatorBase::GetRepeatedArgument<int>("pads")),
        mode_(StringToPadMode(
            OperatorBase::GetSingleArgument<string>("mode", "constant"))) {
    for (int i = 0; i < pads_.size(); i++) {
      CAFFE_ENFORCE(pads_[i] >= 0, "pads value must be non-negative");
    }
  }
  ~PadGradientOp() {}

  bool RunOnDevice() override;

 private:
  vector<int> pads_;
  PadMode mode_;
  T value_;
  // Input: dY
  // Output: dX
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_PAD_NEW_OP_H_
