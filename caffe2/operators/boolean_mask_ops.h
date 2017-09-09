#ifndef BOOLEAN_MASK_OPS_H
#define BOOLEAN_MASK_OPS_H

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/conversions.h"

namespace caffe2 {

template <class Context>
class BooleanMaskOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  BooleanMaskOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override;
};

template <class Context>
class SequenceMaskOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  explicit SequenceMaskOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(OperatorBase::GetSingleArgument<int>("axis", 1)),
        radius_(OperatorBase::GetSingleArgument<int>("radius", 10)),
        grad_(OperatorBase::GetSingleArgument<bool>("grad", false)),
        fill_val_(OperatorBase::GetSingleArgument<float>(
            "fill_val",
            -1.0f * std::numeric_limits<float>::infinity())) {
    // Mode argument is required
    mode_ = GetArgument(operator_def, "mode").s();
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
};

} // caffe2

#endif
