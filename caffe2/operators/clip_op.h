#ifndef CAFFE2_OPERATORS_CLIP_OP_H_
#define CAFFE2_OPERATORS_CLIP_OP_H_

#include <limits>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

template <typename T, class Context>
class ClipOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ClipOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        min_(std::numeric_limits<T>::min()),
        max_(std::numeric_limits<T>::max()) {
    if (HasArgument("min")) {
      min_ = static_cast<T>(
          OperatorBase::GetSingleArgument<float>("min", 0));
    }
    if (HasArgument("max")) {
      max_ = static_cast<T>(
          OperatorBase::GetSingleArgument<float>("max", 0));
    }
  }

  bool RunOnDevice() override;

 protected:
  T min_;
  T max_;
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(ClipOp);
};

template <typename T, class Context>
class ClipGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ClipGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        min_(std::numeric_limits<T>::min()),
        max_(std::numeric_limits<T>::max()) {
    if (HasArgument("min")) {
      min_ = static_cast<T>(
          OperatorBase::GetSingleArgument<float>("min", 0));
    }
    if (HasArgument("max")) {
      max_ = static_cast<T>(
          OperatorBase::GetSingleArgument<float>("max", 0));
    }
  }

  bool RunOnDevice() override;

 protected:
  T min_;
  T max_;
  // Input: Y, dY; Output: dX
  INPUT_OUTPUT_STATS(2, 2, 1, 1);
  DISABLE_COPY_AND_ASSIGN(ClipGradientOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_CLIP_OP_H_
