#ifndef CAFFE2_OPERATORS_CLIP_OP_H_
#define CAFFE2_OPERATORS_CLIP_OP_H_

#include <limits>

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class ClipOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit ClipOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        min_(std::numeric_limits<T>::lowest()),
        max_(std::numeric_limits<T>::max()) {
    if (HasArgument("min")) {
      min_ = static_cast<T>(this->template GetSingleArgument<float>("min", 0));
    }
    if (HasArgument("max")) {
      max_ = static_cast<T>(this->template GetSingleArgument<float>("max", 0));
    }
  }

  bool RunOnDevice() override;

 protected:
  T min_;
  T max_;
};

template <typename T, class Context>
class ClipGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit ClipGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        min_(std::numeric_limits<T>::lowest()),
        max_(std::numeric_limits<T>::max()) {
    if (HasArgument("min")) {
      min_ = static_cast<T>(this->template GetSingleArgument<float>("min", 0));
    }
    if (HasArgument("max")) {
      max_ = static_cast<T>(this->template GetSingleArgument<float>("max", 0));
    }
  }

  bool RunOnDevice() override;

 protected:
  T min_;
  T max_;
  // Input: Y, dY; Output: dX
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CLIP_OP_H_
