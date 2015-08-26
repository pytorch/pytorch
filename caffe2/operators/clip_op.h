#ifndef CAFFE2_OPERATORS_CLIP_OP_H_
#define CAFFE2_OPERATORS_CLIP_OP_H_

#include <limits>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "glog/logging.h"

namespace caffe2 {

template <typename dtype, class DeviceContext>
class ClipOp final : public Operator<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  ClipOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<dtype, DeviceContext>(operator_def, ws),
        min_(std::numeric_limits<dtype>::min()),
        max_(std::numeric_limits<dtype>::max()) {
    if (HasArgument("min")) {
      min_ = static_cast<dtype>(
          OperatorBase::GetSingleArgument<float>("min", 0));
    }
    if (HasArgument("max")) {
      max_ = static_cast<dtype>(
          OperatorBase::GetSingleArgument<float>("max", 0));
    }
  }

  bool RunOnDevice();

 protected:
  dtype min_;
  dtype max_;
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(ClipOp);
};

template <typename dtype, class DeviceContext>
class ClipGradientOp final : public Operator<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  ClipGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<dtype, DeviceContext>(operator_def, ws),
        min_(std::numeric_limits<dtype>::min()),
        max_(std::numeric_limits<dtype>::max()) {
    if (HasArgument("min")) {
      min_ = static_cast<dtype>(
          OperatorBase::GetSingleArgument<float>("min", 0));
    }
    if (HasArgument("max")) {
      max_ = static_cast<dtype>(
          OperatorBase::GetSingleArgument<float>("max", 0));
    }
  }

  bool RunOnDevice();

 protected:
  dtype min_;
  dtype max_;
  // Input: X, dY; Output: dX
  INPUT_OUTPUT_STATS(2, 2, 1, 1);
  DISABLE_COPY_AND_ASSIGN(ClipGradientOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_CLIP_OP_H_
