#ifndef CAFFE2_OPERATORS_CROSS_ENTROPY_OP_H_
#define CAFFE2_OPERATORS_CROSS_ENTROPY_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "glog/logging.h"

namespace caffe2 {

template <typename dtype, class DeviceContext>
class LabelCrossEntropyOp final : public Operator<dtype, DeviceContext> {
 public:
  USE_SIMPLE_CTOR_DTOR(LabelCrossEntropyOp);
  USE_OPERATOR_BASE_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  static constexpr dtype kLOG_THRESHOLD() { return 1e-20; }
  // Input: X, label
  // Output: Y
  INPUT_OUTPUT_STATS(2, 2, 1, 1);
  DISABLE_COPY_AND_ASSIGN(LabelCrossEntropyOp);
};

template <typename dtype, class DeviceContext>
class LabelCrossEntropyGradientOp final
    : public Operator<dtype, DeviceContext> {
 public:
  USE_SIMPLE_CTOR_DTOR(LabelCrossEntropyGradientOp);
  USE_OPERATOR_BASE_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  // Input: X, label, dY
  // Ouptut: dX. There is no gradient with respect to the label.
  static constexpr dtype kLOG_THRESHOLD() { return 1e-20; }
  INPUT_OUTPUT_STATS(3, 3, 1, 1);
  DISABLE_COPY_AND_ASSIGN(LabelCrossEntropyGradientOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_CROSS_ENTROPY_OP_H_
