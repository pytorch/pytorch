
#ifndef CAFFE2_OPERATORS_RMAC_REGIONS_OP_H
#define CAFFE2_OPERATORS_RMAC_REGIONS_OP_H

#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class RMACRegionsOp final : public Operator<Context> {
 public:
  RMACRegionsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        scales_(OperatorBase::GetSingleArgument<int>("scales", 3)),
        overlap_(OperatorBase::GetSingleArgument<float>("overlap", 0.4f)) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  int scales_;
  float overlap_;
  Tensor<Context> num_rois_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_RMAC_REGIONS_OP_H
