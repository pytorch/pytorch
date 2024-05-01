
#ifndef CAFFE2_OPERATORS_RMAC_REGIONS_OP_H
#define CAFFE2_OPERATORS_RMAC_REGIONS_OP_H

#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class RMACRegionsOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit RMACRegionsOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        scales_(this->template GetSingleArgument<int>("scales", 3)),
        overlap_(this->template GetSingleArgument<float>("overlap", 0.4f)) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  int scales_;
  float overlap_;
  Tensor num_rois_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_RMAC_REGIONS_OP_H
