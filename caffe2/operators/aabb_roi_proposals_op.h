#ifndef CAFFE2_OPERATORS_AABB_ROI_PROPOSALS_OP_H_
#define CAFFE2_OPERATORS_AABB_ROI_PROPOSALS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class AABBRoIProposalsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  AABBRoIProposalsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<std::string>("order", "NCHW"))),
        roi_stride_(
            this->template GetSingleArgument<float>("roi_stride", 1.0f)),
        max_pre_nms_proposals_(this->template GetSingleArgument<int>(
            "max_pre_nms_proposals",
            6000)),
        max_post_nms_proposals_(this->template GetSingleArgument<int>(
            "max_post_nms_proposals",
            300)),
        max_iou_(this->template GetSingleArgument<float>("max_iou", 0.7f)),
        min_size_(this->template GetSingleArgument<float>("min_size", 16)) {}

  ~AABBRoIProposalsOp() {}

  bool RunOnDevice() override;

 protected:
  StorageOrder order_{StorageOrder::NCHW};
  float roi_stride_{1.0};

  int max_pre_nms_proposals_{6000};
  int max_post_nms_proposals_{300};
  float max_iou_{0.7};
  float min_size_{16};
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_AABB_ROI_PROPOSALS_OP_H_
