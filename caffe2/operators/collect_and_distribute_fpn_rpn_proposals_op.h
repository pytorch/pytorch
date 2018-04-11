#ifndef CAFFE2_OPERATORS_COLLECT_AND_DISTRIBUTE_FPN_RPN_PROPOSALS_OP_H_
#define CAFFE2_OPERATORS_COLLECT_AND_DISTRIBUTE_FPN_RPN_PROPOSALS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace utils {

// Compute the area of an array of boxes.
ERArrXXf BoxesArea(const ERArrXXf& boxes);

// Determine which FPN level each RoI in a set of RoIs should map to based
// on the heuristic in the FPN paper.
ERArrXXf MapRoIsToFpnLevels(Eigen::Ref<const ERArrXXf> rois,
                            const float k_min, const float k_max,
                            const float s0, const float lvl0);

// Sort RoIs from highest to lowest individual RoI score based on
// values from scores array and limit to n results
void SortAndLimitRoIsByScores(Eigen::Ref<const EArrXf> scores, int n,
                              ERArrXXf& rois);

// Updates arr to be indices that would sort the array. Implementation of
// https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
void ArgSort(EArrXi& arr);

// Update out_filtered and out_indices with rows from rois where lvl matches
// value in lvls passed in.
void RowsWhereRoILevelEquals(Eigen::Ref<const ERArrXXf> rois,
                             const ERArrXXf& lvls, const int lvl,
                             ERArrXXf* out_filtered, EArrXi* out_indices);

} // namespace utils

// C++ implementation of CollectAndDistributeFpnRpnProposalsOp
// Merge RPN proposals generated at multiple FPN levels and then
//    distribute those proposals to their appropriate FPN levels for Faster RCNN.
//    An anchor at one FPN level may predict an RoI that will map to another
//    level, hence the need to redistribute the proposals.
// Reference: detectron/lib/ops/collect_and_distribute_fpn_rpn_proposals.py
template <class Context>
class CollectAndDistributeFpnRpnProposalsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CollectAndDistributeFpnRpnProposalsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        roi_canonical_scale_(
            OperatorBase::GetSingleArgument<int>("roi_canonical_scale", 224)),
        roi_canonical_level_(
            OperatorBase::GetSingleArgument<int>("roi_canonical_level", 4)),
        roi_max_level_(
            OperatorBase::GetSingleArgument<int>("roi_max_level", 5)),
        roi_min_level_(
            OperatorBase::GetSingleArgument<int>("roi_min_level", 2)),
        rpn_max_level_(
            OperatorBase::GetSingleArgument<int>("rpn_max_level", 6)),
        rpn_min_level_(
            OperatorBase::GetSingleArgument<int>("rpn_min_level", 2)),
        rpn_post_nms_topN_(
            OperatorBase::GetSingleArgument<int>("post_nms_topN", 2000)) {
    CAFFE_ENFORCE_GE(
        roi_max_level_,
        roi_min_level_,
        "roi_max_level " + caffe2::to_string(roi_max_level_) +
            " must be greater than or equal to roi_min_level " +
            caffe2::to_string(roi_min_level_) + ".");
    CAFFE_ENFORCE_GE(
        rpn_max_level_,
        rpn_min_level_,
        "rpn_max_level " + caffe2::to_string(rpn_max_level_) +
            " must be greater than or equal to rpn_min_level " +
            caffe2::to_string(rpn_min_level_) + ".");
  }

  ~CollectAndDistributeFpnRpnProposalsOp() {}

  bool RunOnDevice() override;

 protected:
  // ROI_CANONICAL_SCALE
  int roi_canonical_scale_{224};
  // ROI_CANONICAL_LEVEL
  int roi_canonical_level_{4};
  // ROI_MAX_LEVEL
  int roi_max_level_{5};
  // ROI_MIN_LEVEL
  int roi_min_level_{2};
  // RPN_MAX_LEVEL
  int rpn_max_level_{6};
  // RPN_MIN_LEVEL
  int rpn_min_level_{2};
  // RPN_POST_NMS_TOP_N
  int rpn_post_nms_topN_{2000};
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_COLLECT_AND_DISTRIBUTE_FPN_RPN_PROPOSALS_OP_H_
