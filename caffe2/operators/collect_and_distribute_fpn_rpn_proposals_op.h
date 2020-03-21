#ifndef CAFFE2_OPERATORS_COLLECT_AND_DISTRIBUTE_FPN_RPN_PROPOSALS_OP_H_
#define CAFFE2_OPERATORS_COLLECT_AND_DISTRIBUTE_FPN_RPN_PROPOSALS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/export_caffe2_op_to_c10.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(CollectAndDistributeFpnRpnProposals);
C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(CollectRpnProposals);
C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(DistributeFpnProposals);

namespace caffe2 {

namespace utils {

// Compute the area of an array of boxes.
ERArrXXf BoxesArea(const ERArrXXf& boxes, const bool legacy_plus_one = false);

// Determine which FPN level each RoI in a set of RoIs should map to based
// on the heuristic in the FPN paper.
ERArrXXf MapRoIsToFpnLevels(
    Eigen::Ref<const ERArrXXf> rois,
    const float k_min,
    const float k_max,
    const float s0,
    const float lvl0,
    const bool legacy_plus_one = false);

// Sort RoIs from highest to lowest individual RoI score based on
// values from scores array and limit to n results
void SortAndLimitRoIsByScores(
    Eigen::Ref<const EArrXf> scores,
    int n,
    ERArrXXf& rois);

// Updates arr to be indices that would sort the array. Implementation of
// https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
void ArgSort(EArrXi& arr);

// Update out_filtered and out_indices with rows from rois where lvl matches
// value in lvls passed in.
void RowsWhereRoILevelEquals(
    Eigen::Ref<const ERArrXXf> rois,
    const ERArrXXf& lvls,
    const int lvl,
    ERArrXXf* out_filtered,
    EArrXi* out_indices);

} // namespace utils

// C++ implementation of CollectAndDistributeFpnRpnProposalsOp
// Merge RPN proposals generated at multiple FPN levels and then
//    distribute those proposals to their appropriate FPN levels for Faster
//    RCNN. An anchor at one FPN level may predict an RoI that will map to
//    another level, hence the need to redistribute the proposals.
// Reference:
// facebookresearch/Detectron/detectron/ops/collect_and_distribute_fpn_rpn_proposals.py
template <class Context>
class CollectAndDistributeFpnRpnProposalsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit CollectAndDistributeFpnRpnProposalsOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        roi_canonical_scale_(
            this->template GetSingleArgument<int>("roi_canonical_scale", 224)),
        roi_canonical_level_(
            this->template GetSingleArgument<int>("roi_canonical_level", 4)),
        roi_max_level_(
            this->template GetSingleArgument<int>("roi_max_level", 5)),
        roi_min_level_(
            this->template GetSingleArgument<int>("roi_min_level", 2)),
        rpn_max_level_(
            this->template GetSingleArgument<int>("rpn_max_level", 6)),
        rpn_min_level_(
            this->template GetSingleArgument<int>("rpn_min_level", 2)),
        rpn_post_nms_topN_(
            this->template GetSingleArgument<int>("rpn_post_nms_topN", 2000)),
        legacy_plus_one_(
            this->template GetSingleArgument<bool>("legacy_plus_one", true)) {
    CAFFE_ENFORCE_GE(
        roi_max_level_,
        roi_min_level_,
        "roi_max_level " + c10::to_string(roi_max_level_) +
            " must be greater than or equal to roi_min_level " +
            c10::to_string(roi_min_level_) + ".");
    CAFFE_ENFORCE_GE(
        rpn_max_level_,
        rpn_min_level_,
        "rpn_max_level " + c10::to_string(rpn_max_level_) +
            " must be greater than or equal to rpn_min_level " +
            c10::to_string(rpn_min_level_) + ".");
  }

  ~CollectAndDistributeFpnRpnProposalsOp() override {}

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
  // The infamous "+ 1" for box width and height dating back to the DPM days
  bool legacy_plus_one_{true};
};

template <class Context>
class CollectRpnProposalsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit CollectRpnProposalsOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        rpn_max_level_(
            this->template GetSingleArgument<int>("rpn_max_level", 6)),
        rpn_min_level_(
            this->template GetSingleArgument<int>("rpn_min_level", 2)),
        rpn_post_nms_topN_(
            this->template GetSingleArgument<int>("rpn_post_nms_topN", 2000)) {
    CAFFE_ENFORCE_GE(
        rpn_max_level_,
        rpn_min_level_,
        "rpn_max_level " + c10::to_string(rpn_max_level_) +
            " must be greater than or equal to rpn_min_level " +
            c10::to_string(rpn_min_level_) + ".");
  }

  ~CollectRpnProposalsOp() override {}

  bool RunOnDevice() override;

 protected:
  // RPN_MAX_LEVEL
  int rpn_max_level_{6};
  // RPN_MIN_LEVEL
  int rpn_min_level_{2};
  // RPN_POST_NMS_TOP_N
  int rpn_post_nms_topN_{2000};
};

template <class Context>
class DistributeFpnProposalsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit DistributeFpnProposalsOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        roi_canonical_scale_(
            this->template GetSingleArgument<int>("roi_canonical_scale", 224)),
        roi_canonical_level_(
            this->template GetSingleArgument<int>("roi_canonical_level", 4)),
        roi_max_level_(
            this->template GetSingleArgument<int>("roi_max_level", 5)),
        roi_min_level_(
            this->template GetSingleArgument<int>("roi_min_level", 2)),
        legacy_plus_one_(
            this->template GetSingleArgument<bool>("legacy_plus_one", true)) {
    CAFFE_ENFORCE_GE(
        roi_max_level_,
        roi_min_level_,
        "roi_max_level " + c10::to_string(roi_max_level_) +
            " must be greater than or equal to roi_min_level " +
            c10::to_string(roi_min_level_) + ".");
  }

  ~DistributeFpnProposalsOp() override {}

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
  // The infamous "+ 1" for box width and height dating back to the DPM days
  bool legacy_plus_one_{true};
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_COLLECT_AND_DISTRIBUTE_FPN_RPN_PROPOSALS_OP_H_
