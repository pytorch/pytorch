// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef BOX_WITH_NMS_AND_LIMIT_OP_H_
#define BOX_WITH_NMS_AND_LIMIT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

// C++ implementation of function insert_box_results_with_nms_and_limit()
template <class Context>
class BoxWithNMSLimitOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  BoxWithNMSLimitOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        score_thres_(
            OperatorBase::GetSingleArgument<float>("score_thresh", 0.05)),
        nms_thres_(OperatorBase::GetSingleArgument<float>("nms", 0.3)),
        detections_per_im_(
            OperatorBase::GetSingleArgument<int>("detections_per_im", 100)),
        soft_nms_enabled_(
            OperatorBase::GetSingleArgument<bool>("soft_nms_enabled", false)),
        soft_nms_method_str_(OperatorBase::GetSingleArgument<std::string>(
            "soft_nms_method",
            "linear")),
        soft_nms_sigma_(
            OperatorBase::GetSingleArgument<float>("soft_nms_sigma", 0.5)),
        soft_nms_min_score_thres_(OperatorBase::GetSingleArgument<float>(
            "soft_nms_min_score_thres",
            0.001)) {
    CAFFE_ENFORCE(
        soft_nms_method_str_ == "linear" || soft_nms_method_str_ == "gaussian",
        "Unexpected soft_nms_method");
    soft_nms_method_ = (soft_nms_method_str_ == "linear") ? 1 : 2;
  }

  ~BoxWithNMSLimitOp() {}

  bool RunOnDevice() override;

 protected:
  // TEST.SCORE_THRESH
  float score_thres_ = 0.05;
  // TEST.NMS
  float nms_thres_ = 0.3;
  // TEST.DETECTIONS_PER_IM
  int detections_per_im_ = 100;
  // TEST.SOFT_NMS.ENABLED
  bool soft_nms_enabled_ = false;
  // TEST.SOFT_NMS.METHOD
  std::string soft_nms_method_str_ = "linear";
  unsigned int soft_nms_method_ = 1; // linear
  // TEST.SOFT_NMS.SIGMA
  float soft_nms_sigma_ = 0.5;
  // Lower-bound on updated scores to discard boxes
  float soft_nms_min_score_thres_ = 0.001;
};

} // namespace caffe2
#endif // BOX_WITH_NMS_AND_LIMIT_OP_H_
