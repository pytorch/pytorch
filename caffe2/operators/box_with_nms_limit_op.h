// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef BOX_WITH_NMS_AND_LIMIT_OP_H_
#define BOX_WITH_NMS_AND_LIMIT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/export_caffe2_op_to_c10.h"
#include "caffe2/core/operator.h"

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(BoxWithNMSLimit)

namespace caffe2 {

// C++ implementation of function insert_box_results_with_nms_and_limit()
template <class Context>
class BoxWithNMSLimitOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit BoxWithNMSLimitOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        score_thres_(
            this->template GetSingleArgument<float>("score_thresh", 0.05)),
        nms_thres_(this->template GetSingleArgument<float>("nms", 0.3)),
        detections_per_im_(
            this->template GetSingleArgument<int>("detections_per_im", 100)),
        soft_nms_enabled_(
            this->template GetSingleArgument<bool>("soft_nms_enabled", false)),
        soft_nms_method_str_(this->template GetSingleArgument<std::string>(
            "soft_nms_method",
            "linear")),
        soft_nms_sigma_(
            this->template GetSingleArgument<float>("soft_nms_sigma", 0.5)),
        soft_nms_min_score_thres_(this->template GetSingleArgument<float>(
            "soft_nms_min_score_thres",
            0.001)),
        rotated_(this->template GetSingleArgument<bool>("rotated", false)),
        cls_agnostic_bbox_reg_(this->template GetSingleArgument<bool>(
            "cls_agnostic_bbox_reg",
            false)),
        input_boxes_include_bg_cls_(this->template GetSingleArgument<bool>(
            "input_boxes_include_bg_cls",
            true)),
        output_classes_include_bg_cls_(this->template GetSingleArgument<bool>(
            "output_classes_include_bg_cls",
            true)),
        legacy_plus_one_(
            this->template GetSingleArgument<bool>("legacy_plus_one", true)) {
    CAFFE_ENFORCE(
        soft_nms_method_str_ == "linear" || soft_nms_method_str_ == "gaussian",
        "Unexpected soft_nms_method");
    soft_nms_method_ = (soft_nms_method_str_ == "linear") ? 1 : 2;

    // When input `boxes` doesn't include background class, the score will skip
    // background class and start with foreground classes directly, and put the
    // background class in the end, i.e. score[:, 0:NUM_CLASSES-1] represents
    // foreground classes and score[:,NUM_CLASSES] represents background class.
    input_scores_fg_cls_starting_id_ = (int)input_boxes_include_bg_cls_;
  }

  ~BoxWithNMSLimitOp() override {}

  bool RunOnDevice() override {
    if (InputSize() > 2) {
      return DispatchHelper<TensorTypes<int, float>>::call(this, Input(2));
    } else {
      return DoRunWithType<float>();
    }
  }

  template <typename T>
  bool DoRunWithType();

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
  // Set for RRPN case to handle rotated boxes. Inputs should be in format
  // [ctr_x, ctr_y, width, height, angle (in degrees)].
  bool rotated_{false};
  // MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
  bool cls_agnostic_bbox_reg_{false};
  // Whether input `boxes` includes background class. If true, boxes will have
  // shape of (N, (num_fg_class+1) * 4or5), otherwise (N, num_fg_class * 4or5)
  bool input_boxes_include_bg_cls_{true};
  // Whether output `classes` includes background class. If true, index 0 will
  // represent background, and valid outputs start from 1.
  bool output_classes_include_bg_cls_{true};
  // The index where foreground starts in scoures. Eg. if 0 represents
  // background class then foreground class starts with 1.
  int input_scores_fg_cls_starting_id_{1};
  // The infamous "+ 1" for box width and height dating back to the DPM days
  bool legacy_plus_one_{true};

  // Map a class id (starting with background and then foreground) from (0, 1,
  // ..., NUM_FG_CLASSES) to it's matching value in box
  inline int get_box_cls_index(int bg_fg_cls_id) {
    if (cls_agnostic_bbox_reg_) {
      return 0;
    } else if (!input_boxes_include_bg_cls_) {
      return bg_fg_cls_id - 1;
    } else {
      return bg_fg_cls_id;
    }
  }

  // Map a class id (starting with background and then foreground) from (0, 1,
  // ..., NUM_FG_CLASSES) to it's matching value in score
  inline int get_score_cls_index(int bg_fg_cls_id) {
    return bg_fg_cls_id - 1 + input_scores_fg_cls_starting_id_;
  }
};

} // namespace caffe2
#endif // BOX_WITH_NMS_AND_LIMIT_OP_H_
