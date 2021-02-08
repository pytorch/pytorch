// Copyright 2004-present Facebook. All Rights Reserved.

#include "caffe2/utils/eigen_utils.h"
#include <ATen/native/vulkan/ops/generate_proposals_op_util_nms.h>
#include <torch/csrc/api/include/torch/types.h>

namespace at {
namespace native {

  // Map a class id (starting with background and then foreground) from (0, 1,
  // ..., NUM_FG_CLASSES) to it's matching value in box
  inline int get_box_cls_index(int bg_fg_cls_id, bool cls_agnostic_bbox_reg_, bool input_boxes_include_bg_cls_) {
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
  inline int get_score_cls_index(int bg_fg_cls_id, bool input_boxes_include_bg_cls_) {
    return bg_fg_cls_id - 1 + (input_boxes_include_bg_cls_ ? 1 : 0);
  }

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> BoxWithNMSLimitCPUKernel(
    const torch::Tensor& tscores,
    const torch::Tensor& tboxes,
    const torch::Tensor& tbatch_splits,
    double score_thres_,
    double nms_thres_,
    int64_t detections_per_im_,
    bool soft_nms_enabled_,
    std::string soft_nms_method_str_,
    double soft_nms_sigma_,
    double soft_nms_min_score_thres_,
    bool rotated_,
    bool cls_agnostic_bbox_reg_,
    bool input_boxes_include_bg_cls_,
    bool output_classes_include_bg_cls_,
    bool legacy_plus_one_,
    c10::optional<std::vector<torch::Tensor>> /* unused */ = {}
  );

} // namespace fb
} // namespace caffe2
