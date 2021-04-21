#include <caffe2/fb/custom_ops/maskrcnn/bbox_transform.h>
#include <caffe2/fb/custom_ops/maskrcnn/box_with_nms_limit.h>
#include <caffe2/fb/custom_ops/maskrcnn/generate_proposals.h>
#include <caffe2/fb/custom_ops/maskrcnn/heatmap_max_keypoint.h>
#include <torch/script.h>

namespace torch {
namespace fb {
namespace metal {

std::tuple<torch::Tensor, torch::Tensor> GenerateProposals(
    const torch::Tensor& scores,
    const torch::Tensor& bbox_deltas,
    const torch::Tensor& im_info_tensor,
    const torch::Tensor& anchors_tensor,
    double spatial_scale_,
    int64_t rpn_pre_nms_topN_,
    int64_t post_nms_topN_,
    double nms_thresh_,
    double rpn_min_size_,
    bool angle_bound_on_,
    int64_t angle_bound_lo_,
    int64_t angle_bound_hi_,
    double clip_angle_thresh_,
    bool legacy_plus_one_,
    c10::optional<std::vector<torch::Tensor>> = {}) {
  // call the cpu ops  
  auto result = caffe2::fb::GenerateProposalsCPUKernel(
      scores.cpu(),
      bbox_deltas.cpu(),
      im_info_tensor.cpu(),
      anchors_tensor,
      spatial_scale_,
      rpn_pre_nms_topN_,
      10,
      nms_thresh_,
      rpn_min_size_,
      angle_bound_on_,
      angle_bound_lo_,
      angle_bound_hi_,
      clip_angle_thresh_,
      legacy_plus_one_,
      {});
  return result;
}

std::tuple<torch::Tensor, torch::Tensor> BBoxTransform(
    const torch::Tensor& roi_in,
    const torch::Tensor& delta_in,
    const torch::Tensor& iminfo_in,
    std::vector<double> weights_,
    bool apply_scale_,
    bool rotated_,
    bool angle_bound_on_,
    int64_t angle_bound_lo_,
    int64_t angle_bound_hi_,
    double clip_angle_thresh_,
    bool legacy_plus_one_,
    c10::optional<std::vector<torch::Tensor>> /* unused */ = {}) {
  return caffe2::fb::BBoxTransformCPUKernel(
      roi_in.cpu(),
      delta_in.cpu(),
      iminfo_in.cpu(),
      weights_,
      apply_scale_,
      rotated_,
      angle_bound_on_,
      angle_bound_lo_,
      angle_bound_lo_,
      clip_angle_thresh_,
      legacy_plus_one_,
      {});
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
BoxWithNMSLimit(
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
    c10::optional<std::vector<torch::Tensor>> = {}) {
  return caffe2::fb::BoxWithNMSLimitCPUKernel(
      tscores.cpu(),
      tboxes.cpu(),
      tbatch_splits.cpu(),
      score_thres_,
      nms_thres_,
      detections_per_im_,
      soft_nms_enabled_,
      soft_nms_method_str_,
      soft_nms_sigma_,
      soft_nms_min_score_thres_,
      rotated_,
      cls_agnostic_bbox_reg_,
      input_boxes_include_bg_cls_,
      output_classes_include_bg_cls_,
      legacy_plus_one_,
      {});
}

torch::Tensor HeatmapMaxKeypoint(
    const torch::Tensor& heatmaps_in_,
    const torch::Tensor& bboxes_in_,
    bool should_output_softmax_,
    c10::optional<std::vector<torch::Tensor>> = {}) {
  return caffe2::fb::HeatmapMaxKeypointCPUKernel(
      heatmaps_in_.cpu(), bboxes_in_.cpu(), should_output_softmax_, {});
}

}
}
}

TORCH_LIBRARY_IMPL(_caffe2, Metal, m) {
  m.impl(
      "_caffe2::GenerateProposals",
      TORCH_FN(torch::fb::metal::GenerateProposals));
  m.impl("_caffe2::BBoxTransform", TORCH_FN(torch::fb::metal::BBoxTransform));
  ;
  m.impl(
      "_caffe2::BoxWithNMSLimit", TORCH_FN(torch::fb::metal::BoxWithNMSLimit));
  m.impl(
      "_caffe2::HeatmapMaxKeypoint",
      TORCH_FN(torch::fb::metal::HeatmapMaxKeypoint));
}
