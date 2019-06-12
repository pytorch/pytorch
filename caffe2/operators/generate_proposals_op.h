#ifndef CAFFE2_OPERATORS_GENERATE_PROPOSALS_OP_H_
#define CAFFE2_OPERATORS_GENERATE_PROPOSALS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace utils {

// A sub tensor view
template <class T>
class ConstTensorView {
 public:
  ConstTensorView(const T* data, const std::vector<int>& dims)
      : data_(data), dims_(dims) {}

  int ndim() const {
    return dims_.size();
  }
  const std::vector<int>& dims() const {
    return dims_;
  }
  int dim(int i) const {
    DCHECK_LE(i, dims_.size());
    return dims_[i];
  }
  const T* data() const {
    return data_;
  }
  size_t size() const {
    return std::accumulate(
        dims_.begin(), dims_.end(), 1, std::multiplies<size_t>());
  }

 private:
  const T* data_ = nullptr;
  std::vector<int> dims_;
};

// Generate a list of bounding box shapes for each pixel based on predefined
//     bounding box shapes 'anchors'.
// anchors: predefined anchors, size(A, 4)
// Return: all_anchors_vec: (H * W, A * 4)
// Need to reshape to (H * W * A, 4) to match the format in python
CAFFE2_API ERMatXf ComputeAllAnchors(
    const TensorCPU& anchors,
    int height,
    int width,
    float feat_stride);

} // namespace utils

// C++ implementation of GenerateProposalsOp
// Generate bounding box proposals for Faster RCNN. The propoasls are generated
//     for a list of images based on image score 'score', bounding box
//     regression result 'deltas' as well as predefined bounding box shapes
//     'anchors'. Greedy non-maximum suppression is applied to generate the
//     final bounding boxes.
// Reference: facebookresearch/Detectron/detectron/ops/generate_proposals.py
template <class Context>
class GenerateProposalsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  GenerateProposalsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        spatial_scale_(
            this->template GetSingleArgument<float>("spatial_scale", 1.0 / 16)),
        feat_stride_(1.0 / spatial_scale_),
        rpn_pre_nms_topN_(
            this->template GetSingleArgument<int>("pre_nms_topN", 6000)),
        rpn_post_nms_topN_(
            this->template GetSingleArgument<int>("post_nms_topN", 300)),
        rpn_nms_thresh_(
            this->template GetSingleArgument<float>("nms_thresh", 0.7f)),
        rpn_min_size_(this->template GetSingleArgument<float>("min_size", 16)),
        correct_transform_coords_(this->template GetSingleArgument<bool>(
            "correct_transform_coords",
            false)),
        angle_bound_on_(
            this->template GetSingleArgument<bool>("angle_bound_on", true)),
        angle_bound_lo_(
            this->template GetSingleArgument<int>("angle_bound_lo", -90)),
        angle_bound_hi_(
            this->template GetSingleArgument<int>("angle_bound_hi", 90)),
        clip_angle_thresh_(
            this->template GetSingleArgument<float>("clip_angle_thresh", 1.0)) {}

  ~GenerateProposalsOp() {}

  bool RunOnDevice() override;

  // Generate bounding box proposals for a given image
  // im_info: [height, width, im_scale]
  // all_anchors: (H * W * A, 4)
  // bbox_deltas_tensor: (4 * A, H, W)
  // scores_tensor: (A, H, W)
  // out_boxes: (n, 5)
  // out_probs: n
  void ProposalsForOneImage(
      const Eigen::Array3f& im_info,
      const Eigen::Map<const ERMatXf>& all_anchors,
      const utils::ConstTensorView<float>& bbox_deltas_tensor,
      const utils::ConstTensorView<float>& scores_tensor,
      ERArrXXf* out_boxes,
      EArrXf* out_probs) const;

 protected:
  // spatial_scale_ must be declared before feat_stride_
  float spatial_scale_{1.0};
  float feat_stride_{1.0};

  // RPN_PRE_NMS_TOP_N
  int rpn_pre_nms_topN_{6000};
  // RPN_POST_NMS_TOP_N
  int rpn_post_nms_topN_{300};
  // RPN_NMS_THRESH
  float rpn_nms_thresh_{0.7};
  // RPN_MIN_SIZE
  float rpn_min_size_{16};
  // Correct bounding box transform coordates, see bbox_transform() in boxes.py
  // Set to true to match the detectron code, set to false for backward
  // compatibility
  bool correct_transform_coords_{false};
  // If set, for rotated boxes in RRPN, output angles are normalized to be
  // within [angle_bound_lo, angle_bound_hi].
  bool angle_bound_on_{true};
  int angle_bound_lo_{-90};
  int angle_bound_hi_{90};
  // For RRPN, clip almost horizontal boxes within this threshold of
  // tolerance for backward compatibility. Set to negative value for
  // no clipping.
  float clip_angle_thresh_{1.0};
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_GENERATE_PROPOSALS_OP_H_
