#include "caffe2/operators/aabb_roi_proposals_op.h"

namespace caffe2 {

namespace {

struct AABB {
  inline AABB(float x1, float y1, float x2, float y2)
      : x1(x1), y1(y1), x2(x2), y2(y2) {}

  inline float center_x() const {
    return 0.5f * (x1 + x2);
  }

  inline float center_y() const {
    return 0.5f * (y1 + y2);
  }

  inline float height() const {
    return y2 - y1;
  }

  inline float width() const {
    return x2 - x1;
  }

  inline float area() const {
    return height() * width();
  }

  inline AABB apply_deltas(float dx, float dy, float dw, float dh) const {
    const float width = this->width();
    const float height = this->height();

    const float new_center_x = center_x() + dx * width;
    const float new_center_y = center_y() + dy * height;
    const float new_width = expf(dw) * width;
    const float new_height = expf(dh) * height;

    return AABB(
        new_center_x - 0.5f * new_width,
        new_center_y - 0.5f * new_height,
        new_center_x + 0.5f * new_width,
        new_center_y + 0.5f * new_height);
  }

  inline AABB intersect(const AABB& other) const {
    const float intersection_x1 = std::max(x1, other.x1);
    const float intersection_y1 = std::max(y1, other.y1);
    const float intersection_x2 = std::min(x2, other.x2);
    const float intersection_y2 = std::min(y2, other.y2);
    return AABB(
        intersection_x1,
        intersection_y1,
        std::max(intersection_x1, intersection_x2),
        std::max(intersection_y1, intersection_y2));
  }

  inline AABB clip(float max_x, float max_y) const {
    return AABB(
        std::min(std::max(x1, 0.0f), max_x),
        std::min(std::max(y1, 0.0f), max_y),
        std::min(std::max(x2, 0.0f), max_x),
        std::min(std::max(y2, 0.0f), max_y));
  }

  float x1;
  float y1;
  float x2;
  float y2;
};

} // namespace

template <>
bool AABBRoIProposalsOp<CPUContext>::RunOnDevice() {
  /* Validate inputs */
  CAFFE_ENFORCE(order_ == StorageOrder::NCHW || order_ == StorageOrder::NHWC);

  const Tensor& input_scores_tensor = Input(0);
  CAFFE_ENFORCE_EQ(input_scores_tensor.dim(), 4, input_scores_tensor.dim());
  CAFFE_ENFORCE(input_scores_tensor.template IsType<float>());
  const auto num_images = input_scores_tensor.size(0);
  const auto num_anchors = order_ == StorageOrder::NCHW
      ? input_scores_tensor.size(1)
      : input_scores_tensor.size(3);
  const auto features_height = order_ == StorageOrder::NCHW
      ? input_scores_tensor.size(2)
      : input_scores_tensor.size(1);
  const auto features_width = order_ == StorageOrder::NCHW
      ? input_scores_tensor.size(3)
      : input_scores_tensor.size(2);
  const auto features_pixels = features_height * features_width;
  const float* input_scores_ptr = input_scores_tensor.data<float>();

  const Tensor& input_deltas_tensor = Input(1);
  if (order_ == StorageOrder::NCHW) {
    CAFFE_ENFORCE_EQ(
        input_deltas_tensor.sizes(),
        (vector<int64_t>{
            num_images, num_anchors * 4, features_height, features_width}));
  } else {
    CAFFE_ENFORCE_EQ(order_, StorageOrder::NHWC);
    CAFFE_ENFORCE_EQ(
        input_deltas_tensor.sizes(),
        (vector<int64_t>{
            num_images, features_height, features_width, num_anchors * 4}));
  }
  const float* input_deltas_ptr = input_deltas_tensor.data<float>();

  const Tensor& anchors_tensor = Input(2);
  CAFFE_ENFORCE_EQ(anchors_tensor.sizes(), (vector<int64_t>{num_anchors, 4}));
  CAFFE_ENFORCE(anchors_tensor.template IsType<float>());
  const float* anchors_ptr = anchors_tensor.data<float>();

  const Tensor& image_info_tensor = Input(3);
  CAFFE_ENFORCE_EQ(image_info_tensor.sizes(), (vector<int64_t>{num_images, 2}));
  CAFFE_ENFORCE(image_info_tensor.template IsType<float>());
  const float* image_info_ptr = image_info_tensor.data<float>();

  /* Setup outputs */
  int64_t max_proposals_per_image = max_post_nms_proposals_;
  if (max_post_nms_proposals_ <= 0) {
    max_proposals_per_image = max_pre_nms_proposals_;
    if (max_proposals_per_image <= 0) {
      max_proposals_per_image = features_pixels * num_anchors;
    }
  }
  const int64_t max_total_proposals = max_proposals_per_image * num_images;

  Tensor* output_batch_splits_tensor =
      Output(0, {num_images}, at::dtype<int32_t>());
  Tensor* output_boxes_tensor =
      Output(1, {max_total_proposals, 4}, at::dtype<float>());
  Tensor* output_scores_tensor = nullptr;

  int32_t* output_batch_splits_ptr =
      output_batch_splits_tensor->template mutable_data<int32_t>();
  float* output_boxes_ptr = output_boxes_tensor->template mutable_data<float>();
  float* output_scores_ptr = nullptr;

  if (OutputSize() > 1) {
    output_scores_tensor = Output(2, {max_total_proposals}, at::dtype<float>());
    output_scores_ptr = output_scores_tensor->template mutable_data<float>();
  }

  int64_t num_output_rois = 0;
  for (int image_idx = 0; image_idx < num_images; image_idx++) {
    const float image_height = image_info_ptr[image_idx * 2 + 0];
    const float image_width = image_info_ptr[image_idx * 2 + 1];

    const float* image_deltas =
        input_deltas_ptr + image_idx * num_anchors * 4 * features_pixels;
    const float* image_scores =
        input_scores_ptr + image_idx * num_anchors * features_pixels;

    std::vector<int> score_ordered_proposals(num_anchors * features_pixels);
    std::iota(
        score_ordered_proposals.begin(), score_ordered_proposals.end(), 0);
    if (max_pre_nms_proposals_ <= 0 ||
        max_pre_nms_proposals_ >= score_ordered_proposals.size()) {
      std::sort(
          score_ordered_proposals.begin(),
          score_ordered_proposals.end(),
          [image_scores](int lhs, int rhs) {
            return image_scores[lhs] > image_scores[rhs];
          });
    } else {
      std::partial_sort(
          score_ordered_proposals.begin(),
          score_ordered_proposals.begin() + max_pre_nms_proposals_,
          score_ordered_proposals.end(),
          [image_scores](int lhs, int rhs) {
            return image_scores[lhs] > image_scores[rhs];
          });
      score_ordered_proposals.resize(max_pre_nms_proposals_);
    }

    std::vector<AABB> ordered_boxes;
    std::vector<float> ordered_scores;
    ordered_boxes.reserve(score_ordered_proposals.size());
    ordered_scores.reserve(score_ordered_proposals.size());
    if (order_ == StorageOrder::NCHW) {
      for (int roi_idx : score_ordered_proposals) {
        const int anchor_idx = roi_idx / features_pixels;
        const int spatial_idx = roi_idx % features_pixels;
        const int x = spatial_idx % features_width;
        const int y = spatial_idx / features_width;

        const AABB anchor_aabb(
            float(x) * roi_stride_ + anchors_ptr[anchor_idx * 4 + 0],
            float(y) * roi_stride_ + anchors_ptr[anchor_idx * 4 + 1],
            float(x) * roi_stride_ + anchors_ptr[anchor_idx * 4 + 2],
            float(y) * roi_stride_ + anchors_ptr[anchor_idx * 4 + 3]);

        const float dx =
            image_deltas[(anchor_idx * 4 + 0) * features_pixels + spatial_idx];
        const float dy =
            image_deltas[(anchor_idx * 4 + 1) * features_pixels + spatial_idx];
        const float dw =
            image_deltas[(anchor_idx * 4 + 2) * features_pixels + spatial_idx];
        const float dh =
            image_deltas[(anchor_idx * 4 + 3) * features_pixels + spatial_idx];

        const AABB corrected_aabb = anchor_aabb.apply_deltas(dx, dy, dw, dh);

        /* clip RoI boxes */
        const AABB clipped_aabb =
            corrected_aabb.clip(image_width, image_height);

        /* ignore RoI boxes with either features_height and features_width below
         * min_size */
        if (std::min(clipped_aabb.width(), clipped_aabb.height()) >=
            min_size_) {
          ordered_boxes.push_back(clipped_aabb);
          ordered_scores.emplace_back(image_scores[roi_idx]);
        }
      }
    } else {
      CAFFE_ENFORCE_EQ(order_, StorageOrder::NHWC);
      for (int roi_idx : score_ordered_proposals) {
        const int anchor_idx = roi_idx % num_anchors;
        const int spatial_idx = roi_idx / num_anchors;
        const int x = spatial_idx % features_width;
        const int y = spatial_idx / features_width;

        const AABB anchor_aabb(
            float(x) * roi_stride_ + anchors_ptr[anchor_idx * 4 + 0],
            float(y) * roi_stride_ + anchors_ptr[anchor_idx * 4 + 1],
            float(x) * roi_stride_ + anchors_ptr[anchor_idx * 4 + 2],
            float(y) * roi_stride_ + anchors_ptr[anchor_idx * 4 + 3]);

        const float dx = image_deltas[roi_idx * 4 + 0];
        const float dy = image_deltas[roi_idx * 4 + 1];
        const float dw = image_deltas[roi_idx * 4 + 2];
        const float dh = image_deltas[roi_idx * 4 + 3];

        const AABB corrected_aabb = anchor_aabb.apply_deltas(dx, dy, dw, dh);

        /* clip RoI boxes */
        const AABB clipped_aabb =
            corrected_aabb.clip(image_width, image_height);

        /* ignore RoI boxes with either features_height and features_width below
         * min_size */
        if (std::min(clipped_aabb.width(), clipped_aabb.height()) >=
            min_size_) {
          ordered_boxes.push_back(clipped_aabb);
          ordered_scores.emplace_back(image_scores[roi_idx]);
        }
      }
    }
    CAFFE_ENFORCE_EQ(ordered_boxes.size(), ordered_scores.size());

    std::vector<size_t> orderedIndices(ordered_boxes.size());
    std::iota(orderedIndices.begin(), orderedIndices.end(), 0);
    const int max_post_nms_rois = max_post_nms_proposals_ > 0
        ? max_post_nms_proposals_
        : orderedIndices.size();
    std::vector<size_t> outputIndices;
    while (orderedIndices.size() > 0 &&
           outputIndices.size() < max_post_nms_rois) {
      /* get the highest scored remaining RoI */
      const int p = orderedIndices[0];
      outputIndices.push_back(p);

      const float p_area = ordered_boxes[p].area();

      std::vector<size_t> newOrderedIndices;
      for (size_t i = 1; i < orderedIndices.size(); i++) {
        const int idx = orderedIndices[i];
        const float i_area = ordered_boxes[idx].area();

        const AABB intersection =
            ordered_boxes[idx].intersect(ordered_boxes[p]);
        const float intersection_area = intersection.area();
        const float union_area = i_area + p_area - intersection_area;

        if (intersection_area <= max_iou_ * union_area) {
          newOrderedIndices.push_back(idx);
        }
      }
      orderedIndices = std::move(newOrderedIndices);
    }
    for (size_t outputIdx : outputIndices) {
      *output_boxes_ptr++ = ordered_boxes[outputIdx].x1;
      *output_boxes_ptr++ = ordered_boxes[outputIdx].y1;
      *output_boxes_ptr++ = ordered_boxes[outputIdx].x2;
      *output_boxes_ptr++ = ordered_boxes[outputIdx].y2;
      if (output_scores_ptr) {
        *output_scores_ptr++ = ordered_scores[outputIdx];
      }
    }
    output_batch_splits_ptr[image_idx] =
        static_cast<int32_t>(outputIndices.size());
    num_output_rois += outputIndices.size();
  }
  output_boxes_tensor->ShrinkTo(num_output_rois);
  output_scores_tensor->ShrinkTo(num_output_rois);

  return true;
}

REGISTER_CPU_OPERATOR(AABBRoIProposals, AABBRoIProposalsOp<CPUContext>);

OPERATOR_SCHEMA(AABBRoIProposals)
    .NumInputs(4)
    .NumOutputs(2, 3)
    .SetDoc(R"DOC(
Generate axis-aligned bounding box (AABB) proposals for Faster RCNN or Mask
RCNN. The proposals are generated for a list of images based on scores,
bounding box regression result (deltas), and predefined bounding box shapes
(anchors). Greedy non-maximum suppression is applied to generate the final
bounding boxes.
)DOC")
    .Arg(
        "roi_stride",
        "(float) Stride, in pixels, for output RoIs. "
        "RoI stride specifies how many RoI pixels correspond to a single "
        "pixel of the feature maps in scores and deltas inputs. "
        "Typically it is set to the ratio of image size on the input of the "
        "network to the image size in scores and deltas inputs.")
    .Arg(
        "max_pre_nms_proposals",
        "(int) Maximum number of proposals to "
        "consider before non-maximum suppression. If the number of input "
        "proposals exceeds this threshold, proposals with the lowest scores are "
        "discarded.")
    .Arg(
        "max_post_nms_proposals",
        "(int) Maximum number of proposals to "
        "keep after non-maximum suppression. If the number of post-NMS proposals "
        "exceeds this threshold, proposals with the lowest scores are discarded.")
    .Arg(
        "max_iou",
        "(float) Maximum allowed IoU between bounding boxes. "
        "Bounding boxes which have higher IoU than this threshold with "
        "another bounding box proposals are discarded.")
    .Arg(
        "min_size",
        "(float) Minimum bounding box size, in pixel coordinates. "
        "Bounding boxes where either height or width is below this threshold "
        "are discarded.")
    .Arg(
        "order",
        "(string): order of dimensions of scores and deltas tensor. "
        "Only \"NCHW\" (default) and \"NHWC\" order is supported.")
    .Input(
        0,
        "scores",
        "Scores for bounding box proposals. "
        "If order is \"NCHW\" the shape is (num_images, num_anchors, height, width). "
        "If order is \"NHWC\" the shape is (num_images, height, width, num_anchors). "
        "The format of deltas is (dx, dy, dw, dh).")
    .Input(
        1,
        "deltas",
        "Bounding box deltas for bounding box proposals. "
        "If order is \"NCHW\" the shape is (num_images, num_anchors * 4, height, width). "
        "If order is \"NHWC\" the shape is (num_images, height, width, num_anchors * 4). "
        "The format of deltas is (dx, dy, dw, dh).")
    .Input(
        2,
        "anchors",
        "Position-independent bounding box anchors, in pixel coordinates, "
        "size (num_anchors, 4), "
        "format (x1, y1, x2, y2)")
    .Input(
        3,
        "image_info",
        "Image information, "
        "size (num_images, 2), "
        "format (height, width).")
    .Output(
        0,
        "batch_splits",
        "1D tensor of shape (num_images) where each element specifies the "
        "number of proposals for the corresponding image in the input batch")
    .Output(
        1,
        "boxes",
        "Axis-aligned bounding boxes for RoI proposals in pixel coordinates, "
        "size (num_proposals, 4), "
        "format (x1, y1, x2, y2)")
    .Output(2, "scores", "Scores of proposals, size (num_proposals)");

SHOULD_NOT_DO_GRADIENT(AABBRoIProposals);

} // namespace caffe2
