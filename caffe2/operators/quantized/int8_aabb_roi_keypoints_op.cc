#include "caffe2/operators/quantized/int8_aabb_roi_keypoints_op.h"
#include "caffe2/operators/quantized/int8_utils.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(Int8AABBRoIKeypoints, Int8AABBRoIKeypointsOp<CPUContext>);

OPERATOR_SCHEMA(Int8AABBRoIKeypoints)
    .NumInputs(2)
    .NumOutputs(2)
    .SetDoc(R"DOC(
Estimate keypoints' location from heatmap.
)DOC")
    .Arg(
        "order",
        "(string): order of dimensions of scores and deltas tensor. "
        "Only \"NHWC\" order is supported.")
    .Arg("Y_scale",
        "Scale quantization parameter for output scores tensor.")
    .Arg("Y_zero_point",
        "Zero point quantization parameter for output scores tensor.")
    .Input(
        0,
        "heatmaps",
        "Tensor with heatmaps of keypoint scores for each keypoint and each RoI. "
        "Size (num_rois, num_keypoints, height, width).")
    .Input(
        1,
        "boxes",
        "Axis-aligned bounding boxes for RoIs in pixel coordinates, "
        "Size (num_rois, 4), format [x1, y1, x2, y2].")
    .Output(
        0,
        "keypoints",
        "Pixel coordinates for the keypoints, in the space of input image, "
        "Size (num_rois, num_keypoints, 2), "
        "format [x, y].")
    .Output(
        0,
        "scores",
        "Scores for the corresponding keypoints, "
        "Size (num_rois, num_keypoints).");

SHOULD_NOT_DO_GRADIENT(Int8AABBRoIKeypoints);
} // namespace

/**
Mask R-CNN uses bicubic upscaling before taking the maximum of the heat map
for keypoints. We would like to avoid bicubic upscaling, because it is
computationally expensive. This approach uses the Taylor expansion up to the
quadratic terms on approximation of the heatmap function.
**/
template <>
bool Int8AABBRoIKeypointsOp<CPUContext>::RunOnDevice() {
  const int8::Int8TensorCPU& input_heatmaps_tensor =
      Inputs()[0]->template Get<int8::Int8TensorCPU>();
  const Tensor& input_boxes_tensor = Input(1);

  CAFFE_ENFORCE_EQ(input_heatmaps_tensor.t.dim(), 4);
  const size_t num_rois = input_heatmaps_tensor.t.size(0);
  const size_t num_keypoints = input_heatmaps_tensor.t.size(3);
  const size_t heatmap_height = input_heatmaps_tensor.t.size(1);
  const size_t heatmap_width = input_heatmaps_tensor.t.size(2);
  const int32_t heatmap_zero_point = input_heatmaps_tensor.zero_point;
  const float heatmap_scale = input_heatmaps_tensor.scale;
  CAFFE_ENFORCE_GE(heatmap_height, 2); // at least 2x2 for approx
  CAFFE_ENFORCE_GE(heatmap_width, 2); // at least 2x2 for approx

  CAFFE_ENFORCE_EQ(input_boxes_tensor.dim(), 2);
  CAFFE_ENFORCE_EQ(input_boxes_tensor.dim32(0), num_rois);
  CAFFE_ENFORCE_GE(input_boxes_tensor.dim32(1), 4);

  const uint8_t* input_heatmaps_ptr = input_heatmaps_tensor.t.data<uint8_t>();
  const uint16_t* input_boxes_ptr = input_boxes_tensor.data<uint16_t>();

  Tensor* output_keypoints_tensor =
      Output(0, {num_rois, num_keypoints, 2}, at::dtype<uint16_t>());
  Tensor* output_scores_tensor = nullptr;

  uint16_t* output_keypoints_ptr = output_keypoints_tensor->mutable_data<uint16_t>();
  uint8_t* output_scores_ptr = nullptr;

  const int32_t scores_zero_point =
    this->template GetSingleArgument<int>("Y_zero_point", 0);
  const float scores_scale = this->template GetSingleArgument<float>("Y_scale", 1.0f);

  if (OutputSize() > 1) {
    int8::Int8TensorCPU* output_scores_tensor =
      Outputs()[1]->template GetMutable<int8::Int8TensorCPU>();
    output_scores_tensor->scale = scores_scale;
    output_scores_tensor->zero_point = scores_zero_point;
    output_scores_tensor->t.Resize(num_rois, num_keypoints);
    output_scores_ptr = output_scores_tensor->t.template mutable_data<uint8_t>();
  }

  for (size_t roi_idx = 0; roi_idx < num_rois; roi_idx++) { // For each box, even skipped
    const float x1 = float(input_boxes_ptr[roi_idx * 4 + 0]) * 0.125f;
    const float y1 = float(input_boxes_ptr[roi_idx * 4 + 1]) * 0.125f;
    const float x2 = float(input_boxes_ptr[roi_idx * 4 + 2]) * 0.125f;
    const float y2 = float(input_boxes_ptr[roi_idx * 4 + 3]) * 0.125f;

    const float roi_width = std::max(x2 - x1, 1.0f);
    const float roi_height = std::max(y2 - y1, 1.0f);

    const float stride_x = roi_width / float(int32_t(heatmap_width));
    const float stride_y = roi_height / float(int32_t(heatmap_height));

    const float roi_x = x1 + stride_x * 0.5f;
    const float roi_y = y1 + stride_y * 0.5f;

    const uint8_t* roi_heatmap_ptr =
        input_heatmaps_ptr + roi_idx * num_keypoints * heatmap_height * heatmap_width;
    uint16_t* roi_keypoints_ptr =
        output_keypoints_ptr + roi_idx * num_keypoints * 2;

    // Extract max keypoints and probabilities from heatmaps
    for (size_t keypoint_idx = 0; keypoint_idx < num_keypoints; keypoint_idx++) {
      size_t max_y = 0;
      size_t max_x = 0;
      float max_score = heatmap_scale * (int32_t(roi_heatmap_ptr[keypoint_idx]) - heatmap_zero_point);
      for (size_t y = 0; y < heatmap_height; y++) {
        for (size_t x = 0; x < heatmap_width; x++) {
          const float score = heatmap_scale *
            (int32_t(roi_heatmap_ptr[(y * heatmap_width + x) * num_keypoints + keypoint_idx]) - heatmap_zero_point);
          if (score > max_score) {
            max_score = score;
            max_y = y;
            max_x = x;
          }
        }
      
}
      /*
       * Interpolate location of the maximum to obtain non-integer keypoint
       * position.
       * - Extract 3x3 neighbourhood of the maximum
       * - Treat the values in 3x3 grid as values of a continuous function
       * - Use the values in 3x3 grid to approximate first- and second-order
       * derivatives of the function
       * - Use the derivatives to perform one Newton-Raphson iteration to refine
       * the position of the maximum
       */

      const size_t iy1 = max_y;
      const size_t ix1 = max_x;
      size_t iy0 = iy1 - 1;
      size_t ix0 = ix1 - 1;
      size_t iy2 = iy1 + 1;
      size_t ix2 = ix1 + 1;
      if (iy1 == 0) {
        iy0 = iy2;
      }
      if (ix1 == 0) {
        ix0 = ix2;
      }
      if (iy2 >= heatmap_height) {
        iy2 = iy0;
      }
      if (ix2 >= heatmap_width) {
        ix2 = ix0;
      }

      const float fmax00 = heatmap_scale *
        (int32_t(roi_heatmap_ptr[(iy0 * heatmap_width + ix0) * num_keypoints + keypoint_idx]) - heatmap_zero_point);
      const float fmax01 = heatmap_scale *
        (int32_t(roi_heatmap_ptr[(iy0 * heatmap_width + ix1) * num_keypoints + keypoint_idx]) - heatmap_zero_point);
      const float fmax02 = heatmap_scale *
        (int32_t(roi_heatmap_ptr[(iy0 * heatmap_width + ix2) * num_keypoints + keypoint_idx]) - heatmap_zero_point);
      const float fmax10 = heatmap_scale *
        (int32_t(roi_heatmap_ptr[(iy1 * heatmap_width + ix0) * num_keypoints + keypoint_idx]) - heatmap_zero_point);
      const float fmax11 = max_score;
      const float fmax12 = heatmap_scale *
        (int32_t(roi_heatmap_ptr[(iy1 * heatmap_width + ix2) * num_keypoints + keypoint_idx]) - heatmap_zero_point);
      const float fmax20 = heatmap_scale *
        (int32_t(roi_heatmap_ptr[(iy2 * heatmap_width + ix0) * num_keypoints + keypoint_idx]) - heatmap_zero_point);
      const float fmax21 = heatmap_scale *
        (int32_t(roi_heatmap_ptr[(iy2 * heatmap_width + ix1) * num_keypoints + keypoint_idx]) - heatmap_zero_point);
      const float fmax22 = heatmap_scale *
        (int32_t(roi_heatmap_ptr[(iy2 * heatmap_width + ix2) * num_keypoints + keypoint_idx]) - heatmap_zero_point);

      /* b = [b0, b1] = -f'(0) */
      const float b0 = 0.5f * (fmax10 - fmax12);
      const float b1 = 0.5f * (fmax01 - fmax21);

      /* A = f''(0) */
      const float a00 = fmax10 - 2.0f * fmax11 + fmax12;
      const float a01 = 0.25f * (fmax22 - fmax20 - fmax02 + fmax00);
      /* const float a10 = a01; */
      const float a11 = fmax01 - 2.0f * fmax11 + fmax21;

      float d0 = 0.0f, d1 = 0.0f;
      const float d_limit = 1.5f;
      const float det_a = a00 * a11 - a01 * a01;
      if (fabsf(det_a) >= 1.0e-8f) {
        /* Directly solve the system using Cramer's rule */
        d0 = (b0 * a11 - b1 * a01) / det_a;
        d1 = (b1 * a00 - b0 * a01) / det_a;

        /* clip d if going out-of-range of 3x3 grid */
        const float d_max = std::max(fabsf(d0), fabsf(d1));
        if (d_max > d_limit) {
          const float d_scale = d_limit / d_max;
          d0 *= d_scale;
          d1 *= d_scale;
        }
        max_score += 0.5f * (d0 * d0 * a00 + d1 * d1 * a11) + a01 * d0 * d1 -
            (b0 * d0 + b1 * d1);
      }

      roi_keypoints_ptr[keypoint_idx * 2 + 0] =
          uint16_t(std::min(std::max((roi_x + (float(int32_t(max_x)) + d0) * stride_x) * 8.0f, 0.0f), 65535.0f));
      roi_keypoints_ptr[keypoint_idx * 2 + 1] =
          uint16_t(std::min(std::max((roi_y + (float(int32_t(max_y)) + d1) * stride_y) * 8.0f, 0.0f), 65535.0f));
      if (output_scores_ptr != nullptr) {
        output_scores_ptr[roi_idx * num_keypoints + keypoint_idx] =
          int8::QuantizeUint8(scores_scale, scores_zero_point, max_score);
      }
    }
  }

  return true;
}

} // namespace caffe2
