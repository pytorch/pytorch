#include "caffe2/operators/aabb_roi_nms_op.h"

namespace caffe2 {

template <>
bool AABBRoINMSOp<CPUContext>::RunOnDevice() {
  const Tensor& input_batch_splits_tensor = Input(0);
  const Tensor& input_scores_tensor = Input(1);
  const Tensor& input_boxes_tensor = Input(2);

  const int box_dim = 4;

  // input_scores_tensor: (num_boxes, num_classes), 0 for background
  if (input_scores_tensor.dim() == 4) {
    CAFFE_ENFORCE_EQ(input_scores_tensor.size(2), 1);
    CAFFE_ENFORCE_EQ(input_scores_tensor.size(3), 1);
  } else {
    CAFFE_ENFORCE_EQ(input_scores_tensor.dim(), 2);
  }
  CAFFE_ENFORCE(input_scores_tensor.template IsType<float>());

  // input_boxes_tensor: (num_boxes, num_classes * box_dim)
  if (input_boxes_tensor.dim() == 4) {
    CAFFE_ENFORCE_EQ(input_boxes_tensor.size(2), 1);
    CAFFE_ENFORCE_EQ(input_boxes_tensor.size(3), 1);
  } else {
    CAFFE_ENFORCE_EQ(input_boxes_tensor.dim(), 2);
  }
  CAFFE_ENFORCE(input_boxes_tensor.template IsType<float>());

  int N = input_scores_tensor.size(0);
  int num_classes = input_scores_tensor.size(1);

  CAFFE_ENFORCE_EQ(N, input_boxes_tensor.size(0));
  CAFFE_ENFORCE_EQ(num_classes * box_dim, input_boxes_tensor.size(1));

  // input_scores_tensor and input_boxes_tensor have items from multiple images
  // in a batch. Get the corresponding batch splits from input.
  CAFFE_ENFORCE_EQ(input_batch_splits_tensor.dim(), 1);
  const int batch_size = input_batch_splits_tensor.size(0);
  const int32_t* batch_splits_data = input_batch_splits_tensor.data<int32_t>();

  Tensor* output_batch_splits_tensor =
      Output(0, {batch_size}, at::dtype<int32_t>());
  Tensor* output_scores_tensor = Output(1, {0}, at::dtype<float>());
  Tensor* output_boxes_tensor = Output(2, {0, box_dim}, at::dtype<float>());
  Tensor* output_classes_tensor = Output(3, {0}, at::dtype<int32_t>());

  vector<int> total_keep_per_batch(batch_size);
  const float* scores_ptr = input_scores_tensor.data<float>();
  const float* boxes_ptr = input_boxes_tensor.data<float>();
  int roi_start = 0;
  for (int b = 0; b < batch_size; ++b) {
    int num_boxes = batch_splits_data[b];
    vector<vector<int>> keeps(num_classes);

    // Perform nms to each class
    // skip class_idx = 0, because it's the background class
    int total_keep_count = 0;
    for (int class_idx = 1; class_idx < num_classes; class_idx++) {
      std::vector<int> indices;
      for (int i = 0; i < num_boxes; i++) {
        if (scores_ptr[i * num_classes + class_idx] > min_score_) {
          indices.push_back(i);
        }
      }
      std::sort(
          indices.begin(),
          indices.end(),
          [scores_ptr, num_classes, class_idx](int lhs, int rhs) {
            return scores_ptr[lhs * num_classes + class_idx] >
                scores_ptr[rhs * num_classes + class_idx];
          });
      const int max_post_nms_proposals =
          max_objects_ > 0 ? max_objects_ : num_boxes;
      std::vector<int>& keep = keeps[class_idx];
      while (indices.size() > 0 && keep.size() < max_post_nms_proposals) {
        /* get the highest scored remaining proposal */
        const int p = indices[0];
        keep.push_back(p);

        const float p_x1 = boxes_ptr[(p * num_classes + class_idx) * 4 + 0];
        const float p_y1 = boxes_ptr[(p * num_classes + class_idx) * 4 + 1];
        const float p_x2 = boxes_ptr[(p * num_classes + class_idx) * 4 + 2];
        const float p_y2 = boxes_ptr[(p * num_classes + class_idx) * 4 + 3];
        const float p_width = p_x2 - p_x1;
        const float p_height = p_y2 - p_y1;
        const float p_area = p_width * p_height;

        std::vector<int> new_indices;
        for (size_t i = 1; i < indices.size(); i++) {
          const int idx = indices[i];
          const float i_x1 = boxes_ptr[(idx * num_classes + class_idx) * 4 + 0];
          const float i_y1 = boxes_ptr[(idx * num_classes + class_idx) * 4 + 1];
          const float i_x2 = boxes_ptr[(idx * num_classes + class_idx) * 4 + 2];
          const float i_y2 = boxes_ptr[(idx * num_classes + class_idx) * 4 + 3];
          const float i_width = i_x2 - i_x1;
          const float i_height = i_y2 - i_y1;
          const float i_area = i_width * i_height;

          /* compute coordinates of the intersection of proposals p and i */
          const float intersection_x1 = std::max(p_x1, i_x1);
          const float intersection_y1 = std::max(p_y1, i_y1);
          const float intersection_x2 = std::min(p_x2, i_x2);
          const float intersection_y2 = std::min(p_y2, i_y2);

          /* compute dimensions of the intersection of proposals p and i */
          const float intersection_width =
              std::max(intersection_x2 - intersection_x1, 0.0f);
          const float intersection_height =
              std::max(intersection_y2 - intersection_y1, 0.0f);
          const float intersection_area =
              intersection_width * intersection_height;
          const float union_area = i_area + p_area - intersection_area;

          if (intersection_area <= max_iou_ * union_area) {
            new_indices.push_back(idx);
          }
        }
        indices = std::move(new_indices);
      }
      total_keep_count += keep.size();
    }

    // Limit to max_rois detections *over all classes*
    if (max_objects_ > 0 && total_keep_count > max_objects_) {
      // merge all scores (represented by indices) together and sort
      // flatten keeps[i][class_idx] to [pair(i, keeps[i][class_idx]), ...]
      // first: class index (1 ~ keeps.size() - 1),
      // second: values in keeps[first]
      vector<std::pair<int, int>> all_objects;
      all_objects.reserve(total_keep_count);

      for (int i = 1; i < num_classes; i++) {
        for (auto& ckv : keeps[i]) {
          all_objects.push_back(std::pair<int, int>{i, ckv});
        }
      }

      std::sort(
          all_objects.begin(),
          all_objects.end(),
          [scores_ptr, num_classes](
              const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) {
            return scores_ptr[lhs.second * num_classes + lhs.first] >
                scores_ptr[rhs.second * num_classes + rhs.first];
          });

      // Pick the first `max_objects_` boxes with highest scores
      DCHECK_GT(all_objects.size(), max_objects_);

      // Reconstruct keeps from `all_rois`
      for (auto& keep : keeps) {
        keep.clear();
      }
      for (int i = 0; i < max_objects_; i++) {
        DCHECK_GT(all_objects.size(), i);
        const std::pair<int, int> obj = all_objects[i];
        keeps[obj.first].push_back(obj.second);
      }
      total_keep_count = max_objects_;
    }
    total_keep_per_batch[b] = total_keep_count;

    // Write results
    int cur_start_idx = output_scores_tensor->size(0);
    output_scores_tensor->Extend(total_keep_count, 50);
    output_boxes_tensor->Extend(total_keep_count, 50);
    output_classes_tensor->Extend(total_keep_count, 50);

    float* output_scores_ptr =
        output_scores_tensor->template mutable_data<float>();
    float* output_boxes_ptr =
        output_boxes_tensor->template mutable_data<float>();
    int32_t* output_classes_ptr =
        output_classes_tensor->template mutable_data<int32_t>();

    int cur_out_idx = 0;
    for (int class_idx = 1; class_idx < num_classes; class_idx++) {
      auto& cur_keep = keeps[class_idx];

      for (int k = 0; k < keeps[class_idx].size(); k++) {
        const int box_idx = keeps[class_idx][k];
        output_scores_ptr[cur_start_idx + cur_out_idx + k] =
            scores_ptr[box_idx * num_classes + class_idx];
        for (int component_idx = 0; component_idx < 4; component_idx++) {
          output_boxes_ptr
              [(cur_start_idx + cur_out_idx + k) * 4 + component_idx] =
                  boxes_ptr
                      [(box_idx * num_classes + class_idx) * 4 + component_idx];
        }
      }
      for (int k = 0; k < keeps[class_idx].size(); k++) {
        output_classes_ptr[cur_start_idx + cur_out_idx + k] = class_idx;
      }

      cur_out_idx += keeps[class_idx].size();
    }

    roi_start += num_boxes;
  }

  int32_t* batch_splits_out_data =
      output_batch_splits_tensor->template mutable_data<int32_t>();
  for (int i = 0; i < batch_size; i++) {
    batch_splits_out_data[i] = int32_t(total_keep_per_batch[i]);
  }

  return true;
}

namespace {

REGISTER_CPU_OPERATOR(AABBRoINMS, AABBRoINMSOp<CPUContext>);

OPERATOR_SCHEMA(AABBRoINMS)
    .NumInputs(3)
    .NumOutputs(4)
    .SetDoc(R"DOC(
Apply NMS to each class (except background) and limit the number of
returned boxes.
)DOC")
    .Arg(
        "min_score",
        "(float) Minimum score for preserved bounding boxes. "
        "Input bounding boxes with lower scores are discarded.")
    .Arg(
        "max_iou",
        "(float) Maximum allowed IoU between bounding boxes of the same class. "
        "Input bounding boxes which have higher IoU than this threshold with "
        "another bounding box of the same class are discarded.")
    .Arg("max_objects", "(int) Maximum number of detected objects per image.")
    .Input(
        0,
        "batch_splits",
        "Tensor of shape (batch_size) with each element denoting the number "
        "of RoIs/boxes belonging to the corresponding image in batch. "
        "Sum should add up to total count of scores/boxes.")
    .Input(1, "scores", "Scores, size (num_rois, num_classes)")
    .Input(
        2,
        "boxes",
        "Bounding box for each class, size (count, num_classes * 4). "
        "Size: (num_rois, num_classes * 4).")
    .Output(
        0,
        "batch_splits",
        "Output batch splits for scores/boxes after applying NMS")
    .Output(1, "scores", "Filtered scores, size (n)")
    .Output(2, "boxes", "Filtered boxes, size (n, 4).")
    .Output(3, "classes", "Class id for each filtered score/box, size (n)");

SHOULD_NOT_DO_GRADIENT(AABBRoINMS);

} // namespace
} // namespace caffe2
