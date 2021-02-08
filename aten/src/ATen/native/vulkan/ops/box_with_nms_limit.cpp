// Copyright 2004-present Facebook. All Rights Reserved.

#include <ATen/native/vulkan/ops/box_with_nms_limit.h>

namespace at {
namespace native {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> BoxWithNMSLimitCPUKernel(
    const at::Tensor& tscores_,
    const at::Tensor& tboxes_,
    const at::Tensor& tbatch_splits_,
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
    c10::optional<std::vector<at::Tensor>>
  ) {
  const auto tscores = tscores_.contiguous();
  const auto tboxes = tboxes_.contiguous();
  const auto tbatch_splits = tbatch_splits_.contiguous();

  const int box_dim = rotated_ ? 5 : 4;
  TORCH_CHECK(
        soft_nms_method_str_ == "linear" || soft_nms_method_str_ == "gaussian",
        "Unexpected soft_nms_method");
  int soft_nms_method_ = (soft_nms_method_str_ == "linear") ? 1 : 2;

  // tscores: (num_boxes, num_classes), 0 for background
  if (tscores.dim() == 4) {
    TORCH_CHECK(tscores.size(2) == 1);
    TORCH_CHECK(tscores.size(3) == 1);
  } else {
    TORCH_CHECK(tscores.dim() == 2);
  }
  TORCH_CHECK(tscores.dtype() == torch::kFloat32, tscores.dtype().name());
  // tboxes: (num_boxes, num_classes * box_dim)
  if (tboxes.dim() == 4) {
    TORCH_CHECK(tboxes.size(2) == 1);
    TORCH_CHECK(tboxes.size(3) == 1);
  } else {
    TORCH_CHECK(tboxes.dim() == 2);
  }
  TORCH_CHECK(tboxes.dtype() == torch::kFloat32, tboxes.dtype().name());

  int N = tscores.size(0);
  int num_classes = tscores.size(1);

  TORCH_CHECK(N == tboxes.size(0));
  int num_boxes_classes = get_box_cls_index(num_classes - 1, cls_agnostic_bbox_reg_, input_boxes_include_bg_cls_) + 1;
  TORCH_CHECK(num_boxes_classes * box_dim == tboxes.size(1));

  int batch_size = 1;
  caffe2::vector<float> batch_splits_default(1, tscores.size(0));
  const float* batch_splits_data = batch_splits_default.data();

  // tscores and tboxes have items from multiple images in a batch. Get the
  // corresponding batch splits from input.
  TORCH_CHECK(tbatch_splits.dim() == 1);
  batch_size = tbatch_splits.size(0);
  batch_splits_data = tbatch_splits.data_ptr<float>();

  Eigen::Map<const caffe2::EArrXf> batch_splits(batch_splits_data, batch_size);
  TORCH_CHECK(batch_splits.sum() == N);

  // outputs 1,2,3
  at::Tensor out_scores = torch::zeros({0}, torch::dtype(torch::kFloat32));
  at::Tensor out_boxes = torch::zeros({0, box_dim}, torch::dtype(torch::kFloat32));
  at::Tensor out_classes = torch::zeros({0}, torch::dtype(torch::kFloat32));

  // outputs 5,6
  at::Tensor out_keeps = torch::zeros({0}, torch::dtype(torch::kInt));
  at::Tensor out_keeps_size = torch::zeros({batch_size, num_classes}, torch::dtype(torch::kInt));

  std::vector<int> total_keep_per_batch(batch_size);
  int offset = 0;
  for (int b = 0; b < batch_splits.size(); ++b) {
    int num_boxes = batch_splits(b);
    Eigen::Map<const caffe2::ERArrXXf> scores(
        tscores.data_ptr<float>() + offset * tscores.size(1),
        num_boxes,
        tscores.size(1));
    Eigen::Map<const caffe2::ERArrXXf> boxes(
        tboxes.data_ptr<float>() + offset * tboxes.size(1),
        num_boxes,
        tboxes.size(1));

    // To store updated scores if SoftNMS is used
    caffe2::ERArrXXf soft_nms_scores(num_boxes, tscores.size(1));
    caffe2::vector<caffe2::vector<int>> keeps(num_classes);

    // Perform nms to each class
    // skip j = 0, because it's the background class
    int total_keep_count = 0;
    for (int j = 1; j < num_classes; j++) {
      auto cur_scores = scores.col(get_score_cls_index(j, input_boxes_include_bg_cls_));
      auto inds = caffe2::utils::GetArrayIndices(cur_scores > score_thres_);
      auto cur_boxes =
          boxes.block(0, get_box_cls_index(j, cls_agnostic_bbox_reg_, input_boxes_include_bg_cls_) * box_dim, boxes.rows(), box_dim);

      if (soft_nms_enabled_) {
        auto cur_soft_nms_scores = soft_nms_scores.col(get_score_cls_index(j, input_boxes_include_bg_cls_));
        keeps[j] = caffe2::utils::soft_nms_cpu(
            &cur_soft_nms_scores,
            cur_boxes,
            cur_scores,
            inds,
            soft_nms_sigma_,
            nms_thres_,
            soft_nms_min_score_thres_,
            soft_nms_method_,
            -1, /* topN */
            legacy_plus_one_);
      } else {
        std::sort(
            inds.data(),
            inds.data() + inds.size(),
            [&cur_scores](int lhs, int rhs) {
              return cur_scores(lhs) > cur_scores(rhs);
            });
        int keep_max = detections_per_im_ > 0 ? detections_per_im_ : -1;
        keeps[j] = caffe2::utils::nms_cpu(
            cur_boxes,
            cur_scores,
            inds,
            nms_thres_,
            keep_max,
            legacy_plus_one_);
      }
      total_keep_count += keeps[j].size();
    }

    if (soft_nms_enabled_) {
      // Re-map scores to the updated SoftNMS scores
      new (&scores) Eigen::Map<const caffe2::ERArrXXf>(
          soft_nms_scores.data(),
          soft_nms_scores.rows(),
          soft_nms_scores.cols());
    }

    // Limit to max_per_image detections *over all classes*
    if (detections_per_im_ > 0 && total_keep_count > detections_per_im_) {
      // merge all scores (represented by indices) together and sort
      auto get_all_scores_sorted = [&]() {
        // flatten keeps[i][j] to [pair(i, keeps[i][j]), ...]
        // first: class index (1 ~ keeps.size() - 1),
        // second: values in keeps[first]
        using KeepIndex = std::pair<int, int>;
        caffe2::vector<KeepIndex> ret(total_keep_count);

        int ret_idx = 0;
        for (int j = 1; j < num_classes; j++) {
          auto& cur_keep = keeps[j];
          for (auto& ckv : cur_keep) {
            ret[ret_idx++] = {j, ckv};
          }
        }

        std::sort(
            ret.data(),
            ret.data() + ret.size(),
            [&input_boxes_include_bg_cls_, &scores](const KeepIndex& lhs, const KeepIndex& rhs) {
              return scores(lhs.second, get_score_cls_index(lhs.first, input_boxes_include_bg_cls_)) >
                  scores(rhs.second, get_score_cls_index(rhs.first, input_boxes_include_bg_cls_));
            });

        return ret;
      };

      // Pick the first `detections_per_im_` boxes with highest scores
      auto all_scores_sorted = get_all_scores_sorted();
      TORCH_CHECK(all_scores_sorted.size() > detections_per_im_);

      // Reconstruct keeps from `all_scores_sorted`
      for (auto& cur_keep : keeps) {
        cur_keep.clear();
      }
      for (int i = 0; i < detections_per_im_; i++) {
        TORCH_CHECK(all_scores_sorted.size() > i);
        auto& cur = all_scores_sorted[i];
        keeps[cur.first].push_back(cur.second);
      }
      total_keep_count = detections_per_im_;
    }
    total_keep_per_batch[b] = total_keep_count;

    // Write results
    int cur_start_idx = out_scores.size(0);
    out_scores.resize_({out_scores.size(0) + total_keep_count});
    out_boxes.resize_({out_boxes.size(0) + total_keep_count, out_boxes.size(1)});
    out_classes.resize_({out_classes.size(0) + total_keep_count});

    int cur_out_idx = 0;
    for (int j = 1; j < num_classes; j++) {
      auto cur_scores = scores.col(get_score_cls_index(j, input_boxes_include_bg_cls_));
      auto cur_boxes =
          boxes.block(0, get_box_cls_index(j, cls_agnostic_bbox_reg_, input_boxes_include_bg_cls_) * box_dim, boxes.rows(), box_dim);
      auto& cur_keep = keeps[j];
      Eigen::Map<caffe2::EArrXf> cur_out_scores(
          out_scores.data_ptr<float>() + cur_start_idx +
              cur_out_idx,
          cur_keep.size());
      Eigen::Map<caffe2::ERArrXXf> cur_out_boxes(
          out_boxes.data_ptr<float>() +
              (cur_start_idx + cur_out_idx) * box_dim,
          cur_keep.size(),
          box_dim);
      Eigen::Map<caffe2::EArrXf> cur_out_classes(
          out_classes.data_ptr<float>() + cur_start_idx +
              cur_out_idx,
          cur_keep.size());

      caffe2::utils::GetSubArray(
          cur_scores, caffe2::utils::AsEArrXt(cur_keep), &cur_out_scores);
      caffe2::utils::GetSubArrayRows(
          cur_boxes, caffe2::utils::AsEArrXt(cur_keep), &cur_out_boxes);
      for (int k = 0; k < cur_keep.size(); k++) {
        cur_out_classes[k] =
            static_cast<float>(j - (output_classes_include_bg_cls_ ? 0 : 1));
      }

      cur_out_idx += cur_keep.size();
    }

    out_keeps.resize_({out_keeps.size(0) + total_keep_count});
    Eigen::Map<caffe2::EArrXi> out_keeps_arr(
        out_keeps.data_ptr<int>() + cur_start_idx,
        total_keep_count);
    Eigen::Map<caffe2::EArrXi> cur_out_keeps_size(
        out_keeps_size.data_ptr<int>() + b * num_classes,
        num_classes);

    cur_out_idx = 0;
    for (int j = 0; j < num_classes; j++) {

      out_keeps_arr.segment(cur_out_idx, keeps[j].size()) =
          caffe2::utils::AsEArrXt(keeps[j]);
      cur_out_keeps_size[j] = keeps[j].size();
      cur_out_idx += keeps[j].size();
    }

    offset += num_boxes;
  }

  at::Tensor batch_splits_out = torch::zeros({batch_size}, torch::dtype(torch::kFloat32));
  Eigen::Map<caffe2::EArrXf> batch_splits_out_map(batch_splits_out.data_ptr<float>(), batch_size);
  batch_splits_out_map =
      Eigen::Map<const caffe2::EArrXi>(total_keep_per_batch.data(), batch_size)
          .cast<float>();

  return std::make_tuple(
    out_scores,
    out_boxes,
    out_classes,
    batch_splits_out,
    out_keeps,
    out_keeps_size
  );
}

} // namespace fb
} // namespace caffe2
