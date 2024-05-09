#include "box_with_nms_limit_op.h"
#include "caffe2/utils/eigen_utils.h"
#include "generate_proposals_op_util_nms.h"

namespace caffe2 {

template <>
template <typename T>
bool BoxWithNMSLimitOp<CPUContext>::DoRunWithType() {
const auto& tscores = Input(0);
  const auto& tboxes = Input(1);

  const int box_dim = rotated_ ? 5 : 4;

  // tscores: (num_boxes, num_classes), 0 for background
  if (tscores.dim() == 4) {
    CAFFE_ENFORCE_EQ(tscores.size(2), 1);
    CAFFE_ENFORCE_EQ(tscores.size(3), 1);
  } else {
    CAFFE_ENFORCE_EQ(tscores.dim(), 2);
  }
  CAFFE_ENFORCE(tscores.template IsType<float>(), tscores.dtype().name());
  // tboxes: (num_boxes, num_classes * box_dim)
  if (tboxes.dim() == 4) {
    CAFFE_ENFORCE_EQ(tboxes.size(2), 1);
    CAFFE_ENFORCE_EQ(tboxes.size(3), 1);
  } else {
    CAFFE_ENFORCE_EQ(tboxes.dim(), 2);
  }
  CAFFE_ENFORCE(tboxes.template IsType<float>(), tboxes.dtype().name());

  int N = tscores.size(0);
  int num_classes = tscores.size(1);

  CAFFE_ENFORCE_EQ(N, tboxes.size(0));
  int num_boxes_classes = get_box_cls_index(num_classes - 1) + 1;
  CAFFE_ENFORCE_EQ(num_boxes_classes * box_dim, tboxes.size(1));

  // Default value for batch_size and batch_splits
  int batch_size = 1;
  vector<T> batch_splits_default(1, tscores.size(0));
  const T* batch_splits_data = batch_splits_default.data();
  if (InputSize() > 2) {
    // tscores and tboxes have items from multiple images in a batch. Get the
    // corresponding batch splits from input.
    const auto& tbatch_splits = Input(2);
    CAFFE_ENFORCE_EQ(tbatch_splits.dim(), 1);
    batch_size = tbatch_splits.size(0);
    batch_splits_data = tbatch_splits.data<T>();
  }
  Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> batch_splits(batch_splits_data, batch_size);
  CAFFE_ENFORCE_EQ(batch_splits.sum(), N);

  auto* out_scores = Output(0, {0}, at::dtype<float>());
  auto* out_boxes = Output(1, {0, box_dim}, at::dtype<float>());
  auto* out_classes = Output(2, {0}, at::dtype<float>());

  Tensor* out_keeps = nullptr;
  Tensor* out_keeps_size = nullptr;
  if (OutputSize() > 4) {
    out_keeps = Output(4);
    out_keeps_size = Output(5);
    out_keeps->Resize(0);
    out_keeps_size->Resize(batch_size, num_classes);
  }

  vector<int> total_keep_per_batch(batch_size);
  int offset = 0;
  for (int b = 0; b < batch_splits.size(); ++b) {
    int num_boxes = batch_splits[b];
    Eigen::Map<const ERArrXXf> scores(
        tscores.data<float>() + offset * tscores.size(1),
        num_boxes,
        tscores.size(1));
    Eigen::Map<const ERArrXXf> boxes(
        tboxes.data<float>() + offset * tboxes.size(1),
        num_boxes,
        tboxes.size(1));

    // To store updated scores if SoftNMS is used
    ERArrXXf soft_nms_scores(num_boxes, tscores.size(1));
    vector<vector<int>> keeps(num_classes);

    // Perform nms to each class
    // skip j = 0, because it's the background class
    int total_keep_count = 0;
    for (int j = 1; j < num_classes; j++) {
      auto cur_scores = scores.col(get_score_cls_index(j));
      auto inds = utils::GetArrayIndices(cur_scores > score_thres_);
      auto cur_boxes =
          boxes.block(0, get_box_cls_index(j) * box_dim, boxes.rows(), box_dim);

      if (soft_nms_enabled_) {
        auto cur_soft_nms_scores = soft_nms_scores.col(get_score_cls_index(j));
        keeps[j] = utils::soft_nms_cpu(
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
        std::stable_sort(
            inds.data(),
            inds.data() + inds.size(),
            [&cur_scores](int lhs, int rhs) {
              return cur_scores(lhs) > cur_scores(rhs);
            });
        int keep_max = detections_per_im_ > 0 ? detections_per_im_ : -1;
        keeps[j] = utils::nms_cpu(
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
      new (&scores) Eigen::Map<const ERArrXXf>(
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
        vector<KeepIndex> ret(total_keep_count);

        int ret_idx = 0;
        for (int j = 1; j < num_classes; j++) {
          auto& cur_keep = keeps[j];
          for (auto& ckv : cur_keep) {
            ret[ret_idx++] = {j, ckv};
          }
        }

        std::stable_sort(
            ret.data(),
            ret.data() + ret.size(),
            [this, &scores](const KeepIndex& lhs, const KeepIndex& rhs) {
              return scores(lhs.second, this->get_score_cls_index(lhs.first)) >
                  scores(rhs.second, this->get_score_cls_index(rhs.first));
            });

        return ret;
      };

      // Pick the first `detections_per_im_` boxes with highest scores
      auto all_scores_sorted = get_all_scores_sorted();
      TORCH_DCHECK_GT(all_scores_sorted.size(), detections_per_im_);

      // Reconstruct keeps from `all_scores_sorted`
      for (auto& cur_keep : keeps) {
        cur_keep.clear();
      }
      for (int i = 0; i < detections_per_im_; i++) {
        TORCH_DCHECK_GT(all_scores_sorted.size(), i);
        auto& cur = all_scores_sorted[i];
        keeps[cur.first].push_back(cur.second);
      }
      total_keep_count = detections_per_im_;
    }
    total_keep_per_batch[b] = total_keep_count;

    // Write results
    int cur_start_idx = out_scores->size(0);
    out_scores->Extend(total_keep_count, 50);
    out_boxes->Extend(total_keep_count, 50);
    out_classes->Extend(total_keep_count, 50);

    int cur_out_idx = 0;
    for (int j = 1; j < num_classes; j++) {
      auto cur_scores = scores.col(get_score_cls_index(j));
      auto cur_boxes =
          boxes.block(0, get_box_cls_index(j) * box_dim, boxes.rows(), box_dim);
      auto& cur_keep = keeps[j];
      Eigen::Map<EArrXf> cur_out_scores(
          out_scores->template mutable_data<float>() + cur_start_idx +
              cur_out_idx,
          cur_keep.size());
      Eigen::Map<ERArrXXf> cur_out_boxes(
          out_boxes->mutable_data<float>() +
              (cur_start_idx + cur_out_idx) * box_dim,
          cur_keep.size(),
          box_dim);
      Eigen::Map<EArrXf> cur_out_classes(
          out_classes->template mutable_data<float>() + cur_start_idx +
              cur_out_idx,
          cur_keep.size());

      utils::GetSubArray(
          cur_scores, utils::AsEArrXt(cur_keep), &cur_out_scores);
      utils::GetSubArrayRows(
          cur_boxes, utils::AsEArrXt(cur_keep), &cur_out_boxes);
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      for (int k = 0; k < cur_keep.size(); k++) {
        cur_out_classes[k] =
            static_cast<float>(j - !output_classes_include_bg_cls_);
      }

      cur_out_idx += cur_keep.size();
    }

    if (out_keeps) {
      out_keeps->Extend(total_keep_count, 50);

      Eigen::Map<EArrXi> out_keeps_arr(
          out_keeps->template mutable_data<int>() + cur_start_idx,
          total_keep_count);
      Eigen::Map<EArrXi> cur_out_keeps_size(
          out_keeps_size->template mutable_data<int>() + b * num_classes,
          num_classes);

      cur_out_idx = 0;
      for (int j = 0; j < num_classes; j++) {
        out_keeps_arr.segment(cur_out_idx, keeps[j].size()) =
            utils::AsEArrXt(keeps[j]);
        cur_out_keeps_size[j] = keeps[j].size();
        cur_out_idx += keeps[j].size();
      }
    }

    offset += num_boxes;
  }

  if (OutputSize() > 3) {
    auto* batch_splits_out = Output(3, {batch_size}, at::dtype<float>());
    Eigen::Map<EArrXf> batch_splits_out_map(
        batch_splits_out->template mutable_data<float>(), batch_size);
    batch_splits_out_map =
        Eigen::Map<const EArrXi>(total_keep_per_batch.data(), batch_size)
            .cast<float>();
  }

  return true;
}

namespace {

REGISTER_CPU_OPERATOR(BoxWithNMSLimit, BoxWithNMSLimitOp<CPUContext>);

OPERATOR_SCHEMA(BoxWithNMSLimit)
    .NumInputs(2, 3)
    .NumOutputs(3, 6)
    .SetDoc(R"DOC(
Apply NMS to each class (except background) and limit the number of
returned boxes.
)DOC")
    .Arg("score_thresh", "(float) TEST.SCORE_THRESH")
    .Arg("nms", "(float) TEST.NMS")
    .Arg("detections_per_im", "(int) TEST.DETECTIONS_PER_IM")
    .Arg("soft_nms_enabled", "(bool) TEST.SOFT_NMS.ENABLED")
    .Arg("soft_nms_method", "(string) TEST.SOFT_NMS.METHOD")
    .Arg("soft_nms_sigma", "(float) TEST.SOFT_NMS.SIGMA")
    .Arg(
        "soft_nms_min_score_thres",
        "(float) Lower bound on updated scores to discard boxes")
    .Arg(
        "rotated",
        "bool (default false). If true, then boxes (rois and deltas) include "
        "angle info to handle rotation. The format will be "
        "[ctr_x, ctr_y, width, height, angle (in degrees)].")
    .Input(0, "scores", "Scores, size (count, num_classes)")
    .Input(
        1,
        "boxes",
        "Bounding box for each class, size (count, num_classes * 4). "
        "For rotated boxes, this would have an additional angle (in degrees) "
        "in the format [<optional_batch_id>, ctr_x, ctr_y, w, h, angle]. "
        "Size: (count, num_classes * 5).")
    .Input(
        2,
        "batch_splits",
        "Tensor of shape (batch_size) with each element denoting the number "
        "of RoIs/boxes belonging to the corresponding image in batch. "
        "Sum should add up to total count of scores/boxes.")
    .Output(0, "scores", "Filtered scores, size (n)")
    .Output(
        1,
        "boxes",
        "Filtered boxes, size (n, 4). "
        "For rotated boxes, size (n, 5), format [ctr_x, ctr_y, w, h, angle].")
    .Output(2, "classes", "Class id for each filtered score/box, size (n)")
    .Output(
        3,
        "batch_splits",
        "Output batch splits for scores/boxes after applying NMS")
    .Output(4, "keeps", "Optional filtered indices, size (n)")
    .Output(
        5,
        "keeps_size",
        "Optional number of filtered indices per class, size (num_classes)");

SHOULD_NOT_DO_GRADIENT(BoxWithNMSLimit);

} // namespace
} // namespace caffe2

// clang-format off
C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    BoxWithNMSLimit,
    "_caffe2::BoxWithNMSLimit("
      "Tensor scores, "
      "Tensor boxes, "
      "Tensor batch_splits, "
      "float score_thresh, "
      "float nms, "
      "int detections_per_im, "
      "bool soft_nms_enabled, "
      "str soft_nms_method, "
      "float soft_nms_sigma, "
      "float soft_nms_min_score_thres, "
      "bool rotated, "
      "bool cls_agnostic_bbox_reg, "
      "bool input_boxes_include_bg_cls, "
      "bool output_classes_include_bg_cls, "
      "bool legacy_plus_one "
    ") -> ("
      "Tensor scores, "
      "Tensor boxes, "
      "Tensor classes, "
      "Tensor batch_splits, "
      "Tensor keeps, "
      "Tensor keeps_size"
    ")",
    caffe2::BoxWithNMSLimitOp<caffe2::CPUContext>);

// clang-format on
