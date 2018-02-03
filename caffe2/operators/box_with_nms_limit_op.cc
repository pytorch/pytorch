#include "box_with_nms_limit_op.h"
#include "caffe2/utils/eigen_utils.h"
#include "generate_proposals_op_util_nms.h"

#ifdef CAFFE2_USE_MKL
#include "caffe2/mkl/operators/operator_fallback_mkl.h"
#endif // CAFFE2_USE_MKL

namespace caffe2 {

namespace {

template <class Derived, class Func>
vector<int> filter_with_indices(
    const Eigen::ArrayBase<Derived>& array,
    const vector<int>& indices,
    const Func& func) {
  vector<int> ret;
  for (auto& cur : indices) {
    if (func(array[cur])) {
      ret.push_back(cur);
    }
  }
  return ret;
}

} // namespace

template <>
bool BoxWithNMSLimitOp<CPUContext>::RunOnDevice() {
  const auto& tscores = Input(0);
  const auto& tboxes = Input(1);
  auto* out_scores = Output(0);
  auto* out_boxes = Output(1);
  auto* out_classes = Output(2);

  // tscores: (num_boxes, num_classes), 0 for background
  if (tscores.ndim() == 4) {
    CAFFE_ENFORCE_EQ(tscores.dim(2), 1, tscores.dim(2));
    CAFFE_ENFORCE_EQ(tscores.dim(3), 1, tscores.dim(3));
  } else {
    CAFFE_ENFORCE_EQ(tscores.ndim(), 2, tscores.ndim());
  }
  CAFFE_ENFORCE(tscores.template IsType<float>(), tscores.meta().name());
  // tboxes: (num_boxes, num_classes * 4)
  if (tboxes.ndim() == 4) {
    CAFFE_ENFORCE_EQ(tboxes.dim(2), 1, tboxes.dim(2));
    CAFFE_ENFORCE_EQ(tboxes.dim(3), 1, tboxes.dim(3));
  } else {
    CAFFE_ENFORCE_EQ(tboxes.ndim(), 2, tboxes.ndim());
  }
  CAFFE_ENFORCE(tboxes.template IsType<float>(), tboxes.meta().name());

  int num_classes = tscores.dim(1);

  CAFFE_ENFORCE_EQ(tscores.dim(0), tboxes.dim(0));
  CAFFE_ENFORCE_EQ(num_classes * 4, tboxes.dim(1));

  Eigen::Map<const ERArrXXf> scores(
      tscores.data<float>(), tscores.dim(0), tscores.dim(1));
  Eigen::Map<const ERArrXXf> boxes(
      tboxes.data<float>(), tboxes.dim(0), tboxes.dim(1));

  // To store updated scores if SoftNMS is used
  ERArrXXf soft_nms_scores(tscores.dim(0), tscores.dim(1));

  vector<vector<int>> keeps(num_classes);

  // Perform nms to each class
  // skip j = 0, because it's the background class
  int total_keep_count = 0;
  for (int j = 1; j < num_classes; j++) {
    auto cur_scores = scores.col(j);
    auto inds = utils::GetArrayIndices(cur_scores > score_thres_);
    auto cur_boxes = boxes.block(0, j * 4, boxes.rows(), 4);

    if (soft_nms_enabled_) {
      auto out_scores = soft_nms_scores.col(j);
      keeps[j] = utils::soft_nms_cpu(
          &out_scores,
          cur_boxes,
          cur_scores,
          inds,
          soft_nms_sigma_,
          nms_thres_,
          soft_nms_min_score_thres_,
          soft_nms_method_);
    } else {
      std::sort(
          inds.data(),
          inds.data() + inds.size(),
          [&cur_scores](int lhs, int rhs) {
            return cur_scores(lhs) > cur_scores(rhs);
          });
      keeps[j] = utils::nms_cpu(cur_boxes, cur_scores, inds, nms_thres_);
    }
    total_keep_count += keeps[j].size();
  }

  if (soft_nms_enabled_) {
    // Re-map scores to the updated SoftNMS scores
    new (&scores) Eigen::Map<const ERArrXXf>(
        soft_nms_scores.data(), soft_nms_scores.rows(), soft_nms_scores.cols());
  }

  // Limit to max_per_image detections *over all classes*
  if (detections_per_im_ > 0 && total_keep_count > detections_per_im_) {
    // merge all scores together and sort
    auto get_all_scores_sorted = [&scores, &keeps, total_keep_count]() {
      EArrXf ret(total_keep_count);

      int ret_idx = 0;
      for (int i = 1; i < keeps.size(); i++) {
        auto& cur_keep = keeps[i];
        auto cur_scores = scores.col(i);
        auto cur_ret = ret.segment(ret_idx, cur_keep.size());
        utils::GetSubArray(cur_scores, utils::AsEArrXt(keeps[i]), &cur_ret);
        ret_idx += cur_keep.size();
      }

      std::sort(ret.data(), ret.data() + ret.size());

      return ret;
    };

    // Compute image thres based on all classes
    auto all_scores_sorted = get_all_scores_sorted();
    DCHECK_GT(all_scores_sorted.size(), detections_per_im_);
    auto image_thresh =
        all_scores_sorted[all_scores_sorted.size() - detections_per_im_];

    total_keep_count = 0;
    // filter results with image_thresh
    for (int j = 1; j < num_classes; j++) {
      auto& cur_keep = keeps[j];
      auto cur_scores = scores.col(j);
      keeps[j] =
          filter_with_indices(cur_scores, cur_keep, [&image_thresh](float sc) {
            return sc >= image_thresh;
          });
      total_keep_count += keeps[j].size();
    }
  }

  // Write results
  out_scores->Resize(total_keep_count);
  out_boxes->Resize(total_keep_count, 4);
  out_classes->Resize(total_keep_count);
  int cur_out_idx = 0;
  for (int j = 1; j < num_classes; j++) {
    auto cur_scores = scores.col(j);
    auto cur_boxes = boxes.block(0, j * 4, boxes.rows(), 4);
    auto& cur_keep = keeps[j];
    Eigen::Map<EArrXf> cur_out_scores(
        out_scores->mutable_data<float>() + cur_out_idx, cur_keep.size());
    Eigen::Map<ERArrXXf> cur_out_boxes(
        out_boxes->mutable_data<float>() + cur_out_idx * 4, cur_keep.size(), 4);
    Eigen::Map<EArrXf> cur_out_classes(
        out_classes->mutable_data<float>() + cur_out_idx, cur_keep.size());

    utils::GetSubArray(cur_scores, utils::AsEArrXt(cur_keep), &cur_out_scores);
    utils::GetSubArrayRows(
        cur_boxes, utils::AsEArrXt(cur_keep), &cur_out_boxes);
    for (int k = 0; k < cur_keep.size(); k++) {
      cur_out_classes[k] = static_cast<float>(j);
    }

    cur_out_idx += cur_keep.size();
  }

  if (OutputSize() > 3) {
    auto* out_keeps = Output(3);
    auto* out_keeps_size = Output(4);
    out_keeps->Resize(total_keep_count);
    out_keeps_size->Resize(num_classes);

    Eigen::Map<EArrXi> cur_out_keeps_size(
        out_keeps_size->mutable_data<int>(), num_classes);

    cur_out_idx = 0;
    Eigen::Map<EArrXi> out_keeps_arr(
        out_keeps->mutable_data<int>(), total_keep_count);
    for (int j = 0; j < num_classes; j++) {
      out_keeps_arr.segment(cur_out_idx, keeps[j].size()) =
          utils::AsEArrXt(keeps[j]);

      cur_out_keeps_size[j] = keeps[j].size();
      cur_out_idx += keeps[j].size();
    }
  }

  return true;
}

namespace {

REGISTER_CPU_OPERATOR(BoxWithNMSLimit, BoxWithNMSLimitOp<CPUContext>);

#ifdef CAFFE2_HAS_MKL_DNN
REGISTER_MKL_OPERATOR(
    BoxWithNMSLimit,
    mkl::MKLFallbackOp<BoxWithNMSLimitOp<CPUContext>>);
#endif // CAFFE2_HAS_MKL_DNN

OPERATOR_SCHEMA(BoxWithNMSLimit)
    .NumInputs(2)
    .NumOutputs(3, 5)
    .SetDoc(R"DOC(
Apply NMS to each class (except background) and limit the number of
returned boxes.
)DOC")
    .Arg("score_thres", "(float) TEST.SCORE_THRES")
    .Arg("nms", "(float) TEST.NMS")
    .Arg("detections_per_im", "(int) TEST.DEECTIONS_PER_IM")
    .Arg("soft_nms_enabled", "(bool) TEST.SOFT_NMS.ENABLED")
    .Arg("soft_nms_method", "(string) TEST.SOFT_NMS.METHOD")
    .Arg("soft_nms_sigma", "(float) TEST.SOFT_NMS.SIGMA")
    .Arg(
        "soft_nms_min_score_thres",
        "(float) Lower bound on updated scores to discard boxes")
    .Input(0, "scores", "Scores, size (count, num_classes)")
    .Input(
        1,
        "boxes",
        "Bounding box for each class, size (count, num_classes * 4)")
    .Output(0, "scores", "Filtered scores, size (n)")
    .Output(1, "boxes", "Filtered boxes, size (n, 4)")
    .Output(2, "classes", "Class id for each filtered score/box, size (n)")
    .Output(3, "keeps", "Optional filtered indices, size (n)")
    .Output(
        4,
        "keeps_size",
        "Optional number of filtered indices per class, size (num_classes)");

SHOULD_NOT_DO_GRADIENT(BoxWithNMSLimit);

} // namespace
} // namespace caffe2
