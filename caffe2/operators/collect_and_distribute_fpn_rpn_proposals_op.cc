#include "caffe2/operators/collect_and_distribute_fpn_rpn_proposals_op.h"

namespace caffe2 {

namespace utils {

// Compute the area of an array of boxes.
ERArrXXf BoxesArea(const ERArrXXf& boxes, const bool legacy_plus_one) {
  // equivalent to python code
  //   w = (boxes[:, 2] - boxes[:, 0] + 1)
  //   h = (boxes[:, 3] - boxes[:, 1] + 1)
  //   areas = w * h
  //   assert np.all(areas >= 0), 'Negative areas founds'
  const auto w = boxes.col(2) - boxes.col(0) + int(legacy_plus_one);
  const auto h = boxes.col(3) - boxes.col(1) + int(legacy_plus_one);
  const ERArrXXf areas = w * h;
  CAFFE_ENFORCE((areas >= 0).all(), "Negative areas founds: ", boxes);
  // NOLINTNEXTLINE(performance-no-automatic-move)
  return areas;
}

// Determine which FPN level each RoI in a set of RoIs should map to based
// on the heuristic in the FPN paper.
ERArrXXf MapRoIsToFpnLevels(
    Eigen::Ref<const ERArrXXf> rois,
    const float k_min,
    const float k_max,
    const float s0,
    const float lvl0,
    const bool legacy_plus_one) {
  // Compute level ids
  ERArrXXf s = BoxesArea(rois, legacy_plus_one).sqrt();
  // s0 = cfg.FPN.ROI_CANONICAL_SCALE  # default: 224
  // lvl0 = cfg.FPN.ROI_CANONICAL_LEVEL  # default: 4

  // Eqn.(1) in FPN paper
  // equivalent to python code
  //   target_lvls = np.floor(lvl0 + np.log2(s / s0 + 1e-6))
  //   target_lvls = np.clip(target_lvls, k_min, k_max)
  auto target_lvls = (lvl0 + (s / s0 + 1e-6).log() / log(2)).floor();
  auto target_lvls_clipped = target_lvls.min(k_max).max(k_min);
  return target_lvls_clipped;
}

// Sort RoIs from highest to lowest individual RoI score based on
// values from scores array and limit to n results
void SortAndLimitRoIsByScores(
    Eigen::Ref<const EArrXf> scores,
    int n,
    ERArrXXf& rois) {
  CAFFE_ENFORCE(rois.rows() == scores.size(), "RoIs and scores count mismatch");
  // Create index array with 0, 1, ... N
  std::vector<int> idxs(rois.rows());
  std::iota(idxs.begin(), idxs.end(), 0);
  // Reuse a comparator based on scores and store a copy of RoIs that
  // will be truncated and manipulated below
  auto comp = [&scores](int lhs, int rhs) {
    if (scores(lhs) > scores(rhs)) {
      return true;
    }
    if (scores(lhs) < scores(rhs)) {
      return false;
    }
    // To ensure the sort is stable
    return lhs < rhs;
  };
  ERArrXXf rois_copy = rois;
  // Note that people have found nth_element + sort to be much faster
  // than partial_sort so we use it here
  if (n > 0 && n < rois.rows()) {
    std::nth_element(idxs.begin(), idxs.begin() + n, idxs.end(), comp);
    rois.resize(n, rois.cols());
  } else {
    n = rois.rows();
  }
  std::sort(idxs.begin(), idxs.begin() + n, comp);
  // Update RoIs based on new order
  for (int i = 0; i < n; i++) {
    rois.row(i) = rois_copy.row(idxs[i]);
  }
}

// Updates arr to be indices that would sort the array. Implementation of
// https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
void ArgSort(EArrXi& arr) {
  // Create index array with 0, 1, ... N and sort based on array values
  std::vector<int> idxs(arr.size());
  std::iota(std::begin(idxs), std::end(idxs), 0);
  std::sort(idxs.begin(), idxs.end(), [&arr](int lhs, int rhs) {
    return arr(lhs) < arr(rhs);
  });
  // Update array to match new order
  for (int i = 0; i < arr.size(); i++) {
    arr(i) = idxs[i];
  }
}

// Update out_filtered and out_indices with rows from rois where lvl matches
// value in lvls passed in.
void RowsWhereRoILevelEquals(
    Eigen::Ref<const ERArrXXf> rois,
    const ERArrXXf& lvls,
    const int lvl,
    ERArrXXf* out_filtered,
    EArrXi* out_indices) {
  CAFFE_ENFORCE(out_filtered != nullptr, "Output filtered required");
  CAFFE_ENFORCE(out_indices != nullptr, "Output indices required");
  CAFFE_ENFORCE(rois.rows() == lvls.rows(), "RoIs and lvls count mismatch");
  // Calculate how many rows we need
  int filtered_size = (lvls == lvl).rowwise().any().count();
  // Fill in the rows and indices
  out_filtered->resize(filtered_size, rois.cols());
  out_indices->resize(filtered_size);
  for (int i = 0, filtered_idx = 0; i < rois.rows(); i++) {
    auto lvl_row = lvls.row(i);
    if ((lvl_row == lvl).any()) {
      out_filtered->row(filtered_idx) = rois.row(i);
      (*out_indices)(filtered_idx) = i;
      filtered_idx++;
    }
  }
}

} // namespace utils

template <>
bool CollectAndDistributeFpnRpnProposalsOp<CPUContext>::RunOnDevice() {
  int num_rpn_lvls = rpn_max_level_ - rpn_min_level_ + 1;
  CAFFE_ENFORCE_EQ(InputSize(), 2 * num_rpn_lvls);

  int num_roi_lvls = roi_max_level_ - roi_min_level_ + 1;
  CAFFE_ENFORCE_EQ(OutputSize(), num_roi_lvls + 2);

  // Collect rois and scores in Eigen
  // rois are in [[batch_idx, x0, y0, x1, y2], ...] format
  // Combine predictions across all levels and retain the top scoring
  //
  // equivalent to python code
  //   roi_inputs = inputs[:num_rpn_lvls]
  //   score_inputs = inputs[num_rpn_lvls:]
  //   rois = np.concatenate([blob.data for blob in roi_inputs])
  //   scores = np.concatenate([blob.data for blob in score_inputs]).squeeze()
  int proposal_num = 0;
  for (int i = 0; i < num_rpn_lvls; i++) {
    const auto& roi_in = Input(i);
    proposal_num += roi_in.size(0);
  }
  ERArrXXf rois(proposal_num, 5);
  EArrXf scores(proposal_num);
  int len = 0;
  for (int i = 0; i < num_rpn_lvls; i++) {
    const auto& roi_in = Input(i);
    const int n = roi_in.size(0);

    Eigen::Map<const ERArrXXf> roi(roi_in.data<float>(), n, 5);
    rois.block(len, 0, n, 5) = roi;

    const auto& score_in = Input(num_rpn_lvls + i);
    CAFFE_ENFORCE_EQ(score_in.size(0), n);

    // No need to squeeze, since we are reshaping when converting to Eigen
    // https://docs.scipy.org/doc/numpy/reference/generated/numpy.squeeze.html
    Eigen::Map<const EArrXf> score(score_in.data<float>(), n);
    scores.segment(len, n) = score;

    len += n;
  }

  // Grab only top rpn_post_nms_topN rois
  // equivalent to python code
  //   inds = np.argsort(-scores)[:rpn_post_nms_topN]
  //   rois = rois[inds, :]
  utils::SortAndLimitRoIsByScores(scores, rpn_post_nms_topN_, rois);

  // Distribute
  // equivalent to python code
  //   lvl_min = cfg.FPN.ROI_MIN_LEVEL
  //   lvl_max = cfg.FPN.ROI_MAX_LEVEL
  //   lvls = fpn.map_rois_to_fpn_levels(rois[:, 1:5], lvl_min, lvl_max)
  const int lvl_min = roi_min_level_;
  const int lvl_max = roi_max_level_;
  const int canon_scale = roi_canonical_scale_;
  const int canon_level = roi_canonical_level_;
  auto rois_block = rois.block(0, 1, rois.rows(), 4);
  auto lvls = utils::MapRoIsToFpnLevels(
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      rois_block, lvl_min, lvl_max, canon_scale, canon_level, legacy_plus_one_);

  // equivalent to python code
  //   outputs[0].reshape(rois.shape)
  //   outputs[0].data[...] = rois

  auto* rois_out = Output(0, {rois.rows(), rois.cols()}, at::dtype<float>());
  Eigen::Map<ERArrXXf> rois_out_mat(
      rois_out->template mutable_data<float>(), rois.rows(), rois.cols());
  rois_out_mat = rois;

  // Create new roi blobs for each FPN level
  // (See: modeling.FPN.add_multilevel_roi_blobs which is similar but annoying
  // to generalize to support this particular case.)
  //
  // equivalent to python code
  //   rois_idx_order = np.empty((0, ))
  //   for (output_idx, lvl in enumerate(range(lvl_min, lvl_max + 1)))
  //       idx_lvl = np.where(lvls == lvl)[0]
  //       blob_roi_level = rois[idx_lvl, :]
  //       outputs[output_idx + 1].reshape(blob_roi_level.shape)
  //       outputs[output_idx + 1].data[...] = blob_roi_level
  //       rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
  //   rois_idx_restore = np.argsort(rois_idx_order)
  //   blob_utils.py_op_copy_blob(rois_idx_restore.astype(np.int32),
  //   outputs[-1])
  EArrXi rois_idx_restore;
  for (int i = 0, lvl = lvl_min; i < num_roi_lvls; i++, lvl++) {
    ERArrXXf blob_roi_level;
    EArrXi idx_lvl;
    utils::RowsWhereRoILevelEquals(rois, lvls, lvl, &blob_roi_level, &idx_lvl);

    // Output blob_roi_level

    auto* roi_out = Output(
        i + 1,
        {blob_roi_level.rows(), blob_roi_level.cols()},
        at::dtype<float>());
    Eigen::Map<ERArrXXf> roi_out_mat(
        roi_out->template mutable_data<float>(),
        blob_roi_level.rows(),
        blob_roi_level.cols());
    roi_out_mat = blob_roi_level;

    // Append indices from idx_lvl to rois_idx_restore
    rois_idx_restore.conservativeResize(
        rois_idx_restore.size() + idx_lvl.size());
    rois_idx_restore.tail(idx_lvl.size()) = idx_lvl;
  }
  utils::ArgSort(rois_idx_restore);

  auto* rois_idx_restore_out =
      Output(OutputSize() - 1, {rois_idx_restore.size()}, at::dtype<int>());
  Eigen::Map<EArrXi> rois_idx_restore_out_mat(
      rois_idx_restore_out->template mutable_data<int>(),
      rois_idx_restore.size());
  rois_idx_restore_out_mat = rois_idx_restore;

  return true;
}

template <>
bool CollectRpnProposalsOp<CPUContext>::RunOnDevice() {
  int num_rpn_lvls = rpn_max_level_ - rpn_min_level_ + 1;
  CAFFE_ENFORCE_EQ(InputSize(), 2 * num_rpn_lvls);

  // Collect rois and scores in Eigen
  // rois are in [[batch_idx, x0, y0, x1, y2], ...] format
  // Combine predictions across all levels and retain the top scoring
  //
  // equivalent to python code
  //   roi_inputs = inputs[:num_rpn_lvls]
  //   score_inputs = inputs[num_rpn_lvls:]
  //   rois = np.concatenate([blob.data for blob in roi_inputs])
  //   scores = np.concatenate([blob.data for blob in score_inputs]).squeeze()
  int proposal_num = 0;
  for (int i = 0; i < num_rpn_lvls; i++) {
    const auto& roi_in = Input(i);
    proposal_num += roi_in.size(0);
  }
  ERArrXXf rois(proposal_num, 5);
  EArrXf scores(proposal_num);
  int len = 0;
  for (int i = 0; i < num_rpn_lvls; i++) {
    const auto& roi_in = Input(i);
    const int n = roi_in.size(0);

    Eigen::Map<const ERArrXXf> roi(roi_in.data<float>(), n, 5);
    rois.block(len, 0, n, 5) = roi;

    const auto& score_in = Input(num_rpn_lvls + i);
    CAFFE_ENFORCE_EQ(score_in.size(0), n);

    // No need to squeeze, since we are reshaping when converting to Eigen
    // https://docs.scipy.org/doc/numpy/reference/generated/numpy.squeeze.html
    Eigen::Map<const EArrXf> score(score_in.data<float>(), n);
    scores.segment(len, n) = score;

    len += n;
  }

  // Grab only top rpn_post_nms_topN rois
  // equivalent to python code
  //   inds = np.argsort(-scores)[:rpn_post_nms_topN]
  //   rois = rois[inds, :]
  utils::SortAndLimitRoIsByScores(scores, rpn_post_nms_topN_, rois);

  // equivalent to python code
  //   outputs[0].reshape(rois.shape)
  //   outputs[0].data[...] = rois

  auto* rois_out = Output(0, {rois.rows(), rois.cols()}, at::dtype<float>());
  Eigen::Map<ERArrXXf> rois_out_mat(
      rois_out->template mutable_data<float>(), rois.rows(), rois.cols());
  rois_out_mat = rois;

  return true;
}

template <>
bool DistributeFpnProposalsOp<CPUContext>::RunOnDevice() {
  int num_roi_lvls = roi_max_level_ - roi_min_level_ + 1;
  CAFFE_ENFORCE_EQ(OutputSize(), num_roi_lvls + 1);

  // Load Input(0) to rois
  const auto& rois_in = Input(0);
  const int num_rois = rois_in.size(0);
  const int dim_rois = rois_in.size(1);
  CAFFE_ENFORCE(dim_rois == 4 || dim_rois == 5);
  Eigen::Map<const ERArrXXf> rois_4or5(
      rois_in.data<float>(), num_rois, dim_rois);
  ERArrXXf rois = ERArrXXf::Zero(num_rois, 5);
  rois.rightCols(dim_rois) = rois_4or5;

  // Distribute
  // equivalent to python code
  //   lvl_min = cfg.FPN.ROI_MIN_LEVEL
  //   lvl_max = cfg.FPN.ROI_MAX_LEVEL
  //   lvls = fpn.map_rois_to_fpn_levels(rois[:, 1:5], lvl_min, lvl_max)
  const int lvl_min = roi_min_level_;
  const int lvl_max = roi_max_level_;
  const int canon_scale = roi_canonical_scale_;
  const int canon_level = roi_canonical_level_;
  auto rois_block = rois.block(0, 1, rois.rows(), 4);
  auto lvls = utils::MapRoIsToFpnLevels(
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      rois_block, lvl_min, lvl_max, canon_scale, canon_level, legacy_plus_one_);

  // Create new roi blobs for each FPN level
  // (See: modeling.FPN.add_multilevel_roi_blobs which is similar but annoying
  // to generalize to support this particular case.)
  //
  // equivalent to python code
  //   rois_idx_order = np.empty((0, ))
  //   for (output_idx, lvl in enumerate(range(lvl_min, lvl_max + 1)))
  //       idx_lvl = np.where(lvls == lvl)[0]
  //       blob_roi_level = rois[idx_lvl, :]
  //       outputs[output_idx + 1].reshape(blob_roi_level.shape)
  //       outputs[output_idx + 1].data[...] = blob_roi_level
  //       rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
  //   rois_idx_restore = np.argsort(rois_idx_order)
  //   blob_utils.py_op_copy_blob(rois_idx_restore.astype(np.int32),
  //   outputs[-1])
  EArrXi rois_idx_restore;
  for (int i = 0, lvl = lvl_min; i < num_roi_lvls; i++, lvl++) {
    ERArrXXf blob_roi_level;
    EArrXi idx_lvl;
    utils::RowsWhereRoILevelEquals(rois, lvls, lvl, &blob_roi_level, &idx_lvl);

    // Output blob_roi_level

    auto* roi_out = Output(
        i + 0,
        {blob_roi_level.rows(), blob_roi_level.cols()},
        at::dtype<float>());
    Eigen::Map<ERArrXXf> roi_out_mat(
        roi_out->template mutable_data<float>(),
        blob_roi_level.rows(),
        blob_roi_level.cols());
    roi_out_mat = blob_roi_level;

    // Append indices from idx_lvl to rois_idx_restore
    rois_idx_restore.conservativeResize(
        rois_idx_restore.size() + idx_lvl.size());
    rois_idx_restore.tail(idx_lvl.size()) = idx_lvl;
  }
  utils::ArgSort(rois_idx_restore);

  auto* rois_idx_restore_out =
      Output(OutputSize() - 1, {rois_idx_restore.size()}, at::dtype<int>());
  Eigen::Map<EArrXi> rois_idx_restore_out_mat(
      rois_idx_restore_out->template mutable_data<int>(),
      rois_idx_restore.size());
  rois_idx_restore_out_mat = rois_idx_restore;

  return true;
}

namespace {

REGISTER_CPU_OPERATOR(
    CollectAndDistributeFpnRpnProposals,
    CollectAndDistributeFpnRpnProposalsOp<CPUContext>);
REGISTER_CPU_OPERATOR(CollectRpnProposals, CollectRpnProposalsOp<CPUContext>);
REGISTER_CPU_OPERATOR(
    DistributeFpnProposals,
    DistributeFpnProposalsOp<CPUContext>);

OPERATOR_SCHEMA(CollectAndDistributeFpnRpnProposals)
    .NumInputs(2, INT_MAX)
    .NumOutputs(3, INT_MAX)
    .SetDoc(R"DOC(
Merge RPN proposals generated at multiple FPN levels and then
distribute those proposals to their appropriate FPN levels for Faster RCNN.
An anchor at one FPN level may predict an RoI that will map to another level,
hence the need to redistribute the proposals.

Only inference is supported. To train, please use the original Python
operator in Detectron.

Inputs and outputs are examples only; if min/max levels change,
the number of inputs and outputs, as well as their level numbering,
will change.
)DOC")
    .Arg("roi_canonical_scale", "(int) ROI_CANONICAL_SCALE")
    .Arg("roi_canonical_level", "(int) ROI_CANONICAL_LEVEL")
    .Arg("roi_max_level", "(int) ROI_MAX_LEVEL")
    .Arg("roi_min_level", "(int) ROI_MIN_LEVEL")
    .Arg("rpn_max_level", "(int) RPN_MAX_LEVEL")
    .Arg("rpn_min_level", "(int) RPN_MIN_LEVEL")
    .Arg("rpn_post_nms_topN", "(int) RPN_POST_NMS_TOP_N")
    .Input(
        0,
        "rpn_rois_fpn2",
        "RPN proposals for FPN level 2, "
        "format (image_index, x1, y1, x2, y2). See rpn_rois "
        "documentation from GenerateProposals.")
    .Input(
        1,
        "rpn_rois_fpn3",
        "RPN proposals for FPN level 3, "
        "format (image_index, x1, y1, x2, y2). See rpn_rois "
        "documentation from GenerateProposals.")
    .Input(
        2,
        "rpn_rois_fpn4",
        "RPN proposals for FPN level 4, "
        "format (image_index, x1, y1, x2, y2). See rpn_rois "
        "documentation from GenerateProposals.")
    .Input(
        3,
        "rpn_rois_fpn5",
        "RPN proposals for FPN level 5, "
        "format (image_index, x1, y1, x2, y2). See rpn_rois "
        "documentation from GenerateProposals.")
    .Input(
        4,
        "rpn_rois_fpn6",
        "RPN proposals for FPN level 6, "
        "format (image_index, x1, y1, x2, y2). See rpn_rois "
        "documentation from GenerateProposals.")
    .Input(
        5,
        "rpn_roi_probs_fpn2",
        "RPN objectness probabilities for FPN level 2. "
        "See rpn_roi_probs documentation from GenerateProposals.")
    .Input(
        6,
        "rpn_roi_probs_fpn3",
        "RPN objectness probabilities for FPN level 3. "
        "See rpn_roi_probs documentation from GenerateProposals.")
    .Input(
        7,
        "rpn_roi_probs_fpn4",
        "RPN objectness probabilities for FPN level 4. "
        "See rpn_roi_probs documentation from GenerateProposals.")
    .Input(
        8,
        "rpn_roi_probs_fpn5",
        "RPN objectness probabilities for FPN level 5. "
        "See rpn_roi_probs documentation from GenerateProposals.")
    .Input(
        9,
        "rpn_roi_probs_fpn6",
        "RPN objectness probabilities for FPN level 6. "
        "See rpn_roi_probs documentation from GenerateProposals.")
    .Output(
        0,
        "rois",
        "Top proposals limited to rpn_post_nms_topN total, "
        "format (image_index, x1, y1, x2, y2)")
    .Output(
        1,
        "rois_fpn2",
        "RPN proposals for ROI level 2, "
        "format (image_index, x1, y1, x2, y2)")
    .Output(
        2,
        "rois_fpn3",
        "RPN proposals for ROI level 3, "
        "format (image_index, x1, y1, x2, y2)")
    .Output(
        3,
        "rois_fpn4",
        "RPN proposals for ROI level 4, "
        "format (image_index, x1, y1, x2, y2)")
    .Output(
        4,
        "rois_fpn5",
        "RPN proposals for ROI level 5, "
        "format (image_index, x1, y1, x2, y2)")
    .Output(
        5,
        "rois_idx_restore",
        "Permutation on the concatenation of all "
        "rois_fpni, i=min...max, such that when applied the RPN RoIs are "
        "restored to their original order in the input blobs.");

SHOULD_NOT_DO_GRADIENT(CollectAndDistributeFpnRpnProposals);

OPERATOR_SCHEMA(CollectRpnProposals)
    .NumInputs(2, INT_MAX)
    .NumOutputs(1)
    .SetDoc(R"DOC(
...
)DOC")
    .Arg("rpn_max_level", "(int) RPN_MAX_LEVEL")
    .Arg("rpn_min_level", "(int) RPN_MIN_LEVEL")
    .Arg("rpn_post_nms_topN", "(int) RPN_POST_NMS_TOP_N")
    .Input(
        0,
        "rpn_rois_fpn2",
        "RPN proposals for FPN level 2, "
        "format (image_index, x1, y1, x2, y2). See rpn_rois "
        "documentation from GenerateProposals.")
    .Input(
        1,
        "rpn_rois_fpn3",
        "RPN proposals for FPN level 3, "
        "format (image_index, x1, y1, x2, y2). See rpn_rois "
        "documentation from GenerateProposals.")
    .Input(
        2,
        "rpn_rois_fpn4",
        "RPN proposals for FPN level 4, "
        "format (image_index, x1, y1, x2, y2). See rpn_rois "
        "documentation from GenerateProposals.")
    .Input(
        3,
        "rpn_rois_fpn5",
        "RPN proposals for FPN level 5, "
        "format (image_index, x1, y1, x2, y2). See rpn_rois "
        "documentation from GenerateProposals.")
    .Input(
        4,
        "rpn_rois_fpn6",
        "RPN proposals for FPN level 6, "
        "format (image_index, x1, y1, x2, y2). See rpn_rois "
        "documentation from GenerateProposals.")
    .Input(
        5,
        "rpn_roi_probs_fpn2",
        "RPN objectness probabilities for FPN level 2. "
        "See rpn_roi_probs documentation from GenerateProposals.")
    .Input(
        6,
        "rpn_roi_probs_fpn3",
        "RPN objectness probabilities for FPN level 3. "
        "See rpn_roi_probs documentation from GenerateProposals.")
    .Input(
        7,
        "rpn_roi_probs_fpn4",
        "RPN objectness probabilities for FPN level 4. "
        "See rpn_roi_probs documentation from GenerateProposals.")
    .Input(
        8,
        "rpn_roi_probs_fpn5",
        "RPN objectness probabilities for FPN level 5. "
        "See rpn_roi_probs documentation from GenerateProposals.")
    .Input(
        9,
        "rpn_roi_probs_fpn6",
        "RPN objectness probabilities for FPN level 6. "
        "See rpn_roi_probs documentation from GenerateProposals.")
    .Output(
        0,
        "rois",
        "Top proposals limited to rpn_post_nms_topN total, "
        "format (image_index, x1, y1, x2, y2)");

SHOULD_NOT_DO_GRADIENT(CollectRpnProposals);

OPERATOR_SCHEMA(DistributeFpnProposals)
    .NumInputs(1)
    .NumOutputs(2, INT_MAX)
    .SetDoc(R"DOC(
...
)DOC")
    .Arg("roi_canonical_scale", "(int) ROI_CANONICAL_SCALE")
    .Arg("roi_canonical_level", "(int) ROI_CANONICAL_LEVEL")
    .Arg("roi_max_level", "(int) ROI_MAX_LEVEL")
    .Arg("roi_min_level", "(int) ROI_MIN_LEVEL")
    .Input(
        0,
        "rois",
        "Top proposals limited to rpn_post_nms_topN total, "
        "format (image_index, x1, y1, x2, y2)")
    .Output(
        0,
        "rois_fpn2",
        "RPN proposals for ROI level 2, "
        "format (image_index, x1, y1, x2, y2)")
    .Output(
        1,
        "rois_fpn3",
        "RPN proposals for ROI level 3, "
        "format (image_index, x1, y1, x2, y2)")
    .Output(
        2,
        "rois_fpn4",
        "RPN proposals for ROI level 4, "
        "format (image_index, x1, y1, x2, y2)")
    .Output(
        3,
        "rois_fpn5",
        "RPN proposals for ROI level 5, "
        "format (image_index, x1, y1, x2, y2)")
    .Output(
        4,
        "rois_idx_restore",
        "Permutation on the concatenation of all "
        "rois_fpni, i=min...max, such that when applied the RPN RoIs are "
        "restored to their original order in the input blobs.");

SHOULD_NOT_DO_GRADIENT(DistributeFpnProposals);

} // namespace
} // namespace caffe2

// clang-format off
C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    CollectAndDistributeFpnRpnProposals,
    "_caffe2::CollectAndDistributeFpnRpnProposals("
      "Tensor[] input_list, "
      "int roi_canonical_scale, "
      "int roi_canonical_level, "
      "int roi_max_level, "
      "int roi_min_level, "
      "int rpn_max_level, "
      "int rpn_min_level, "
      "int rpn_post_nms_topN, "
      "bool legacy_plus_one"
    ") -> ("
      "Tensor rois, "
      "Tensor rois_fpn2, "
      "Tensor rois_fpn3, "
      "Tensor rois_fpn4, "
      "Tensor rois_fpn5, "
      "Tensor rois_idx_restore_int32"
    ")",
    caffe2::CollectAndDistributeFpnRpnProposalsOp<caffe2::CPUContext>);

C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    CollectRpnProposals,
    "_caffe2::CollectRpnProposals("
      "Tensor[] input_list, "
      "int rpn_max_level, "
      "int rpn_min_level, "
      "int rpn_post_nms_topN"
    ") -> ("
      "Tensor rois"
    ")",
    caffe2::CollectRpnProposalsOp<caffe2::CPUContext>);

C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    DistributeFpnProposals,
    "_caffe2::DistributeFpnProposals("
      "Tensor rois, "
      "int roi_canonical_scale, "
      "int roi_canonical_level, "
      "int roi_max_level, "
      "int roi_min_level, "
      "bool legacy_plus_one"
    ") -> ("
      "Tensor rois_fpn2, "
      "Tensor rois_fpn3, "
      "Tensor rois_fpn4, "
      "Tensor rois_fpn5, "
      "Tensor rois_idx_restore_int32"
    ")",
    caffe2::DistributeFpnProposalsOp<caffe2::CPUContext>);
// clang-format on
