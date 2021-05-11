#include "caffe2/operators/generate_proposals_op.h"
#include "caffe2/operators/generate_proposals_op_util_boxes.h"
#include "generate_proposals_op_util_nms.h"

namespace caffe2 {

namespace {

// Compute the 1-d index of a n-dimensional contiguous row-major tensor for
//     a given n-dimensional index 'index'
size_t ComputeStartIndex(
    const TensorCPU& tensor,
    const std::vector<int>& index) {
  DCHECK_EQ(index.size(), tensor.dim());

  size_t ret = 0;
  // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
  for (int i = 0; i < index.size(); i++) {
    ret += index[i] * tensor.size_from_dim(i + 1);
  }

  return ret;
}

// Get a sub tensor view from 'tensor' using data pointer from 'tensor'
template <class T>
utils::ConstTensorView<T> GetSubTensorView(
    const TensorCPU& tensor,
    int dim0_start_index) {
  DCHECK_EQ(tensor.dtype().itemsize(), sizeof(T));

  if (tensor.numel() == 0) {
    return utils::ConstTensorView<T>(nullptr, {});
  }

  std::vector<int> start_dims(tensor.dim(), 0);
  start_dims.at(0) = dim0_start_index;
  auto st_idx = ComputeStartIndex(tensor, start_dims);
  auto ptr = tensor.data<T>() + st_idx;

  auto input_dims = tensor.sizes();
  std::vector<int> ret_dims(input_dims.begin() + 1, input_dims.end());

  utils::ConstTensorView<T> ret(ptr, ret_dims);
  return ret;
}

} // namespace

namespace utils {

ERMatXf ComputeAllAnchors(
    const TensorCPU& anchors,
    int height,
    int width,
    float feat_stride) {
  const auto K = height * width;
  const auto A = anchors.size(0);
  const auto box_dim = anchors.size(1);
  CAFFE_ENFORCE(box_dim == 4 || box_dim == 5);

  ERMatXf shift_x = (ERVecXf::LinSpaced(width, 0.0, width - 1.0) * feat_stride)
                        .replicate(height, 1);
  ERMatXf shift_y = (EVecXf::LinSpaced(height, 0.0, height - 1.0) * feat_stride)
                        .replicate(1, width);
  Eigen::MatrixXf shifts(K, box_dim);
  if (box_dim == 4) {
    // Upright boxes in [x1, y1, x2, y2] format
    shifts << ConstEigenVectorMap<float>(shift_x.data(), shift_x.size()),
        ConstEigenVectorMap<float>(shift_y.data(), shift_y.size()),
        ConstEigenVectorMap<float>(shift_x.data(), shift_x.size()),
        ConstEigenVectorMap<float>(shift_y.data(), shift_y.size());
  } else {
    // Rotated boxes in [ctr_x, ctr_y, w, h, angle] format.
    // Zero shift for width, height and angle.
    ERMatXf shift_zero = ERMatXf::Constant(height, width, 0.0);
    shifts << ConstEigenVectorMap<float>(shift_x.data(), shift_x.size()),
        ConstEigenVectorMap<float>(shift_y.data(), shift_y.size()),
        ConstEigenVectorMap<float>(shift_zero.data(), shift_zero.size()),
        ConstEigenVectorMap<float>(shift_zero.data(), shift_zero.size()),
        ConstEigenVectorMap<float>(shift_zero.data(), shift_zero.size());
  }

  // Broacast anchors over shifts to enumerate all anchors at all positions
  // in the (H, W) grid:
  //   - add A anchors of shape (1, A, box_dim) to
  //   - K shifts of shape (K, 1, box_dim) to get
  //   - all shifted anchors of shape (K, A, box_dim)
  //   - reshape to (K*A, box_dim) shifted anchors
  ConstEigenMatrixMap<float> anchors_vec(
      anchors.template data<float>(), 1, A * box_dim);
  // equivalent to python code
  //  all_anchors = (
  //        self._model.anchors.reshape((1, A, box_dim)) +
  //        shifts.reshape((1, K, box_dim)).transpose((1, 0, 2)))
  //    all_anchors = all_anchors.reshape((K * A, box_dim))
  // all_anchors_vec: (K, A * box_dim)
  ERMatXf all_anchors_vec =
      anchors_vec.replicate(K, 1) + shifts.rowwise().replicate(A);

  // use the following to reshape to (K * A, box_dim)
  // Eigen::Map<const ERMatXf> all_anchors(
  //            all_anchors_vec.data(), K * A, box_dim);

  return all_anchors_vec;
}

ERArrXXf ComputeSortedAnchors(
    const Eigen::Map<const ERArrXXf>& anchors,
    int height,
    int width,
    float feat_stride,
    const vector<int>& order) {
  const auto box_dim = anchors.cols();
  CAFFE_ENFORCE(box_dim == 4 || box_dim == 5);

  // Order is flattened in (A, H, W) format. Unravel the indices.
  const auto& order_AHW = utils::AsEArrXt(order);
  const auto& order_AH = order_AHW / width;
  const auto& order_W = order_AHW - order_AH * width;
  const auto& order_A = order_AH / height;
  const auto& order_H = order_AH - order_A * height;

  // Generate shifts for each location in the H * W grid corresponding
  // to the sorted scores in (A, H, W) order.
  const auto& shift_x = order_W.cast<float>() * feat_stride;
  const auto& shift_y = order_H.cast<float>() * feat_stride;
  Eigen::MatrixXf shifts(order.size(), box_dim);
  if (box_dim == 4) {
    // Upright boxes in [x1, y1, x2, y2] format
    shifts << shift_x, shift_y, shift_x, shift_y;
  } else {
    // Rotated boxes in [ctr_x, ctr_y, w, h, angle] format.
    // Zero shift for width, height and angle.
    const auto& shift_zero = EArrXf::Constant(order.size(), 0.0);
    shifts << shift_x, shift_y, shift_zero, shift_zero, shift_zero;
  }

  // Apply shifts to the relevant anchors.
  // Equivalent to python code `all_anchors = self._anchors[order_A] + shifts`
  ERArrXXf anchors_sorted;
  utils::GetSubArrayRows(anchors, order_A, &anchors_sorted);
  const auto& all_anchors_sorted = anchors_sorted + shifts.array();
  return all_anchors_sorted;
}

} // namespace utils

template <>
void GenerateProposalsOp<CPUContext>::ProposalsForOneImage(
    const Eigen::Array3f& im_info,
    const Eigen::Map<const ERArrXXf>& anchors,
    const utils::ConstTensorView<float>& bbox_deltas_tensor,
    const utils::ConstTensorView<float>& scores_tensor,
    ERArrXXf* out_boxes,
    EArrXf* out_probs) const {
  const auto& post_nms_topN = rpn_post_nms_topN_;
  const auto& nms_thresh = rpn_nms_thresh_;
  const auto& min_size = rpn_min_size_;
  const int box_dim = static_cast<int>(anchors.cols());
  CAFFE_ENFORCE(box_dim == 4 || box_dim == 5);

  CAFFE_ENFORCE_EQ(bbox_deltas_tensor.ndim(), 3);
  CAFFE_ENFORCE_EQ(bbox_deltas_tensor.dim(0) % box_dim, 0);
  auto A = bbox_deltas_tensor.dim(0) / box_dim;
  auto H = bbox_deltas_tensor.dim(1);
  auto W = bbox_deltas_tensor.dim(2);
  auto K = H * W;
  CAFFE_ENFORCE_EQ(A, anchors.rows());

  // scores are (A, H, W) format from conv output.
  // Maintain the same order without transposing (which is slow)
  // and compute anchors accordingly.
  CAFFE_ENFORCE_EQ(scores_tensor.ndim(), 3);
  CAFFE_ENFORCE_EQ(scores_tensor.dims(), (vector<int>{A, H, W}));
  Eigen::Map<const EArrXf> scores(scores_tensor.data(), scores_tensor.size());

  std::vector<int> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  if (rpn_pre_nms_topN_ <= 0 || rpn_pre_nms_topN_ >= scores.size()) {
    // 4. sort all (proposal, score) pairs by score from highest to lowest
    // 5. take top pre_nms_topN (e.g. 6000)
    std::sort(order.begin(), order.end(), [&scores](int lhs, int rhs) {
      return scores[lhs] > scores[rhs];
    });
  } else {
    // Avoid sorting possibly large arrays; First partition to get top K
    // unsorted and then sort just those (~20x faster for 200k scores)
    std::partial_sort(
        order.begin(),
        order.begin() + rpn_pre_nms_topN_,
        order.end(),
        [&scores](int lhs, int rhs) { return scores[lhs] > scores[rhs]; });
    order.resize(rpn_pre_nms_topN_);
  }

  EArrXf scores_sorted;
  utils::GetSubArray(scores, utils::AsEArrXt(order), &scores_sorted);

  // bbox_deltas are (A * box_dim, H, W) format from conv output.
  // Order them based on scores maintaining the same format without
  // expensive transpose.
  // Note that order corresponds to (A, H * W) in row-major whereas
  // bbox_deltas are in (A, box_dim, H * W) in row-major. Hence, we
  // obtain a sub-view of bbox_deltas for each dim (4 for RPN, 5 for RRPN)
  // in (A, H * W) with an outer stride of box_dim * H * W. Then we apply
  // the ordering and filtering for each dim iteratively.
  ERArrXXf bbox_deltas_sorted(order.size(), box_dim);
  EArrXf bbox_deltas_per_dim(A * K);
  EigenOuterStride stride(box_dim * K);
  for (int j = 0; j < box_dim; ++j) {
    Eigen::Map<ERMatXf>(bbox_deltas_per_dim.data(), A, K) =
        Eigen::Map<const ERMatXf, 0, EigenOuterStride>(
            bbox_deltas_tensor.data() + j * K, A, K, stride);
    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    for (int i = 0; i < order.size(); ++i) {
      bbox_deltas_sorted(i, j) = bbox_deltas_per_dim[order[i]];
    }
  }

  // Compute anchors specific to the ordered and pre-filtered indices
  // in (A, H, W) format.
  const auto& all_anchors_sorted =
      utils::ComputeSortedAnchors(anchors, H, W, feat_stride_, order);

  // Transform anchors into proposals via bbox transformations
  static const std::vector<float> bbox_weights{1.0, 1.0, 1.0, 1.0};
  auto proposals = utils::bbox_transform(
      all_anchors_sorted,
      bbox_deltas_sorted,
      bbox_weights,
      utils::BBOX_XFORM_CLIP_DEFAULT,
      legacy_plus_one_,
      angle_bound_on_,
      angle_bound_lo_,
      angle_bound_hi_);

  // 2. clip proposals to image (may result in proposals with zero area
  // that will be removed in the next step)
  proposals = utils::clip_boxes(
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      proposals, im_info[0], im_info[1], clip_angle_thresh_, legacy_plus_one_);

  // 3. remove predicted boxes with either height or width < min_size
  auto keep =
      utils::filter_boxes(proposals, min_size, im_info, legacy_plus_one_);
  DCHECK_LE(keep.size(), scores_sorted.size());

  // 6. apply loose nms (e.g. threshold = 0.7)
  // 7. take after_nms_topN (e.g. 300)
  // 8. return the top proposals (-> RoIs top)
  // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
  if (post_nms_topN > 0 && post_nms_topN < keep.size()) {
    keep = utils::nms_cpu(
        proposals,
        scores_sorted,
        keep,
        nms_thresh,
        post_nms_topN,
        legacy_plus_one_);
  } else {
    keep = utils::nms_cpu(
        proposals, scores_sorted, keep, nms_thresh, -1, legacy_plus_one_);
  }

  // Generate outputs
  utils::GetSubArrayRows(proposals, utils::AsEArrXt(keep), out_boxes);
  utils::GetSubArray(scores_sorted, utils::AsEArrXt(keep), out_probs);
}

template <>
bool GenerateProposalsOp<CPUContext>::RunOnDevice() {
  const auto& scores = Input(0);
  const auto& bbox_deltas = Input(1);
  const auto& im_info_tensor = Input(2);
  const auto& anchors_tensor = Input(3);

  CAFFE_ENFORCE_EQ(scores.dim(), 4, scores.dim());
  CAFFE_ENFORCE(scores.template IsType<float>(), scores.dtype().name());
  const auto num_images = scores.size(0);
  const auto A = scores.size(1);
  const auto height = scores.size(2);
  const auto width = scores.size(3);
  const auto box_dim = anchors_tensor.size(1);
  CAFFE_ENFORCE(box_dim == 4 || box_dim == 5);

  // bbox_deltas: (num_images, A * box_dim, H, W)
  CAFFE_ENFORCE_EQ(
      bbox_deltas.sizes(),
      (at::ArrayRef<int64_t>{num_images, box_dim * A, height, width}));

  // im_info_tensor: (num_images, 3), format [height, width, scale; ...]
  CAFFE_ENFORCE_EQ(im_info_tensor.sizes(), (vector<int64_t>{num_images, 3}));
  CAFFE_ENFORCE(
      im_info_tensor.template IsType<float>(), im_info_tensor.dtype().name());

  // anchors: (A, box_dim)
  CAFFE_ENFORCE_EQ(anchors_tensor.sizes(), (vector<int64_t>{A, box_dim}));
  CAFFE_ENFORCE(
      anchors_tensor.template IsType<float>(), anchors_tensor.dtype().name());

  Eigen::Map<const ERArrXXf> im_info(
      im_info_tensor.data<float>(),
      im_info_tensor.size(0),
      im_info_tensor.size(1));

  Eigen::Map<const ERArrXXf> anchors(
      anchors_tensor.data<float>(),
      anchors_tensor.size(0),
      anchors_tensor.size(1));

  std::vector<ERArrXXf> im_boxes(num_images);
  std::vector<EArrXf> im_probs(num_images);
  for (int i = 0; i < num_images; i++) {
    auto cur_im_info = im_info.row(i);
    auto cur_bbox_deltas = GetSubTensorView<float>(bbox_deltas, i);
    auto cur_scores = GetSubTensorView<float>(scores, i);

    ERArrXXf& im_i_boxes = im_boxes[i];
    EArrXf& im_i_probs = im_probs[i];
    ProposalsForOneImage(
        cur_im_info,
        anchors,
        cur_bbox_deltas,
        cur_scores,
        &im_i_boxes,
        &im_i_probs);
  }

  int roi_counts = 0;
  for (int i = 0; i < num_images; i++) {
    roi_counts += im_boxes[i].rows();
  }
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  const int roi_col_count = box_dim + 1;
  auto* out_rois = Output(0, {roi_counts, roi_col_count}, at::dtype<float>());
  auto* out_rois_probs = Output(1, {roi_counts}, at::dtype<float>());
  float* out_rois_ptr = out_rois->template mutable_data<float>();
  float* out_rois_probs_ptr = out_rois_probs->template mutable_data<float>();
  for (int i = 0; i < num_images; i++) {
    const ERArrXXf& im_i_boxes = im_boxes[i];
    const EArrXf& im_i_probs = im_probs[i];
    int csz = im_i_boxes.rows();

    // write rois
    Eigen::Map<ERArrXXf> cur_rois(out_rois_ptr, csz, roi_col_count);
    cur_rois.col(0).setConstant(i);
    cur_rois.block(0, 1, csz, box_dim) = im_i_boxes;

    // write rois_probs
    Eigen::Map<EArrXf>(out_rois_probs_ptr, csz) = im_i_probs;

    out_rois_ptr += csz * roi_col_count;
    out_rois_probs_ptr += csz;
  }

  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(GenerateProposals, GenerateProposalsOp<CPUContext>);
// For backward compatibility
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(GenerateProposalsCPP, GenerateProposalsOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(GenerateProposals)
    .NumInputs(4)
    .NumOutputs(2)
    .SetDoc(R"DOC(
Generate bounding box proposals for Faster RCNN. The propoasls are generated for
a list of images based on image score 'score', bounding box regression result
'deltas' as well as predefined bounding box shapes 'anchors'. Greedy
non-maximum suppression is applied to generate the final bounding boxes.
)DOC")
    .Arg("spatial_scale", "(float) spatial scale")
    .Arg("pre_nms_topN", "(int) RPN_PRE_NMS_TOP_N")
    .Arg("post_nms_topN", "(int) RPN_POST_NMS_TOP_N")
    .Arg("nms_thresh", "(float) RPN_NMS_THRESH")
    .Arg("min_size", "(float) RPN_MIN_SIZE")
    .Arg(
        "angle_bound_on",
        "bool (default true). If set, for rotated boxes, angle is "
        "normalized to be within [angle_bound_lo, angle_bound_hi].")
    .Arg(
        "angle_bound_lo",
        "int (default -90 degrees). If set, for rotated boxes, angle is "
        "normalized to be within [angle_bound_lo, angle_bound_hi].")
    .Arg(
        "angle_bound_hi",
        "int (default 90 degrees). If set, for rotated boxes, angle is "
        "normalized to be within [angle_bound_lo, angle_bound_hi].")
    .Arg(
        "clip_angle_thresh",
        "float (default 1.0 degrees). For RRPN, clip almost horizontal boxes "
        "within this threshold of tolerance for backward compatibility. "
        "Set to negative value for no clipping.")
    .Input(0, "scores", "Scores from conv layer, size (img_count, A, H, W)")
    .Input(
        1,
        "bbox_deltas",
        "Bounding box deltas from conv layer, "
        "size (img_count, 4 * A, H, W)")
    .Input(
        2,
        "im_info",
        "Image info, size (img_count, 3), "
        "format (height, width, scale)")
    .Input(3, "anchors", "Bounding box anchors, size (A, 4)")
    .Output(
        0,
        "rois",
        "Proposals, size (n x 5), "
        "format (image_index, x1, y1, x2, y2)")
    .Output(1, "rois_probs", "scores of proposals, size (n)");
// For backward compatibility
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(GenerateProposalsCPP).NumInputs(4).NumOutputs(2);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(GenerateProposals);
// For backward compatibility
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(GenerateProposalsCPP);

} // namespace caffe2

// clang-format off
C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    GenerateProposals,
    "_caffe2::GenerateProposals("
      "Tensor scores, "
      "Tensor bbox_deltas, "
      "Tensor im_info, "
      "Tensor anchors, "
      "float spatial_scale, "
      "int pre_nms_topN, "
      "int post_nms_topN, "
      "float nms_thresh, "
      "float min_size, "
      "bool angle_bound_on, "
      "int angle_bound_lo, "
      "int angle_bound_hi, "
      "float clip_angle_thresh, "
      "bool legacy_plus_one"
    ") -> (Tensor output_0, Tensor output_1)",
    caffe2::GenerateProposalsOp<caffe2::CPUContext>);
// clang-format on
