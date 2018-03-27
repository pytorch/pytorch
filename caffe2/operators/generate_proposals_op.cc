#include "caffe2/operators/generate_proposals_op.h"
#include "caffe2/operators/generate_proposals_op_util_boxes.h"
#include "generate_proposals_op_util_nms.h"

#ifdef CAFFE2_USE_MKL
#include "caffe2/mkl/operators/operator_fallback_mkl.h"
#endif // CAFFE2_USE_MKL

namespace caffe2 {

namespace {

// Compute the 1-d index of a n-dimensional contiguous row-major tensor for
//     a given n-dimensional index 'index'
size_t ComputeStartIndex(
    const TensorCPU& tensor,
    const std::vector<int>& index) {
  DCHECK_EQ(index.size(), tensor.ndim());

  size_t ret = 0;
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
  DCHECK_EQ(tensor.meta().itemsize(), sizeof(T));

  if (tensor.size() == 0) {
    return utils::ConstTensorView<T>(nullptr, {});
  }

  std::vector<int> start_dims(tensor.ndim(), 0);
  start_dims.at(0) = dim0_start_index;
  auto st_idx = ComputeStartIndex(tensor, start_dims);
  auto ptr = tensor.data<T>() + st_idx;

  auto& input_dims = tensor.dims();
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
  const auto A = anchors.dim(0);

  ERMatXf shift_x = (ERVecXf::LinSpaced(width, 0.0, width - 1.0) * feat_stride)
                        .replicate(height, 1);
  ERMatXf shift_y = (EVecXf::LinSpaced(height, 0.0, height - 1.0) * feat_stride)
                        .replicate(1, width);
  Eigen::MatrixXf shifts(K, 4);
  shifts << ConstEigenVectorMap<float>(shift_x.data(), shift_x.size()),
      ConstEigenVectorMap<float>(shift_y.data(), shift_y.size()),
      ConstEigenVectorMap<float>(shift_x.data(), shift_x.size()),
      ConstEigenVectorMap<float>(shift_y.data(), shift_y.size());

  // Broacast anchors over shifts to enumerate all anchors at all positions
  // in the (H, W) grid:
  //   - add A anchors of shape (1, A, 4) to
  //   - K shifts of shape (K, 1, 4) to get
  //   - all shifted anchors of shape (K, A, 4)
  //   - reshape to (K*A, 4) shifted anchors
  ConstEigenMatrixMap<float> anchors_vec(
      anchors.template data<float>(), 1, A * 4);
  // equivalent to python code
  //  all_anchors = (
  //        self._model.anchors.reshape((1, A, 4)) +
  //        shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
  //    all_anchors = all_anchors.reshape((K * A, 4))
  // all_anchors_vec: (K, A * 4)
  ERMatXf all_anchors_vec =
      anchors_vec.replicate(K, 1) + shifts.rowwise().replicate(A);

  // use the following to reshape to (K * A, 4)
  // Eigen::Map<const ERMatXf> all_anchors(all_anchors_vec.data(), K * A, 4);

  return all_anchors_vec;
}

} // namespace utils

template <>
void GenerateProposalsOp<CPUContext>::ProposalsForOneImage(
    const Eigen::Array3f& im_info,
    const Eigen::Map<const ERMatXf>& all_anchors,
    const utils::ConstTensorView<float>& bbox_deltas_tensor,
    const utils::ConstTensorView<float>& scores_tensor,
    ERArrXXf* out_boxes,
    EArrXf* out_probs) const {
  const auto& pre_nms_topN = rpn_pre_nms_topN_;
  const auto& post_nms_topN = rpn_post_nms_topN_;
  const auto& nms_thresh = rpn_nms_thresh_;
  const auto& min_size = rpn_min_size_;

  // Transpose and reshape predicted bbox transformations to get them
  // into the same order as the anchors:
  //   - bbox deltas will be (4 * A, H, W) format from conv output
  //   - transpose to (H, W, 4 * A)
  //   - reshape to (H * W * A, 4) where rows are ordered by (H, W, A)
  //     in slowest to fastest order to match the enumerated anchors
  CAFFE_ENFORCE_EQ(bbox_deltas_tensor.ndim(), 3);
  CAFFE_ENFORCE_EQ(bbox_deltas_tensor.dim(0) % 4, 0);
  auto A = bbox_deltas_tensor.dim(0) / 4;
  auto H = bbox_deltas_tensor.dim(1);
  auto W = bbox_deltas_tensor.dim(2);
  // equivalent to python code
  //  bbox_deltas = bbox_deltas.transpose((1, 2, 0)).reshape((-1, 4))
  ERArrXXf bbox_deltas(H * W * A, 4);
  Eigen::Map<ERMatXf>(bbox_deltas.data(), H * W, 4 * A) =
      Eigen::Map<const ERMatXf>(bbox_deltas_tensor.data(), A * 4, H * W)
          .transpose();
  CAFFE_ENFORCE_EQ(bbox_deltas.rows(), all_anchors.rows());

  // - scores are (A, H, W) format from conv output
  // - transpose to (H, W, A)
  // - reshape to (H * W * A, 1) where rows are ordered by (H, W, A)
  //   to match the order of anchors and bbox_deltas
  CAFFE_ENFORCE_EQ(scores_tensor.ndim(), 3);
  CAFFE_ENFORCE_EQ(scores_tensor.dims(), (vector<int>{A, H, W}));
  // equivalent to python code
  // scores = scores.transpose((1, 2, 0)).reshape((-1, 1))
  EArrXf scores(scores_tensor.size());
  Eigen::Map<ERMatXf>(scores.data(), H * W, A) =
      Eigen::Map<const ERMatXf>(scores_tensor.data(), A, H * W).transpose();

  // Transform anchors into proposals via bbox transformations
  static const std::vector<float> bbox_weights{1.0, 1.0, 1.0, 1.0};
  auto proposals = utils::bbox_transform(
      all_anchors.array(),
      bbox_deltas,
      bbox_weights,
      utils::BBOX_XFORM_CLIP_DEFAULT,
      correct_transform_coords_);

  // 2. clip proposals to image (may result in proposals with zero area
  // that will be removed in the next step)
  proposals = utils::clip_boxes(proposals, im_info[0], im_info[1]);

  // 3. remove predicted boxes with either height or width < min_size
  auto keep = utils::filter_boxes(proposals, min_size, im_info);
  DCHECK_LE(keep.size(), scores.size());

  // 4. sort all (proposal, score) pairs by score from highest to lowest
  // 5. take top pre_nms_topN (e.g. 6000)
  std::sort(keep.begin(), keep.end(), [&scores](int lhs, int rhs) {
    return scores[lhs] > scores[rhs];
  });

  if (pre_nms_topN > 0 && pre_nms_topN < keep.size()) {
    keep.resize(pre_nms_topN);
  }

  // 6. apply loose nms (e.g. threshold = 0.7)
  // 7. take after_nms_topN (e.g. 300)
  // 8. return the top proposals (-> RoIs top)
  if (post_nms_topN > 0 && post_nms_topN < keep.size()) {
    keep = utils::nms_cpu(proposals, scores, keep, nms_thresh, post_nms_topN);
  } else {
    keep = utils::nms_cpu(proposals, scores, keep, nms_thresh);
  }

  // Generate outputs
  utils::GetSubArrayRows(proposals, utils::AsEArrXt(keep), out_boxes);
  utils::GetSubArray(scores, utils::AsEArrXt(keep), out_probs);
}

template <>
bool GenerateProposalsOp<CPUContext>::RunOnDevice() {
  const auto& scores = Input(0);
  const auto& bbox_deltas = Input(1);
  const auto& im_info_tensor = Input(2);
  const auto& anchors = Input(3);
  auto* out_rois = Output(0);
  auto* out_rois_probs = Output(1);

  CAFFE_ENFORCE_EQ(scores.ndim(), 4, scores.ndim());
  CAFFE_ENFORCE(scores.template IsType<float>(), scores.meta().name());
  const auto num_images = scores.dim(0);
  const auto A = scores.dim(1);
  const auto height = scores.dim(2);
  const auto width = scores.dim(3);
  const auto K = height * width;

  // bbox_deltas: (num_images, A * 4, H, W)
  CAFFE_ENFORCE_EQ(
      bbox_deltas.dims(), (vector<TIndex>{num_images, 4 * A, height, width}));

  // im_info_tensor: (num_images, 3), format [height, width, scale; ...]
  CAFFE_ENFORCE_EQ(im_info_tensor.dims(), (vector<TIndex>{num_images, 3}));
  CAFFE_ENFORCE(
      im_info_tensor.template IsType<float>(), im_info_tensor.meta().name());

  // anchors: (A, 4)
  CAFFE_ENFORCE_EQ(anchors.dims(), (vector<TIndex>{A, 4}));
  CAFFE_ENFORCE(anchors.template IsType<float>(), anchors.meta().name());

  // Broadcast the anchors to all pixels
  auto all_anchors_vec =
      utils::ComputeAllAnchors(anchors, height, width, feat_stride_);
  Eigen::Map<const ERMatXf> all_anchors(all_anchors_vec.data(), K * A, 4);

  Eigen::Map<const ERArrXXf> im_info(
      im_info_tensor.data<float>(),
      im_info_tensor.dim(0),
      im_info_tensor.dim(1));

  const int roi_col_count = 5;
  out_rois->Resize(0, roi_col_count);
  out_rois_probs->Resize(0);

  // Use openmp for acceleration?
  for (int i = 0; i < num_images; i++) {
    auto cur_im_info = im_info.row(i);
    auto cur_bbox_deltas = GetSubTensorView<float>(bbox_deltas, i);
    auto cur_scores = GetSubTensorView<float>(scores, i);

    ERArrXXf im_i_boxes;
    EArrXf im_i_probs;
    ProposalsForOneImage(
        cur_im_info,
        all_anchors,
        cur_bbox_deltas,
        cur_scores,
        &im_i_boxes,
        &im_i_probs);

    int csz = im_i_boxes.rows();
    int cur_start_idx = out_rois->dim(0);

    out_rois->Extend(csz, 50, &context_);
    out_rois_probs->Extend(csz, 50, &context_);

    // write rois
    Eigen::Map<ERArrXXf> cur_rois(
        out_rois->mutable_data<float>() + cur_start_idx * roi_col_count,
        csz,
        5);
    cur_rois.col(0).setConstant(i);
    cur_rois.block(0, 1, csz, 4) = im_i_boxes;

    // write rois_probs
    Eigen::Map<EArrXf>(
        out_rois_probs->mutable_data<float>() + cur_start_idx, csz) =
        im_i_probs;
  }

  return true;
}

namespace {

REGISTER_CPU_OPERATOR(GenerateProposals, GenerateProposalsOp<CPUContext>);
// For backward compatibility
REGISTER_CPU_OPERATOR(GenerateProposalsCPP, GenerateProposalsOp<CPUContext>);

#ifdef CAFFE2_HAS_MKL_DNN
REGISTER_MKL_OPERATOR(
    GenerateProposals,
    mkl::MKLFallbackOp<GenerateProposalsOp<CPUContext>>);
// For backward compatibility
REGISTER_MKL_OPERATOR(
    GenerateProposalsCPP,
    mkl::MKLFallbackOp<GenerateProposalsOp<CPUContext>>);
#endif // CAFFE2_HAS_MKL_DNN

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
OPERATOR_SCHEMA(GenerateProposalsCPP).NumInputs(4).NumOutputs(2);

SHOULD_NOT_DO_GRADIENT(GenerateProposals);
// For backward compatibility
SHOULD_NOT_DO_GRADIENT(GenerateProposalsCPP);

} // namespace
} // namespace caffe2
