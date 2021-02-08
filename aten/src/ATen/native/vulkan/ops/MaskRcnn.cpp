// Copyright 2004-present Facebook. All Rights Reserved.

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include "caffe2/utils/eigen_utils.h"
//#include "caffe2/utils/math.h"
#include <ATen/native/vulkan/ops/generate_proposals_op_util_boxes.h>
#include <ATen/native/vulkan/ops/generate_proposals_op_util_nms.h>
#include <ATen/native/vulkan/ops/roi_align_impl-cpu.h>
#include <ATen/native/vulkan/ops/heatmap_max_keypoint.h>
#include <ATen/native/vulkan/ops/bbox_transform.h>
#include <ATen/native/vulkan/ops/box_with_nms_limit.h>

namespace at {
namespace native {
//namespace maskrcnn {

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
    TORCH_CHECK(i <= dims_.size());
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

// Compute the 1-d index of a n-dimensional contiguous row-major tensor for
//     a given n-dimensional index 'index'
size_t ComputeStartIndex(
    const at::Tensor& tensor,
    const std::vector<int>& index) {
  TORCH_CHECK(index.size() == tensor.dim());

  size_t ret = 0;
  for (int i = 0; i < index.size(); i++) {
    ret += index[i] * c10::size_from_dim_(i + 1, tensor.sizes());
  }

  return ret;
}

// Get a sub tensor view from 'tensor' using data pointer from 'tensor'
template <class T>
ConstTensorView<T> GetSubTensorView(
    const at::Tensor& tensor,
    int dim0_start_index) {
  TORCH_CHECK(tensor.dtype().itemsize() == sizeof(T));

  if (tensor.numel() == 0) {
    return ConstTensorView<T>(nullptr, {});
  }

  std::vector<int> start_dims(tensor.dim(), 0);
  start_dims.at(0) = dim0_start_index;
  auto st_idx = ComputeStartIndex(tensor, start_dims);
  auto ptr = tensor.data_ptr<T>() + st_idx;

  auto input_dims = tensor.sizes();
  std::vector<int> ret_dims(input_dims.begin() + 1, input_dims.end());

  ConstTensorView<T> ret(ptr, ret_dims);
  return ret;
}

caffe2::ERMatXf ComputeAllAnchors(
    const at::Tensor& anchors,
    int height,
    int width,
    float feat_stride) {
  const auto K = height * width;
  const auto A = anchors.size(0);
  const auto box_dim = anchors.size(1);
  TORCH_CHECK(box_dim == 4 || box_dim == 5);

  caffe2::ERMatXf shift_x = (caffe2::ERVecXf::LinSpaced(width, 0.0, width - 1.0) * feat_stride)
                        .replicate(height, 1);
  caffe2::ERMatXf shift_y = (caffe2::EVecXf::LinSpaced(height, 0.0, height - 1.0) * feat_stride)
                        .replicate(1, width);
  Eigen::MatrixXf shifts(K, box_dim);
  if (box_dim == 4) {
    // Upright boxes in [x1, y1, x2, y2] format
    shifts << caffe2::ConstEigenVectorMap<float>(shift_x.data(), shift_x.size()),
        caffe2::ConstEigenVectorMap<float>(shift_y.data(), shift_y.size()),
        caffe2::ConstEigenVectorMap<float>(shift_x.data(), shift_x.size()),
        caffe2::ConstEigenVectorMap<float>(shift_y.data(), shift_y.size());
  } else {
    // Rotated boxes in [ctr_x, ctr_y, w, h, angle] format.
    // Zero shift for width, height and angle.
    caffe2::ERMatXf shift_zero = caffe2::ERMatXf::Constant(height, width, 0.0);
    shifts << caffe2::ConstEigenVectorMap<float>(shift_x.data(), shift_x.size()),
        caffe2::ConstEigenVectorMap<float>(shift_y.data(), shift_y.size()),
        caffe2::ConstEigenVectorMap<float>(shift_zero.data(), shift_zero.size()),
        caffe2::ConstEigenVectorMap<float>(shift_zero.data(), shift_zero.size()),
        caffe2::ConstEigenVectorMap<float>(shift_zero.data(), shift_zero.size());
  }

  // Broacast anchors over shifts to enumerate all anchors at all positions
  // in the (H, W) grid:
  //   - add A anchors of shape (1, A, box_dim) to
  //   - K shifts of shape (K, 1, box_dim) to get
  //   - all shifted anchors of shape (K, A, box_dim)
  //   - reshape to (K*A, box_dim) shifted anchors
  caffe2::ConstEigenMatrixMap<float> anchors_vec(
      anchors.data_ptr<float>(), 1, A * box_dim);
  // equivalent to python code
  //  all_anchors = (
  //        self._model.anchors.reshape((1, A, box_dim)) +
  //        shifts.reshape((1, K, box_dim)).transpose((1, 0, 2)))
  //    all_anchors = all_anchors.reshape((K * A, box_dim))
  // all_anchors_vec: (K, A * box_dim)
  caffe2::ERMatXf all_anchors_vec =
      anchors_vec.replicate(K, 1) + shifts.rowwise().replicate(A);

  // use the following to reshape to (K * A, box_dim)
  // Eigen::Map<const caffe2::ERMatXf> all_anchors(
  //            all_anchors_vec.data(), K * A, box_dim);

  return all_anchors_vec;
}

caffe2::ERArrXXf ComputeSortedAnchors(
    const Eigen::Map<const caffe2::ERArrXXf>& anchors,
    int height,
    int width,
    float feat_stride,
    const caffe2::vector<int>& order) {
  const auto box_dim = anchors.cols();
  TORCH_CHECK(box_dim == 4 || box_dim == 5);

  // Order is flattened in (A, H, W) format. Unravel the indices.
  const auto& order_AHW = caffe2::utils::AsEArrXt(order);
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
    const auto& shift_zero = caffe2::EArrXf::Constant(order.size(), 0.0);
    shifts << shift_x, shift_y, shift_zero, shift_zero, shift_zero;
  }

  // Apply shifts to the relevant anchors.
  // Equivalent to python code `all_anchors = self._anchors[order_A] + shifts`
  caffe2::ERArrXXf anchors_sorted;
  caffe2::utils::GetSubArrayRows(anchors, order_A, &anchors_sorted);
  const auto& all_anchors_sorted = anchors_sorted + shifts.array();
  return all_anchors_sorted;
}

//} // namespace maskrcnn

//namespace maskrcnn {
void ProposalsForOneImage(
    const Eigen::Array3f& im_info,
    const Eigen::Map<const caffe2::ERArrXXf>& anchors,
    const ConstTensorView<float>& bbox_deltas_tensor,
    const ConstTensorView<float>& scores_tensor,
    caffe2::ERArrXXf* out_boxes,
    caffe2::EArrXf* out_probs,
    double spatial_scale_,
    int64_t rpn_pre_nms_topN_,
    int64_t post_nms_topN,
    double nms_thresh,
    double min_size,
    bool angle_bound_on_,
    int64_t angle_bound_lo_,
    int64_t angle_bound_hi_,
    double clip_angle_thresh_,
    bool legacy_plus_one_) {
  const double feat_stride_ = 1.0 / spatial_scale_;
  const int box_dim = static_cast<int>(anchors.cols());
  TORCH_CHECK(box_dim == 4 || box_dim == 5);

  TORCH_CHECK(bbox_deltas_tensor.ndim() == 3);
  TORCH_CHECK(bbox_deltas_tensor.dim(0) % box_dim == 0);
  auto A = bbox_deltas_tensor.dim(0) / box_dim;
  auto H = bbox_deltas_tensor.dim(1);
  auto W = bbox_deltas_tensor.dim(2);
  auto K = H * W;
  TORCH_CHECK(A == anchors.rows());

  // scores are (A, H, W) format from conv output.
  // Maintain the same order without transposing (which is slow)
  // and compute anchors accordingly.
  TORCH_CHECK(scores_tensor.ndim() == 3);
  TORCH_CHECK(scores_tensor.dims() == (caffe2::vector<int>{A, H, W}));
  Eigen::Map<const caffe2::EArrXf> scores(scores_tensor.data(), scores_tensor.size());

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

  caffe2::EArrXf scores_sorted;
  caffe2::utils::GetSubArray(scores, caffe2::utils::AsEArrXt(order), &scores_sorted);

  // bbox_deltas are (A * box_dim, H, W) format from conv output.
  // Order them based on scores maintaining the same format without
  // expensive transpose.
  // Note that order corresponds to (A, H * W) in row-major whereas
  // bbox_deltas are in (A, box_dim, H * W) in row-major. Hence, we
  // obtain a sub-view of bbox_deltas for each dim (4 for RPN, 5 for RRPN)
  // in (A, H * W) with an outer stride of box_dim * H * W. Then we apply
  // the ordering and filtering for each dim iteratively.
  caffe2::ERArrXXf bbox_deltas_sorted(order.size(), box_dim);
  caffe2::EArrXf bbox_deltas_per_dim(A * K);
  caffe2::EigenOuterStride stride(box_dim * K);
  for (int j = 0; j < box_dim; ++j) {
    Eigen::Map<caffe2::ERMatXf>(bbox_deltas_per_dim.data(), A, K) =
        Eigen::Map<const caffe2::ERMatXf, 0, caffe2::EigenOuterStride>(
            bbox_deltas_tensor.data() + j * K, A, K, stride);
    for (int i = 0; i < order.size(); ++i) {
      bbox_deltas_sorted(i, j) = bbox_deltas_per_dim[order[i]];
    }
  }

  // Compute anchors specific to the ordered and pre-filtered indices
  // in (A, H, W) format.
  const auto& all_anchors_sorted =
      ComputeSortedAnchors(anchors, H, W, feat_stride_, order);

  // Transform anchors into proposals via bbox transformations
  static const std::vector<float> bbox_weights{1.0, 1.0, 1.0, 1.0};
  auto proposals = caffe2::utils::bbox_transform(
      all_anchors_sorted,
      bbox_deltas_sorted,
      bbox_weights,
      caffe2::utils::BBOX_XFORM_CLIP_DEFAULT,
      legacy_plus_one_,
      angle_bound_on_,
      angle_bound_lo_,
      angle_bound_hi_);

  // 2. clip proposals to image (may result in proposals with zero area
  // that will be removed in the next step)
  proposals = caffe2::utils::clip_boxes(
      proposals, im_info[0], im_info[1], clip_angle_thresh_, legacy_plus_one_);

  // 3. remove predicted boxes with either height or width < min_size
  auto keep =
      caffe2::utils::filter_boxes(proposals, min_size, im_info, legacy_plus_one_);
  TORCH_CHECK(keep.size() <= scores_sorted.size());

  // 6. apply loose nms (e.g. threshold = 0.7)
  // 7. take after_nms_topN (e.g. 300)
  // 8. return the top proposals (-> RoIs top)
  if (post_nms_topN > 0 && post_nms_topN < keep.size()) {
    keep = caffe2::utils::nms_cpu(
        proposals,
        scores_sorted,
        keep,
        nms_thresh,
        post_nms_topN,
        legacy_plus_one_);
  } else {
    keep = caffe2::utils::nms_cpu(
        proposals, scores_sorted, keep, nms_thresh, -1, legacy_plus_one_);
  }

  // Generate outputs
  caffe2::utils::GetSubArrayRows(proposals, caffe2::utils::AsEArrXt(keep), out_boxes);
  caffe2::utils::GetSubArray(scores_sorted, caffe2::utils::AsEArrXt(keep), out_probs);
}

std::tuple<at::Tensor, at::Tensor> GenerateProposalsCPUKernel(
    const at::Tensor& scores_,
    const at::Tensor& bbox_deltas_,
    const at::Tensor& im_info_tensor_,
    const at::Tensor& anchors_tensor_,
    double spatial_scale_,
    int64_t rpn_pre_nms_topN_,
    int64_t post_nms_topN_,
    double nms_thresh_,
    double rpn_min_size_,
    bool angle_bound_on_,
    int64_t angle_bound_lo_,
    int64_t angle_bound_hi_,
    double clip_angle_thresh_,
    bool legacy_plus_one_,
    c10::optional<std::vector<at::Tensor>>
) {
  // ensure it's NCHW format
  const auto scores = scores_.contiguous();
  const auto bbox_deltas = bbox_deltas_.contiguous();
  const auto im_info_tensor = im_info_tensor_.contiguous();
  const auto anchors_tensor = anchors_tensor_.contiguous();

  TORCH_CHECK(scores.dim() == 4 && scores.dim() == 4);
  TORCH_CHECK(scores.dtype() == at::kFloat, scores.dtype().name());
  const auto num_images = scores.size(0);
  const auto A = scores.size(1);
  const auto height = scores.size(2);
  const auto width = scores.size(3);
  const auto box_dim = anchors_tensor.size(1);
  TORCH_CHECK(box_dim == 4 || box_dim == 5);

  // bbox_deltas: (num_images, A * box_dim, H, W)
  TORCH_CHECK(
      bbox_deltas.sizes() == (at::ArrayRef<int64_t>{num_images, box_dim * A, height, width}));

  // im_info_tensor: (num_images, 3), format [height, width, scale; ...]
  TORCH_CHECK(im_info_tensor.sizes() == (caffe2::vector<int64_t>{num_images, 3}));
  TORCH_CHECK(
      im_info_tensor.dtype() == at::kFloat, im_info_tensor.dtype().name());

  // anchors: (A, box_dim)
  TORCH_CHECK(anchors_tensor.sizes() == (caffe2::vector<int64_t>{A, box_dim}));
  TORCH_CHECK(
      anchors_tensor.dtype() == at::kFloat, anchors_tensor.dtype().name());

  Eigen::Map<const caffe2::ERArrXXf> im_info(
      im_info_tensor.data_ptr<float>(),
      im_info_tensor.size(0),
      im_info_tensor.size(1));

  Eigen::Map<const caffe2::ERArrXXf> anchors(
      anchors_tensor.data_ptr<float>(),
      anchors_tensor.size(0),
      anchors_tensor.size(1));

  std::vector<caffe2::ERArrXXf> im_boxes(num_images);
  std::vector<caffe2::EArrXf> im_probs(num_images);
  for (int i = 0; i < num_images; i++) {
    auto cur_im_info = im_info.row(i);
    auto cur_bbox_deltas = GetSubTensorView<float>(bbox_deltas, i);
    auto cur_scores = GetSubTensorView<float>(scores, i);

    caffe2::ERArrXXf& im_i_boxes = im_boxes[i];
    caffe2::EArrXf& im_i_probs = im_probs[i];
    ProposalsForOneImage(
        cur_im_info,
        anchors,
        cur_bbox_deltas,
        cur_scores,
        &im_i_boxes,
        &im_i_probs,
        spatial_scale_,
        rpn_pre_nms_topN_,
        post_nms_topN_,
        nms_thresh_,
        rpn_min_size_,
        angle_bound_on_,
        angle_bound_lo_,
        angle_bound_hi_,
        clip_angle_thresh_,
        legacy_plus_one_);
  }

  int roi_counts = 0;
  for (int i = 0; i < num_images; i++) {
    roi_counts += im_boxes[i].rows();
  }
  const int roi_col_count = box_dim + 1;

  at::Tensor out_rois = at::zeros({roi_counts, roi_col_count}, at::kFloat);
  at::Tensor out_rois_probs = at::zeros({roi_counts}, at::kFloat);
  float* out_rois_ptr = out_rois.data_ptr<float>();
  float* out_rois_probs_ptr = out_rois_probs.data_ptr<float>();

  for (int i = 0; i < num_images; i++) {
    const caffe2::ERArrXXf& im_i_boxes = im_boxes[i];
    const caffe2::EArrXf& im_i_probs = im_probs[i];
    int csz = im_i_boxes.rows();

    // write rois
    Eigen::Map<caffe2::ERArrXXf> cur_rois(out_rois_ptr, csz, roi_col_count);
    cur_rois.col(0).setConstant(i);
    cur_rois.block(0, 1, csz, box_dim) = im_i_boxes;

    // write rois_probs
    Eigen::Map<caffe2::EArrXf>(out_rois_probs_ptr, csz) = im_i_probs;

    out_rois_ptr += csz * roi_col_count;
    out_rois_probs_ptr += csz;
  }

  return std::make_tuple(out_rois, out_rois_probs);
}

at::Tensor ROIAlignForwardCPUKernel(
    const at::Tensor& features,
    const at::Tensor& rois,
    std::string order,
    double spatial_scale,
    int64_t aligned_height,
    int64_t aligned_width,
    int64_t sampling_ratio,
    bool aligned,
    c10::optional<std::vector<at::Tensor>>) {
  TORCH_CHECK(features.dim(), 4);
  TORCH_CHECK(rois.dim(), 2);
  const int64_t roi_cols = rois.size(1);
  TORCH_CHECK(roi_cols == 4 || roi_cols == 5);
  const int64_t N = rois.size(0);
  const int64_t C = features.size(1);
  const int64_t H = features.size(2);
  const int64_t W = features.size(3);
  const int64_t pooled_h = aligned_height;
  const int64_t pooled_w = aligned_width;
	auto features_contig = features.contiguous();
	auto output = at::empty({N, C, pooled_h, pooled_w}, features.options());
	if (N == 0) {
		return output;
	}

	ROIAlignForwardCpuImplWithNCHW(
			N, C, H, W, roi_cols,
			aligned_height, aligned_width, spatial_scale, sampling_ratio, aligned,
			features_contig.data_ptr<float>(), rois.contiguous().data_ptr<float>(), output.data_ptr<float>());

	return output;
}

at::Tensor aten_ROIAlignForwardCPUKernel(
    const at::Tensor& features,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t aligned_height,
    int64_t aligned_width,
    int64_t sampling_ratio,
    bool aligned) {
  std::cout << "XXX roi_align CPU" << std::endl;
  TORCH_CHECK(features.dim(), 4);
  TORCH_CHECK(rois.dim(), 2);
  const int64_t roi_cols = rois.size(1);
  TORCH_CHECK(roi_cols == 4 || roi_cols == 5);
  const int64_t N = rois.size(0);
  const int64_t C = features.size(1);
  const int64_t H = features.size(2);
  const int64_t W = features.size(3);
  const int64_t pooled_h = aligned_height;
  const int64_t pooled_w = aligned_width;
	auto features_contig = features.contiguous();
	auto output = at::empty({N, C, pooled_h, pooled_w}, features.options());
	if (N == 0) {
		return output;
	}

	ROIAlignForwardCpuImplWithNCHW(
			N, C, H, W, roi_cols,
			aligned_height, aligned_width, spatial_scale, sampling_ratio, aligned,
			features_contig.data_ptr<float>(), rois.contiguous().data_ptr<float>(), output.data_ptr<float>());

	return output;
}

//} // namespace maskrcnn
} // namespace native
} // namespace at

/*
TORCH_LIBRARY_FRAGMENT(_caffe2, m) {
	m.def(TORCH_SELECTIVE_SCHEMA(
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
      "bool legacy_plus_one, "
      "Tensor[]? _caffe2_preallocated_outputs=None"
    ") -> (Tensor output_0, Tensor output_1)"
  ));
  m.def(TORCH_SELECTIVE_SCHEMA(
    "_caffe2::RoIAlign("
    "Tensor features,"
    "Tensor rois,"
    "str order,"
    "float spatial_scale,"
    "int pooled_h,"
    "int pooled_w,"
    "int sampling_ratio,"
    "bool aligned, "
    "Tensor[]? _caffe2_preallocated_outputs=None"
    ") -> Tensor"
  ));
  m.def(
      "HeatmapMaxKeypoint("
      "Tensor heatmaps, "
      "Tensor bboxes_in, "
      "bool should_output_softmax = True, "
      "Tensor[]? _caffe2_preallocated_outputs=None"
    ") -> Tensor"
  );
  m.def(
      "BBoxTransform("
        "Tensor rois, "
        "Tensor deltas, "
        "Tensor im_info, "
        "float[] weights, "
        "bool apply_scale, "
        "bool rotated, "
        "bool angle_bound_on, "
        "int angle_bound_lo, "
        "int angle_bound_hi, "
        "float clip_angle_thresh, "
        "bool legacy_plus_one, "
        "Tensor[]? _caffe2_preallocated_outputs=None"
      ") -> ("
        "Tensor output_0, "
        "Tensor output_1"
      ")"
    );
  m.def(
      "BoxWithNMSLimit("
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
        "bool legacy_plus_one, "
        "Tensor[]? _caffe2_preallocated_outputs=None"
      ") -> ("
        "Tensor scores, "
        "Tensor boxes, "
        "Tensor classes, "
        "Tensor batch_splits, "
        "Tensor keeps, "
        "Tensor keeps_size"
      ")"
    );
}

TORCH_LIBRARY_IMPL(_caffe2, Vulkan, m) {
m.impl(
    "_caffe2::GenerateProposals",
    TORCH_FN(at::native::GenerateProposalsCPUKernel));
m.impl(
    "_caffe2::RoIAlign",
    TORCH_FN(at::native::ROIAlignForwardCPUKernel));
m.impl(
    TORCH_SELECTIVE_NAME("_caffe2::HeatmapMaxKeypoint"),
    TORCH_FN(at::native::HeatmapMaxKeypointCPUKernel));
m.impl(
    TORCH_SELECTIVE_NAME("_caffe2::BBoxTransform"),
    TORCH_FN(at::native::BBoxTransformCPUKernel));
m.impl(
    TORCH_SELECTIVE_NAME("_caffe2::BoxWithNMSLimit"),
    TORCH_FN(at::native::BoxWithNMSLimitCPUKernel));
}
*/

