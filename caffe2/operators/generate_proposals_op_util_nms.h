#ifndef CAFFE2_OPERATORS_UTILS_NMS_H_
#define CAFFE2_OPERATORS_UTILS_NMS_H_

#include <list>
#include <vector>

#include "caffe2/utils/eigen_utils.h"

#include "caffe2/core/logging.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace utils {

// Greedy non-maximum suppression for proposed bounding boxes
// Reject a bounding box if its region has an intersection-overunion (IoU)
//    overlap with a higher scoring selected bounding box larger than a
//    threshold.
// Reference: detectron/lib/utils/cython_nms.pyx
// proposals: pixel coordinates of proposed bounding boxes,
//    size: (M, 4), format: [x1; y1; x2; y2]
// scores: scores for each bounding box, size: (M, 1)
// sorted_indices: indices that sorts the scores from high to low
// return: row indices of the selected proposals
template <class Derived1, class Derived2>
std::vector<int> nms_cpu(
    const Eigen::ArrayBase<Derived1>& proposals,
    const Eigen::ArrayBase<Derived2>& scores,
    const std::vector<int>& sorted_indices,
    float thresh,
    int topN = -1) {
  CAFFE_ENFORCE_EQ(proposals.rows(), scores.rows());
  CAFFE_ENFORCE_EQ(proposals.cols(), 4);
  CAFFE_ENFORCE_EQ(scores.cols(), 1);
  CAFFE_ENFORCE_LE(sorted_indices.size(), proposals.rows());

  using EArrX = EArrXt<typename Derived1::Scalar>;

  auto x1 = proposals.col(0);
  auto y1 = proposals.col(1);
  auto x2 = proposals.col(2);
  auto y2 = proposals.col(3);

  EArrX areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0);

  EArrXi order = AsEArrXt(sorted_indices);
  std::vector<int> keep;
  int ci = 0;
  while (order.size() > 0) {
    // exit if already enough proposals
    if (topN >= 0 && keep.size() >= topN) {
      break;
    }

    int i = order[0];
    keep.push_back(i);
    ConstEigenVectorArrayMap<int> rest_indices(
        order.data() + 1, order.size() - 1);
    EArrX xx1 = GetSubArray(x1, rest_indices).cwiseMax(x1[i]);
    EArrX yy1 = GetSubArray(y1, rest_indices).cwiseMax(y1[i]);
    EArrX xx2 = GetSubArray(x2, rest_indices).cwiseMin(x2[i]);
    EArrX yy2 = GetSubArray(y2, rest_indices).cwiseMin(y2[i]);

    EArrX w = (xx2 - xx1 + 1.0).cwiseMax(0.0);
    EArrX h = (yy2 - yy1 + 1.0).cwiseMax(0.0);
    EArrX inter = w * h;
    EArrX ovr = inter / (areas[i] + GetSubArray(areas, rest_indices) - inter);

    // indices for sub array order[1:n]
    auto inds = GetArrayIndices(ovr <= thresh);
    order = GetSubArray(order, AsEArrXt(inds) + 1);
  }

  return keep;
}

// Greedy non-maximum suppression for proposed bounding boxes
// Reject a bounding box if its region has an intersection-overunion (IoU)
//    overlap with a higher scoring selected bounding box larger than a
//    threshold.
// Reference: detectron/lib/utils/cython_nms.pyx
// proposals: pixel coordinates of proposed bounding boxes,
//    size: (M, 4), format: [x1; y1; x2; y2]
// scores: scores for each bounding box, size: (M, 1)
// return: row indices of the selected proposals
template <class Derived1, class Derived2>
std::vector<int> nms_cpu(
    const Eigen::ArrayBase<Derived1>& proposals,
    const Eigen::ArrayBase<Derived2>& scores,
    float thres) {
  std::vector<int> indices(proposals.rows());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(
      indices.data(),
      indices.data() + indices.size(),
      [&scores](int lhs, int rhs) { return scores(lhs) > scores(rhs); });

  return nms_cpu(proposals, scores, indices, thres);
}

/**
 * Soft-NMS implementation as outlined in https://arxiv.org/abs/1704.04503.
 * Reference: detectron/lib/utils/cython_nms.pyx
 * out_scores: Output updated scores after applying Soft-NMS
 * proposals: pixel coordinates of proposed bounding boxes,
 *    size: (M, 4), format: [x1; y1; x2; y2]
 * scores: scores for each bounding box, size: (M, 1)
 * indices: Indices to consider within proposals and scores. Can be used
 *     to pre-filter proposals/scores based on some threshold.
 * sigma: Standard deviation for Gaussian
 * overlap_thresh: Similar to original NMS
 * score_thresh: If updated score falls below this thresh, discard proposal
 * method: 0 - Hard (original) NMS, 1 - Linear, 2 - Gaussian
 * return: row indices of the selected proposals
 */
template <class Derived1, class Derived2, class Derived3>
std::vector<int> soft_nms_cpu(
    Eigen::ArrayBase<Derived3>* out_scores,
    const Eigen::ArrayBase<Derived1>& proposals,
    const Eigen::ArrayBase<Derived2>& scores,
    const std::vector<int>& indices,
    float sigma = 0.5,
    float overlap_thresh = 0.3,
    float score_thresh = 0.001,
    unsigned int method = 1,
    int topN = -1) {
  CAFFE_ENFORCE_EQ(proposals.rows(), scores.rows());
  CAFFE_ENFORCE_EQ(proposals.cols(), 4);
  CAFFE_ENFORCE_EQ(scores.cols(), 1);

  using EArrX = EArrXt<typename Derived1::Scalar>;

  const auto& x1 = proposals.col(0);
  const auto& y1 = proposals.col(1);
  const auto& x2 = proposals.col(2);
  const auto& y2 = proposals.col(3);

  EArrX areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0);

  // Initialize out_scores with original scores. Will be iteratively updated
  // as Soft-NMS is applied.
  *out_scores = scores;

  std::vector<int> keep;
  EArrXi pending = AsEArrXt(indices);
  while (pending.size() > 0) {
    // Exit if already enough proposals
    if (topN >= 0 && keep.size() >= topN) {
      break;
    }

    // Find proposal with max score among remaining proposals
    int max_pos;
    auto max_score = GetSubArray(*out_scores, pending).maxCoeff(&max_pos);
    int i = pending[max_pos];
    keep.push_back(i);

    // Compute IoU of the remaining boxes with the identified max box
    std::swap(pending(0), pending(max_pos));
    const auto& rest_indices = pending.tail(pending.size() - 1);
    EArrX xx1 = GetSubArray(x1, rest_indices).cwiseMax(x1[i]);
    EArrX yy1 = GetSubArray(y1, rest_indices).cwiseMax(y1[i]);
    EArrX xx2 = GetSubArray(x2, rest_indices).cwiseMin(x2[i]);
    EArrX yy2 = GetSubArray(y2, rest_indices).cwiseMin(y2[i]);

    EArrX w = (xx2 - xx1 + 1.0).cwiseMax(0.0);
    EArrX h = (yy2 - yy1 + 1.0).cwiseMax(0.0);
    EArrX inter = w * h;
    EArrX ovr = inter / (areas[i] + GetSubArray(areas, rest_indices) - inter);

    // Update scores based on computed IoU, overlap threshold and NMS method
    for (int j = 0; j < rest_indices.size(); ++j) {
      typename Derived2::Scalar weight;
      switch (method) {
        case 1: // Linear
          weight = (ovr(j) > overlap_thresh) ? (1.0 - ovr(j)) : 1.0;
          break;
        case 2: // Gaussian
          weight = std::exp(-1.0 * ovr(j) * ovr(j) / sigma);
          break;
        default: // Original NMS
          weight = (ovr(j) > overlap_thresh) ? 0.0 : 1.0;
      }
      (*out_scores)(rest_indices[j]) *= weight;
    }

    // Discard boxes with new scores below min threshold and update pending
    // indices
    const auto& rest_scores = GetSubArray(*out_scores, rest_indices);
    const auto& inds = GetArrayIndices(rest_scores >= score_thresh);
    pending = GetSubArray(rest_indices, AsEArrXt(inds));
  }

  return keep;
}

template <class Derived1, class Derived2, class Derived3>
std::vector<int> soft_nms_cpu(
    Eigen::ArrayBase<Derived3>* out_scores,
    const Eigen::ArrayBase<Derived1>& proposals,
    const Eigen::ArrayBase<Derived2>& scores,
    float sigma = 0.5,
    float overlap_thresh = 0.3,
    float score_thresh = 0.001,
    unsigned int method = 1,
    int topN = -1) {
  std::vector<int> indices(proposals.rows());
  std::iota(indices.begin(), indices.end(), 0);
  return soft_nms_cpu(
      out_scores,
      proposals,
      scores,
      indices,
      sigma,
      overlap_thresh,
      score_thresh,
      method,
      topN);
}

} // namespace utils
} // namespace caffe2

#endif // CAFFE2_OPERATORS_UTILS_NMS_H_
