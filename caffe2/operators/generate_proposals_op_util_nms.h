#ifndef CAFFE2_OPERATORS_UTILS_NMS_H_
#define CAFFE2_OPERATORS_UTILS_NMS_H_

#include <vector>

#include "caffe2/core/logging.h"
#include "caffe2/core/macros.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace utils {

// Greedy non-maximum suppression for proposed bounding boxes
// Reject a bounding box if its region has an intersection-overunion (IoU)
//    overlap with a higher scoring selected bounding box larger than a
//    threshold.
// Reference: facebookresearch/Detectron/detectron/utils/cython_nms.pyx
// proposals: pixel coordinates of proposed bounding boxes,
//    size: (M, 4), format: [x1; y1; x2; y2]
// scores: scores for each bounding box, size: (M, 1)
// sorted_indices: indices that sorts the scores from high to low
// return: row indices of the selected proposals
template <class Derived1, class Derived2>
std::vector<int> nms_cpu_upright(
    const Eigen::ArrayBase<Derived1>& proposals,
    const Eigen::ArrayBase<Derived2>& scores,
    const std::vector<int>& sorted_indices,
    float thresh,
    int topN = -1,
    bool legacy_plus_one = false) {
  CAFFE_ENFORCE_EQ(proposals.rows(), scores.rows());
  CAFFE_ENFORCE_EQ(proposals.cols(), 4);
  CAFFE_ENFORCE_EQ(scores.cols(), 1);
  CAFFE_ENFORCE_LE(sorted_indices.size(), proposals.rows());

  using EArrX = EArrXt<typename Derived1::Scalar>;

  auto x1 = proposals.col(0);
  auto y1 = proposals.col(1);
  auto x2 = proposals.col(2);
  auto y2 = proposals.col(3);

  EArrX areas =
      (x2 - x1 + int(legacy_plus_one)) * (y2 - y1 + int(legacy_plus_one));

  EArrXi order = AsEArrXt(sorted_indices);
  std::vector<int> keep;
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

    EArrX w = (xx2 - xx1 + int(legacy_plus_one)).cwiseMax(0.0);
    EArrX h = (yy2 - yy1 + int(legacy_plus_one)).cwiseMax(0.0);
    EArrX inter = w * h;
    EArrX ovr = inter / (areas[i] + GetSubArray(areas, rest_indices) - inter);

    // indices for sub array order[1:n]
    auto inds = GetArrayIndices(ovr <= thresh);
    order = GetSubArray(order, AsEArrXt(inds) + 1);
  }

  return keep;
}

/**
 * Soft-NMS implementation as outlined in https://arxiv.org/abs/1704.04503.
 * Reference: facebookresearch/Detectron/detectron/utils/cython_nms.pyx
 * out_scores: Output updated scores after applying Soft-NMS
 * proposals: pixel coordinates of proposed bounding boxes,
 *    size: (M, 4), format: [x1; y1; x2; y2]
 *    size: (M, 5), format: [ctr_x; ctr_y; w; h; angle (degrees)] for RRPN
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
std::vector<int> soft_nms_cpu_upright(
    Eigen::ArrayBase<Derived3>* out_scores,
    const Eigen::ArrayBase<Derived1>& proposals,
    const Eigen::ArrayBase<Derived2>& scores,
    const std::vector<int>& indices,
    float sigma = 0.5,
    float overlap_thresh = 0.3,
    float score_thresh = 0.001,
    unsigned int method = 1,
    int topN = -1,
    bool legacy_plus_one = false) {
  CAFFE_ENFORCE_EQ(proposals.rows(), scores.rows());
  CAFFE_ENFORCE_EQ(proposals.cols(), 4);
  CAFFE_ENFORCE_EQ(scores.cols(), 1);

  using EArrX = EArrXt<typename Derived1::Scalar>;

  const auto& x1 = proposals.col(0);
  const auto& y1 = proposals.col(1);
  const auto& x2 = proposals.col(2);
  const auto& y2 = proposals.col(3);

  EArrX areas =
      (x2 - x1 + int(legacy_plus_one)) * (y2 - y1 + int(legacy_plus_one));

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

    EArrX w = (xx2 - xx1 + int(legacy_plus_one)).cwiseMax(0.0);
    EArrX h = (yy2 - yy1 + int(legacy_plus_one)).cwiseMax(0.0);
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

namespace {
const int INTERSECT_NONE = 0;
const int INTERSECT_PARTIAL = 1;
const int INTERSECT_FULL = 2;

class RotatedRect {
 public:
  RotatedRect() {}
  RotatedRect(
      const Eigen::Vector2f& p_center,
      const Eigen::Vector2f& p_size,
      float p_angle)
      : center(p_center), size(p_size), angle(p_angle) {}
  void get_vertices(Eigen::Vector2f* pt) const {
    // M_PI / 180. == 0.01745329251
    double _angle = angle * 0.01745329251;
    float b = (float)cos(_angle) * 0.5f;
    float a = (float)sin(_angle) * 0.5f;

    pt[0].x() = center.x() - a * size.y() - b * size.x();
    pt[0].y() = center.y() + b * size.y() - a * size.x();
    pt[1].x() = center.x() + a * size.y() - b * size.x();
    pt[1].y() = center.y() - b * size.y() - a * size.x();
    pt[2] = 2 * center - pt[0];
    pt[3] = 2 * center - pt[1];
  }
  Eigen::Vector2f center;
  Eigen::Vector2f size;
  float angle;
};

template <class Derived>
RotatedRect bbox_to_rotated_rect(const Eigen::ArrayBase<Derived>& box) {
  CAFFE_ENFORCE_EQ(box.size(), 5);
  // cv::RotatedRect takes angle to mean clockwise rotation, but RRPN bbox
  // representation means counter-clockwise rotation.
  return RotatedRect(
      Eigen::Vector2f(box[0], box[1]),
      Eigen::Vector2f(box[2], box[3]),
      -box[4]);
}

// Eigen doesn't seem to support 2d cross product, so we make one here
float cross_2d(const Eigen::Vector2f& A, const Eigen::Vector2f& B) {
  return A.x() * B.y() - B.x() * A.y();
}

// rotated_rect_intersection_pts is a replacement function for
// cv::rotatedRectangleIntersection, which has a bug due to float underflow
// For anyone interested, here're the PRs on OpenCV:
// https://github.com/opencv/opencv/issues/12221
// https://github.com/opencv/opencv/pull/12222
// Note that we do not check if the number of intersections is <= 8 in this case
int rotated_rect_intersection_pts(
    const RotatedRect& rect1,
    const RotatedRect& rect2,
    Eigen::Vector2f* intersections,
    int& num) {
  // Used to test if two points are the same
  const float samePointEps = 0.00001f;
  const float EPS = 1e-14;
  num = 0; // number of intersections

  Eigen::Vector2f vec1[4], vec2[4], pts1[4], pts2[4];

  rect1.get_vertices(pts1);
  rect2.get_vertices(pts2);

  // Specical case of rect1 == rect2
  bool same = true;

  for (int i = 0; i < 4; i++) {
    if (fabs(pts1[i].x() - pts2[i].x()) > samePointEps ||
        (fabs(pts1[i].y() - pts2[i].y()) > samePointEps)) {
      same = false;
      break;
    }
  }

  if (same) {
    for (int i = 0; i < 4; i++) {
      intersections[i] = pts1[i];
    }
    num = 4;
    return INTERSECT_FULL;
  }

  // Line vector
  // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
  for (int i = 0; i < 4; i++) {
    vec1[i] = pts1[(i + 1) % 4] - pts1[i];
    vec2[i] = pts2[(i + 1) % 4] - pts2[i];
  }

  // Line test - test all line combos for intersection
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      // Solve for 2x2 Ax=b

      // This takes care of parallel lines
      float det = cross_2d(vec2[j], vec1[i]);
      if (std::fabs(det) <= EPS) {
        continue;
      }

      auto vec12 = pts2[j] - pts1[i];

      float t1 = cross_2d(vec2[j], vec12) / det;
      float t2 = cross_2d(vec1[i], vec12) / det;

      if (t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f) {
        intersections[num++] = pts1[i] + t1 * vec1[i];
      }
    }
  }

  // Check for vertices from rect1 inside rect2
  {
    const auto& AB = vec2[0];
    const auto& DA = vec2[3];
    auto ABdotAB = AB.squaredNorm();
    auto ADdotAD = DA.squaredNorm();
    for (int i = 0; i < 4; i++) {
      // assume ABCD is the rectangle, and P is the point to be judged
      // P is inside ABCD iff. P's projection on AB lies within AB
      // and P's projection on AD lies within AD

      auto AP = pts1[i] - pts2[0];

      auto APdotAB = AP.dot(AB);
      auto APdotAD = -AP.dot(DA);

      if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) &&
          (APdotAD <= ADdotAD)) {
        intersections[num++] = pts1[i];
      }
    }
  }

  // Reverse the check - check for vertices from rect2 inside rect1
  {
    const auto& AB = vec1[0];
    const auto& DA = vec1[3];
    auto ABdotAB = AB.squaredNorm();
    auto ADdotAD = DA.squaredNorm();
    for (int i = 0; i < 4; i++) {
      auto AP = pts2[i] - pts1[0];

      auto APdotAB = AP.dot(AB);
      auto APdotAD = -AP.dot(DA);

      if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) &&
          (APdotAD <= ADdotAD)) {
        intersections[num++] = pts2[i];
      }
    }
  }

  return num ? INTERSECT_PARTIAL : INTERSECT_NONE;
}

// Compute convex hull using Graham scan algorithm
int convex_hull_graham(
    const Eigen::Vector2f* p,
    const int& num_in,
    Eigen::Vector2f* q,
    bool shift_to_zero = false) {
  CAFFE_ENFORCE(num_in >= 2);
  std::vector<int> order;

  // Step 1:
  // Find point with minimum y
  // if more than 1 points have the same minimum y,
  // pick the one with the mimimum x.
  int t = 0;
  for (int i = 1; i < num_in; i++) {
    if (p[i].y() < p[t].y() || (p[i].y() == p[t].y() && p[i].x() < p[t].x())) {
      t = i;
    }
  }
  auto& s = p[t]; // starting point

  // Step 2:
  // Subtract starting point from every points (for sorting in the next step)
  for (int i = 0; i < num_in; i++) {
    q[i] = p[i] - s;
  }

  // Swap the starting point to position 0
  std::swap(q[0], q[t]);

  // Step 3:
  // Sort point 1 ~ num_in according to their relative cross-product values
  // (essentially sorting according to angles)
  std::sort(
      q + 1,
      q + num_in,
      [](const Eigen::Vector2f& A, const Eigen::Vector2f& B) -> bool {
        float temp = cross_2d(A, B);
        if (fabs(temp) < 1e-6) {
          return A.squaredNorm() < B.squaredNorm();
        } else {
          return temp > 0;
        }
      });

  // Step 4:
  // Make sure there are at least 2 points (that don't overlap with each other)
  // in the stack
  int k; // index of the non-overlapped second point
  for (k = 1; k < num_in; k++) {
    if (q[k].squaredNorm() > 1e-8)
      break;
  }
  if (k == num_in) {
    // We reach the end, which means the convex hull is just one point
    q[0] = p[t];
    return 1;
  }
  q[1] = q[k];
  int m = 2; // 2 elements in the stack
  // Step 5:
  // Finally we can start the scanning process.
  // If we find a non-convex relationship between the 3 points,
  // we pop the previous point from the stack until the stack only has two
  // points, or the 3-point relationship is convex again
  for (int i = k + 1; i < num_in; i++) {
    while (m > 1 && cross_2d(q[i] - q[m - 2], q[m - 1] - q[m - 2]) >= 0) {
      m--;
    }
    q[m++] = q[i];
  }

  // Step 6 (Optional):
  // In general sense we need the original coordinates, so we
  // need to shift the points back (reverting Step 2)
  // But if we're only interested in getting the area/perimeter of the shape
  // We can simply return.
  if (!shift_to_zero) {
    for (int i = 0; i < m; i++)
      q[i] += s;
  }

  return m;
}

double polygon_area(const Eigen::Vector2f* q, const int& m) {
  if (m <= 2)
    return 0;
  double area = 0;
  for (int i = 1; i < m - 1; i++)
    area += fabs(cross_2d(q[i] - q[0], q[i + 1] - q[0]));
  return area / 2.0;
}

/**
 * Returns the intersection area of two rotated rectangles.
 */
double rotated_rect_intersection(
    const RotatedRect& rect1,
    const RotatedRect& rect2) {
  // There are up to 16 intersections returned from
  // rotated_rect_intersection_pts
  Eigen::Vector2f intersectPts[16], orderedPts[16];
  int num = 0; // number of intersections

  // Find points of intersection

  // TODO: rotated_rect_intersection_pts is a replacement function for
  // cv::rotatedRectangleIntersection, which has a bug due to float underflow
  // For anyone interested, here're the PRs on OpenCV:
  // https://github.com/opencv/opencv/issues/12221
  // https://github.com/opencv/opencv/pull/12222
  // Note: it doesn't matter if #intersections is greater than 8 here
  auto ret = rotated_rect_intersection_pts(rect1, rect2, intersectPts, num);
  CAFFE_ENFORCE(num <= 16);
  if (num <= 2)
    return 0.0;

  // If one rectangle is fully enclosed within another, return the area
  // of the smaller one early.
  if (ret == INTERSECT_FULL) {
    return std::min(
        rect1.size.x() * rect1.size.y(), rect2.size.x() * rect2.size.y());
  }

  // Convex Hull to order the intersection points in clockwise or
  // counter-clockwise order and find the countour area.
  int num_convex = convex_hull_graham(intersectPts, num, orderedPts, true);
  return polygon_area(orderedPts, num_convex);
}

} // namespace

/**
 * Find the intersection area of two rotated boxes represented in format
 * [ctr_x, ctr_y, width, height, angle].
 * `angle` represents counter-clockwise rotation in degrees.
 */
template <class Derived1, class Derived2>
double bbox_intersection_rotated(
    const Eigen::ArrayBase<Derived1>& box1,
    const Eigen::ArrayBase<Derived2>& box2) {
  CAFFE_ENFORCE(box1.size() == 5 && box2.size() == 5);
  const auto& rect1 = bbox_to_rotated_rect(box1);
  const auto& rect2 = bbox_to_rotated_rect(box2);
  return rotated_rect_intersection(rect1, rect2);
}

/**
 * Similar to `bbox_overlaps()` in detectron/utils/cython_bbox.pyx,
 * but handles rotated boxes represented in format
 * [ctr_x, ctr_y, width, height, angle].
 * `angle` represents counter-clockwise rotation in degrees.
 */
template <class Derived1, class Derived2>
Eigen::ArrayXXf bbox_overlaps_rotated(
    const Eigen::ArrayBase<Derived1>& boxes,
    const Eigen::ArrayBase<Derived2>& query_boxes) {
  CAFFE_ENFORCE(boxes.cols() == 5 && query_boxes.cols() == 5);

  const auto& boxes_areas = boxes.col(2) * boxes.col(3);
  const auto& query_boxes_areas = query_boxes.col(2) * query_boxes.col(3);

  Eigen::ArrayXXf overlaps(boxes.rows(), query_boxes.rows());
  for (int i = 0; i < boxes.rows(); ++i) {
    for (int j = 0; j < query_boxes.rows(); ++j) {
      auto inter = bbox_intersection_rotated(boxes.row(i), query_boxes.row(j));
      overlaps(i, j) = (inter == 0.0)
          ? 0.0
          : inter / (boxes_areas[i] + query_boxes_areas[j] - inter);
    }
  }
  return overlaps;
}

// Similar to nms_cpu_upright, but handles rotated proposal boxes
// in the format:
//   size (M, 5), format [ctr_x; ctr_y; width; height; angle (in degrees)].
//
// For now, we only consider IoU as the metric for suppression. No angle info
// is used yet.
template <class Derived1, class Derived2>
std::vector<int> nms_cpu_rotated(
    const Eigen::ArrayBase<Derived1>& proposals,
    const Eigen::ArrayBase<Derived2>& scores,
    const std::vector<int>& sorted_indices,
    float thresh,
    int topN = -1) {
  CAFFE_ENFORCE_EQ(proposals.rows(), scores.rows());
  CAFFE_ENFORCE_EQ(proposals.cols(), 5);
  CAFFE_ENFORCE_EQ(scores.cols(), 1);
  CAFFE_ENFORCE_LE(sorted_indices.size(), proposals.rows());

  using EArrX = EArrXt<typename Derived1::Scalar>;

  auto widths = proposals.col(2);
  auto heights = proposals.col(3);
  EArrX areas = widths * heights;

  std::vector<RotatedRect> rotated_rects(proposals.rows());
  for (int i = 0; i < proposals.rows(); ++i) {
    rotated_rects[i] = bbox_to_rotated_rect(proposals.row(i));
  }

  EArrXi order = AsEArrXt(sorted_indices);
  std::vector<int> keep;
  while (order.size() > 0) {
    // exit if already enough proposals
    if (topN >= 0 && keep.size() >= topN) {
      break;
    }

    int i = order[0];
    keep.push_back(i);
    ConstEigenVectorArrayMap<int> rest_indices(
        order.data() + 1, order.size() - 1);

    EArrX inter(rest_indices.size());
    for (int j = 0; j < rest_indices.size(); ++j) {
      inter[j] = rotated_rect_intersection(
          rotated_rects[i], rotated_rects[rest_indices[j]]);
    }
    EArrX ovr = inter / (areas[i] + GetSubArray(areas, rest_indices) - inter);

    // indices for sub array order[1:n].
    // TODO (viswanath): Should angle info be included as well while filtering?
    auto inds = GetArrayIndices(ovr <= thresh);
    order = GetSubArray(order, AsEArrXt(inds) + 1);
  }

  return keep;
}

// Similar to soft_nms_cpu_upright, but handles rotated proposal boxes
// in the format:
//   size (M, 5), format [ctr_x; ctr_y; width; height; angle (in degrees)].
//
// For now, we only consider IoU as the metric for suppression. No angle info
// is used yet.
template <class Derived1, class Derived2, class Derived3>
std::vector<int> soft_nms_cpu_rotated(
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
  CAFFE_ENFORCE_EQ(proposals.cols(), 5);
  CAFFE_ENFORCE_EQ(scores.cols(), 1);

  using EArrX = EArrXt<typename Derived1::Scalar>;

  auto widths = proposals.col(2);
  auto heights = proposals.col(3);
  EArrX areas = widths * heights;

  std::vector<RotatedRect> rotated_rects(proposals.rows());
  for (int i = 0; i < proposals.rows(); ++i) {
    rotated_rects[i] = bbox_to_rotated_rect(proposals.row(i));
  }

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
    EArrX inter(rest_indices.size());
    for (int j = 0; j < rest_indices.size(); ++j) {
      inter[j] = rotated_rect_intersection(
          rotated_rects[i], rotated_rects[rest_indices[j]]);
    }
    EArrX ovr = inter / (areas[i] + GetSubArray(areas, rest_indices) - inter);

    // Update scores based on computed IoU, overlap threshold and NMS method
    // TODO (viswanath): Should angle info be included as well while filtering?
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

template <class Derived1, class Derived2>
std::vector<int> nms_cpu(
    const Eigen::ArrayBase<Derived1>& proposals,
    const Eigen::ArrayBase<Derived2>& scores,
    const std::vector<int>& sorted_indices,
    float thresh,
    int topN = -1,
    bool legacy_plus_one = false) {
  CAFFE_ENFORCE(proposals.cols() == 4 || proposals.cols() == 5);
  if (proposals.cols() == 4) {
    // Upright boxes
    return nms_cpu_upright(
        proposals, scores, sorted_indices, thresh, topN, legacy_plus_one);
  } else {
    // Rotated boxes with angle info
    return nms_cpu_rotated(proposals, scores, sorted_indices, thresh, topN);
  }
}

// Greedy non-maximum suppression for proposed bounding boxes
// Reject a bounding box if its region has an intersection-overunion (IoU)
//    overlap with a higher scoring selected bounding box larger than a
//    threshold.
// Reference: facebookresearch/Detectron/detectron/lib/utils/cython_nms.pyx
// proposals: pixel coordinates of proposed bounding boxes,
//    size: (M, 4), format: [x1; y1; x2; y2]
//    size: (M, 5), format: [ctr_x; ctr_y; w; h; angle (degrees)] for RRPN
// scores: scores for each bounding box, size: (M, 1)
// return: row indices of the selected proposals
template <class Derived1, class Derived2>
std::vector<int> nms_cpu(
    const Eigen::ArrayBase<Derived1>& proposals,
    const Eigen::ArrayBase<Derived2>& scores,
    float thres,
    bool legacy_plus_one = false) {
  std::vector<int> indices(proposals.rows());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(
      indices.data(),
      indices.data() + indices.size(),
      [&scores](int lhs, int rhs) { return scores(lhs) > scores(rhs); });

  return nms_cpu(
      proposals,
      scores,
      indices,
      thres,
      -1 /* topN */,
      legacy_plus_one /* legacy_plus_one */);
}

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
    int topN = -1,
    bool legacy_plus_one = false) {
  CAFFE_ENFORCE(proposals.cols() == 4 || proposals.cols() == 5);
  if (proposals.cols() == 4) {
    // Upright boxes
    return soft_nms_cpu_upright(
        out_scores,
        proposals,
        scores,
        indices,
        sigma,
        overlap_thresh,
        score_thresh,
        method,
        topN,
        legacy_plus_one);
  } else {
    // Rotated boxes with angle info
    return soft_nms_cpu_rotated(
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
    int topN = -1,
    bool legacy_plus_one = false) {
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
      topN,
      legacy_plus_one);
}

} // namespace utils
} // namespace caffe2

#endif // CAFFE2_OPERATORS_UTILS_NMS_H_
