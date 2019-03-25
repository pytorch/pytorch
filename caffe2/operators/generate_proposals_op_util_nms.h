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

#if defined(CV_MAJOR_VERSION) && (CV_MAJOR_VERSION >= 3)
namespace {

template <class Derived>
cv::RotatedRect bbox_to_rotated_rect(const Eigen::ArrayBase<Derived>& box) {
  CAFFE_ENFORCE_EQ(box.size(), 5);
  // cv::RotatedRect takes angle to mean clockwise rotation, but RRPN bbox
  // representation means counter-clockwise rotation.
  return cv::RotatedRect(
      cv::Point2f(box[0], box[1]), cv::Size2f(box[2], box[3]), -box[4]);
}

// TODO: cvfix_rotatedRectangleIntersection is a replacement function for
// cv::rotatedRectangleIntersection, which has a bug due to float underflow
// When OpenCV version is upgraded to be >= 4.0,
// we can remove this replacement function.
// For anyone interested, here're the PRs on OpenCV:
// https://github.com/opencv/opencv/issues/12221
// https://github.com/opencv/opencv/pull/12222
int cvfix_rotatedRectangleIntersection(
    const cv::RotatedRect& rect1,
    const cv::RotatedRect& rect2,
    cv::OutputArray intersectingRegion) {
  // Used to test if two points are the same
  const float samePointEps = 0.00001f;
  const float EPS = 1e-14;

  cv::Point2f vec1[4], vec2[4];
  cv::Point2f pts1[4], pts2[4];

  std::vector<cv::Point2f> intersection;

  rect1.points(pts1);
  rect2.points(pts2);

  int ret = cv::INTERSECT_FULL;

  // Specical case of rect1 == rect2
  {
    bool same = true;

    for (int i = 0; i < 4; i++) {
      if (fabs(pts1[i].x - pts2[i].x) > samePointEps ||
          (fabs(pts1[i].y - pts2[i].y) > samePointEps)) {
        same = false;
        break;
      }
    }

    if (same) {
      intersection.resize(4);

      for (int i = 0; i < 4; i++) {
        intersection[i] = pts1[i];
      }

      cv::Mat(intersection).copyTo(intersectingRegion);

      return cv::INTERSECT_FULL;
    }
  }

  // Line vector
  // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
  for (int i = 0; i < 4; i++) {
    vec1[i].x = pts1[(i + 1) % 4].x - pts1[i].x;
    vec1[i].y = pts1[(i + 1) % 4].y - pts1[i].y;

    vec2[i].x = pts2[(i + 1) % 4].x - pts2[i].x;
    vec2[i].y = pts2[(i + 1) % 4].y - pts2[i].y;
  }

  // Line test - test all line combos for intersection
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      // Solve for 2x2 Ax=b
      float x21 = pts2[j].x - pts1[i].x;
      float y21 = pts2[j].y - pts1[i].y;

      const auto& l1 = vec1[i];
      const auto& l2 = vec2[j];

      // This takes care of parallel lines
      float det = l2.x * l1.y - l1.x * l2.y;
      if (std::fabs(det) <= EPS) {
        continue;
      }

      float t1 = (l2.x * y21 - l2.y * x21) / det;
      float t2 = (l1.x * y21 - l1.y * x21) / det;

      if (t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f) {
        float xi = pts1[i].x + vec1[i].x * t1;
        float yi = pts1[i].y + vec1[i].y * t1;
        intersection.push_back(cv::Point2f(xi, yi));
      }
    }
  }

  if (!intersection.empty()) {
    ret = cv::INTERSECT_PARTIAL;
  }

  // Check for vertices from rect1 inside rect2
  for (int i = 0; i < 4; i++) {
    // We do a sign test to see which side the point lies.
    // If the point all lie on the same sign for all 4 sides of the rect,
    // then there's an intersection
    int posSign = 0;
    int negSign = 0;

    float x = pts1[i].x;
    float y = pts1[i].y;

    for (int j = 0; j < 4; j++) {
      // line equation: Ax + By + C = 0
      // see which side of the line this point is at

      // float causes underflow!
      // Original version:
      // float A = -vec2[j].y;
      // float B = vec2[j].x;
      // float C = -(A * pts2[j].x + B * pts2[j].y);
      // float s = A * x + B * y + C;

      double A = -vec2[j].y;
      double B = vec2[j].x;
      double C = -(A * pts2[j].x + B * pts2[j].y);
      double s = A * x + B * y + C;

      if (s >= 0) {
        posSign++;
      } else {
        negSign++;
      }
    }

    if (posSign == 4 || negSign == 4) {
      intersection.push_back(pts1[i]);
    }
  }

  // Reverse the check - check for vertices from rect2 inside rect1
  for (int i = 0; i < 4; i++) {
    // We do a sign test to see which side the point lies.
    // If the point all lie on the same sign for all 4 sides of the rect,
    // then there's an intersection
    int posSign = 0;
    int negSign = 0;

    float x = pts2[i].x;
    float y = pts2[i].y;

    for (int j = 0; j < 4; j++) {
      // line equation: Ax + By + C = 0
      // see which side of the line this point is at

      // float causes underflow!
      // Original version:
      // float A = -vec1[j].y;
      // float B = vec1[j].x;
      // float C = -(A * pts1[j].x + B * pts1[j].y);
      // float s = A*x + B*y + C;

      double A = -vec1[j].y;
      double B = vec1[j].x;
      double C = -(A * pts1[j].x + B * pts1[j].y);
      double s = A * x + B * y + C;

      if (s >= 0) {
        posSign++;
      } else {
        negSign++;
      }
    }

    if (posSign == 4 || negSign == 4) {
      intersection.push_back(pts2[i]);
    }
  }

  // Get rid of dupes
  for (int i = 0; i < (int)intersection.size() - 1; i++) {
    for (size_t j = i + 1; j < intersection.size(); j++) {
      float dx = intersection[i].x - intersection[j].x;
      float dy = intersection[i].y - intersection[j].y;
      // can be a really small number, need double here
      double d2 = dx * dx + dy * dy;

      if (d2 < samePointEps * samePointEps) {
        // Found a dupe, remove it
        std::swap(intersection[j], intersection.back());
        intersection.pop_back();
        j--; // restart check
      }
    }
  }

  if (intersection.empty()) {
    return cv::INTERSECT_NONE;
  }

  // If this check fails then it means we're getting dupes
  // CV_Assert(intersection.size() <= 8);

  // At this point, there might still be some edge cases failing the check above
  // However, it doesn't affect the result of polygon area,
  // even if the number of intersections is greater than 8.
  // Therefore, we just print out these cases for now instead of assertion.
  // TODO: These cases should provide good reference for improving the accuracy
  // for intersection computation above (for example, we should use
  // cross-product/dot-product of vectors instead of line equation to
  // judge the relationships between the points and line segments)

  if (intersection.size() > 8) {
    LOG(ERROR) << "Intersection size = " << intersection.size();
    LOG(ERROR) << "Rect 1:";
    for (int i = 0; i < 4; i++) {
      LOG(ERROR) << " (" << pts1[i].x << " ," << pts1[i].y << "),";
    }
    LOG(ERROR) << "Rect 2:";
    for (int i = 0; i < 4; i++) {
      LOG(ERROR) << " (" << pts2[i].x << " ," << pts2[i].y << "),";
    }
    LOG(ERROR) << "Intersections:";
    for (auto& p : intersection) {
      LOG(ERROR) << " (" << p.x << " ," << p.y << "),";
    }
  }

  cv::Mat(intersection).copyTo(intersectingRegion);

  return ret;
}

/**
 * Returns the intersection area of two rotated rectangles.
 */
double rotated_rect_intersection(
    const cv::RotatedRect& rect1,
    const cv::RotatedRect& rect2) {
  std::vector<cv::Point2f> intersectPts, orderedPts;

  // Find points of intersection

  // TODO: cvfix_rotatedRectangleIntersection is a replacement function for
  // cv::rotatedRectangleIntersection, which has a bug due to float underflow
  // When OpenCV version is upgraded to be >= 4.0,
  // we can remove this replacement function and use the following instead:
  // auto ret = cv::rotatedRectangleIntersection(rect1, rect2, intersectPts);
  // For anyone interested, here're the PRs on OpenCV:
  // https://github.com/opencv/opencv/issues/12221
  // https://github.com/opencv/opencv/pull/12222
  auto ret = cvfix_rotatedRectangleIntersection(rect1, rect2, intersectPts);
  if (intersectPts.size() <= 2) {
    return 0.0;
  }

  // If one rectangle is fully enclosed within another, return the area
  // of the smaller one early.
  if (ret == cv::INTERSECT_FULL) {
    return std::min(rect1.size.area(), rect2.size.area());
  }

  // Convex Hull to order the intersection points in clockwise or
  // counter-clockwise order and find the countour area.
  cv::convexHull(intersectPts, orderedPts);
  return cv::contourArea(orderedPts);
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

  std::vector<cv::RotatedRect> rotated_rects(proposals.rows());
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

  std::vector<cv::RotatedRect> rotated_rects(proposals.rows());
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
#endif // CV_MAJOR_VERSION >= 3

template <class Derived1, class Derived2>
std::vector<int> nms_cpu(
    const Eigen::ArrayBase<Derived1>& proposals,
    const Eigen::ArrayBase<Derived2>& scores,
    const std::vector<int>& sorted_indices,
    float thresh,
    int topN = -1) {
#if defined(CV_MAJOR_VERSION) && (CV_MAJOR_VERSION >= 3)
  CAFFE_ENFORCE(proposals.cols() == 4 || proposals.cols() == 5);
  if (proposals.cols() == 4) {
    // Upright boxes
    return nms_cpu_upright(proposals, scores, sorted_indices, thresh, topN);
  } else {
    // Rotated boxes with angle info
    return nms_cpu_rotated(proposals, scores, sorted_indices, thresh, topN);
  }
#else
  return nms_cpu_upright(proposals, scores, sorted_indices, thresh, topN);
#endif // CV_MAJOR_VERSION >= 3
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
    float thres) {
  std::vector<int> indices(proposals.rows());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(
      indices.data(),
      indices.data() + indices.size(),
      [&scores](int lhs, int rhs) { return scores(lhs) > scores(rhs); });

  return nms_cpu(proposals, scores, indices, thres);
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
    int topN = -1) {
#if defined(CV_MAJOR_VERSION) && (CV_MAJOR_VERSION >= 3)
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
        topN);
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
#else
  return soft_nms_cpu_upright(
      out_scores,
      proposals,
      scores,
      indices,
      sigma,
      overlap_thresh,
      score_thresh,
      method,
      topN);
#endif // CV_MAJOR_VERSION >= 3
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
