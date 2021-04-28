#include "caffe2/operators/generate_proposals_op_util_boxes.h"
#include "caffe2/utils/eigen_utils.h"

#include <gtest/gtest.h>

namespace caffe2 {

TEST(UtilsBoxesTest, TestBboxTransformRandom) {
  using EMatXf = Eigen::MatrixXf;

  EMatXf bbox(5, 4);
  bbox << 175.62031555, 20.91103172, 253.352005, 155.0145874, 169.24636841,
      4.85241556, 228.8605957, 105.02092743, 181.77426147, 199.82876587,
      192.88427734, 214.0255127, 174.36262512, 186.75761414, 296.19091797,
      231.27906799, 22.73153877, 92.02596283, 135.5695343, 208.80291748;

  EMatXf deltas(5, 4);
  deltas << 0.47861834, 0.13992102, 0.14961673, 0.71495209, 0.29915856,
      -0.35664671, 0.89018666, 0.70815367, -0.03852064, 0.44466892, 0.49492538,
      0.71409376, 0.28052918, 0.02184832, 0.65289006, 1.05060139, -0.38172557,
      -0.08533806, -0.60335309, 0.79052375;

  EMatXf result_gt(5, 4);
  result_gt << 206.949539, -30.715202, 297.387665, 244.448486, 143.871216,
      -83.342888, 290.502289, 121.053398, 177.430283, 198.666245, 196.295273,
      228.703079, 152.251892, 145.431564, 387.215454, 274.594238, 5.062420,
      11.040955, 66.328903, 269.686218;

  const float BBOX_XFORM_CLIP = log(1000.0 / 16.0);
  auto result = utils::bbox_transform(
      bbox.array(),
      deltas.array(),
      std::vector<float>{1.0, 1.0, 1.0, 1.0},
      BBOX_XFORM_CLIP,
      true /* legacy_plus_one */);
  EXPECT_NEAR((result.matrix() - result_gt).norm(), 0.0, 1e-4);
}

TEST(UtilsBoxesTest, TestBboxTransformRotated) {
  // Test rotated bbox transform w/o angle normalization
  using EMatXf = Eigen::MatrixXf;

  EMatXf bbox(5, 5);
  bbox << 214.986, 88.4628, 78.7317, 135.104, 0.0, 199.553, 55.4367, 60.6142,
      101.169, 45.0, 187.829, 207.427, 0012.11, 15.1967, 90.0, 235.777, 209.518,
      122.828, 45.5215, -60.0, 79.6505, 150.914, 113.838, 117.777, 170.5;

  EMatXf deltas(5, 5);
  // 0.174533 radians -> 10 degrees
  deltas << 0.47861834, 0.13992102, 0.14961673, 0.71495209, 0.0, 0.29915856,
      -0.35664671, 0.89018666, 0.70815367, 0.174533, -0.03852064, 0.44466892,
      0.49492538, 0.71409376, 0.174533, 0.28052918, 0.02184832, 0.65289006,
      1.05060139, 0.174533, -0.38172557, -0.08533806, -0.60335309, 0.79052375,
      0.174533;

  EMatXf result_gt(5, 5);
  result_gt << 252.668, 107.367, 91.4381, 276.165, 0.0, 217.686, 19.3551,
      147.631, 205.397, 55.0, 187.363, 214.185, 19.865, 31.0368, 100.0, 270.234,
      210.513, 235.963, 130.163, -50.0, 36.1956, 140.863, 62.2665, 259.645,
      180.5;

  const float BBOX_XFORM_CLIP = log(1000.0 / 16.0);
  auto result = utils::bbox_transform(
      bbox.array(),
      deltas.array(),
      std::vector<float>{1.0, 1.0, 1.0, 1.0},
      BBOX_XFORM_CLIP,
      true, /* legacy_plus_one */
      false /* angle_bound_on */);
  EXPECT_NEAR((result.matrix() - result_gt).norm(), 0.0, 1e-2);
}

TEST(UtilsBoxesTest, TestBboxTransformRotatedNormalized) {
  // Test rotated bbox transform with angle normalization
  using EMatXf = Eigen::MatrixXf;

  EMatXf bbox(5, 5);
  bbox << 214.986, 88.4628, 78.7317, 135.104, 0.0, 199.553, 55.4367, 60.6142,
      101.169, 45.0, 187.829, 207.427, 0012.11, 15.1967, 90.0, 235.777, 209.518,
      122.828, 45.5215, -60.0, 79.6505, 150.914, 113.838, 117.777, 170.5;

  EMatXf deltas(5, 5);
  // 0.174533 radians -> 10 degrees
  deltas << 0.47861834, 0.13992102, 0.14961673, 0.71495209, 0.0, 0.29915856,
      -0.35664671, 0.89018666, 0.70815367, 0.174533, -0.03852064, 0.44466892,
      0.49492538, 0.71409376, 0.174533, 0.28052918, 0.02184832, 0.65289006,
      1.05060139, 0.174533, -0.38172557, -0.08533806, -0.60335309, 0.79052375,
      0.174533;

  EMatXf result_gt(5, 5);
  result_gt << 252.668, 107.367, 91.4381, 276.165, 0.0, 217.686, 19.3551,
      147.631, 205.397, 55.0, 187.363, 214.185, 19.865, 31.0368, -80.0, 270.234,
      210.513, 235.963, 130.163, -50.0, 36.1956, 140.863, 62.2665, 259.645, 0.5;

  const float BBOX_XFORM_CLIP = log(1000.0 / 16.0);
  auto result = utils::bbox_transform(
      bbox.array(),
      deltas.array(),
      std::vector<float>{1.0, 1.0, 1.0, 1.0},
      BBOX_XFORM_CLIP,
      true, /* legacy_plus_one */
      true, /* angle_bound_on */
      -90, /* angle_bound_lo */
      90 /* angle_bound_hi */);
  EXPECT_NEAR((result.matrix() - result_gt).norm(), 0.0, 1e-2);
}

TEST(UtilsBoxesTest, ClipRotatedBoxes) {
  // Test utils::clip_boxes_rotated()
  using EMatXf = Eigen::MatrixXf;

  int height = 800;
  int width = 600;
  EMatXf bbox(5, 5);
  bbox << 20, 20, 200, 150, 0, // Horizontal
      20, 20, 200, 150, 0.5, // Almost horizontal
      20, 20, 200, 150, 30, // Rotated
      300, 300, 200, 150, 30, // Rotated
      579, 779, 200, 150, -0.5; // Almost horizontal

  // Test with no clipping
  float angle_thresh = -1.0;
  auto result = utils::clip_boxes(
      bbox.array(), height, width, angle_thresh, true /* legacy_plus_one */);
  EXPECT_NEAR((result.matrix() - bbox).norm(), 0.0, 1e-4);

  EMatXf result_gt(5, 5);
  result_gt << 59.75, 47.25, 120.5, 95.5, 0, 59.75, 47.25, 120.5, 95.5, 0.5, 20,
      20, 200, 150, 30, 300, 300, 200, 150, 30, 539.25, 751.75, 120.5, 95.5,
      -0.5;

  // Test clipping with tolerance
  angle_thresh = 1.0;
  result = utils::clip_boxes(
      bbox.array(), height, width, angle_thresh, true /* legacy_plus_one */);
  EXPECT_NEAR((result.matrix() - result_gt).norm(), 0.0, 1e-4);
}

} // namespace caffe2
