#include "caffe2/utils/eigen_utils.h"
#include "generate_proposals_op_util_nms.h"

#include <gtest/gtest.h>

namespace caffe2 {

TEST(UtilsNMSTest, TestNMS) {
  Eigen::ArrayXXf input(5, 5);
  input << 10, 10, 50, 60, 0.5, 11, 12, 48, 60, 0.7, 8, 9, 40, 50, 0.6, 100,
      100, 150, 140, 0.9, 99, 110, 155, 139, 0.8;

  std::vector<float> input_thresh{0.1f, 0.3f, 0.5f, 0.8f, 0.9f};
  // ground truth generated based on detection.caffe2/lib/nms/py_cpu_nms.py
  std::vector<std::vector<int>> output_gt{
      {3, 1}, {3, 1}, {3, 1}, {3, 4, 1, 2}, {3, 4, 1, 2, 0}};

  // test utils::nms_cpu without indices input
  auto proposals = input.block(0, 0, input.rows(), 4);
  auto scores = input.col(4);
  for (int i = 0; i < input_thresh.size(); i++) {
    auto cur_out = utils::nms_cpu(proposals, scores, input_thresh[i]);
    EXPECT_EQ(output_gt[i], cur_out);
  }

  // test utils::nms_cpu with indices
  std::vector<int> indices(proposals.rows());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(
      indices.data(),
      indices.data() + indices.size(),
      [&scores](int lhs, int rhs) { return scores(lhs) > scores(rhs); });
  for (int i = 0; i < input_thresh.size(); i++) {
    auto cur_out = utils::nms_cpu(proposals, scores, indices, input_thresh[i]);
    EXPECT_EQ(output_gt[i], cur_out);
  }

  // test utils::nms_cpu with topN
  std::vector<int> top_n = {1, 1, 2, 2, 3};
  auto gt_out = output_gt;
  for (int i = 0; i < input_thresh.size(); i++) {
    auto cur_out =
        utils::nms_cpu(proposals, scores, indices, input_thresh[i], top_n[i]);
    gt_out[i].resize(top_n[i]);
    EXPECT_EQ(gt_out[i], cur_out);
  }
}

TEST(UtilsNMSTest, TestNMS1) {
  Eigen::ArrayXXf proposals(53, 4);
  proposals << 350.9821, 161.8200, 369.9685, 205.2372, 250.5236, 154.2844,
      274.1773, 204.9810, 471.4920, 160.4118, 496.0094, 213.4244, 352.0421,
      164.5933, 366.4458, 205.9624, 166.0765, 169.7707, 183.0102, 232.6606,
      252.3000, 183.1449, 269.6541, 210.6747, 469.7862, 162.0192, 482.1673,
      187.0053, 168.4862, 174.2567, 181.7437, 232.9379, 470.3290, 162.3442,
      496.4272, 214.6296, 251.0450, 155.5911, 272.2693, 203.3675, 252.0326,
      154.7950, 273.7404, 195.3671, 351.7479, 161.9567, 370.6432, 204.3047,
      496.3306, 161.7157, 515.0573, 210.7200, 471.0749, 162.6143, 485.3374,
      207.3448, 250.9745, 160.7633, 264.1924, 206.8350, 470.4792, 169.0351,
      487.1934, 220.2984, 474.4227, 161.9546, 513.1018, 215.5193, 251.9428,
      184.1950, 262.6937, 207.6416, 252.6623, 175.0252, 269.8806, 213.7584,
      260.9884, 157.0351, 288.3554, 206.6027, 251.3629, 164.5101, 263.2179,
      202.4203, 471.8361, 190.8142, 485.6812, 220.8586, 248.6243, 156.9628,
      264.3355, 199.2767, 495.1643, 158.0483, 512.6261, 184.4192, 376.8718,
      168.0144, 387.3584, 201.3210, 122.9191, 160.7433, 172.5612, 231.3837,
      350.3857, 175.8806, 366.2500, 205.4329, 115.2958, 162.7822, 161.9776,
      229.6147, 168.4375, 177.4041, 180.8028, 232.4551, 169.7939, 184.4330,
      181.4767, 232.1220, 347.7536, 175.9356, 355.8637, 197.5586, 495.5434,
      164.6059, 516.4031, 207.7053, 172.1216, 194.6033, 183.1217, 235.2653,
      264.2654, 181.5540, 288.4626, 214.0170, 111.7971, 183.7748, 137.3745,
      225.9724, 253.4919, 186.3945, 280.8694, 210.0731, 165.5334, 169.7344,
      185.9159, 232.8514, 348.3662, 184.5187, 354.9081, 201.4038, 164.6562,
      162.5724, 186.3108, 233.5010, 113.2999, 186.8410, 135.8841, 219.7642,
      117.0282, 179.8009, 142.5375, 221.0736, 462.1312, 161.1004, 495.3576,
      217.2208, 462.5800, 159.9310, 501.2937, 224.1655, 503.5242, 170.0733,
      518.3792, 209.0113, 250.3658, 195.5925, 260.6523, 212.4679, 108.8287,
      163.6994, 146.3642, 229.7261, 256.7617, 187.3123, 288.8407, 211.2013,
      161.2781, 167.4801, 186.3751, 232.7133, 115.3760, 177.5859, 163.3512,
      236.9660, 248.9077, 188.0919, 264.8579, 207.9718, 108.1349, 160.7851,
      143.6370, 229.6243, 465.0900, 156.7555, 490.3561, 213.5704, 107.5338,
      173.4323, 141.0704, 235.2910;

  Eigen::ArrayXXf scores(53, 1);
  scores << 0.1919, 0.3293, 0.0860, 0.1600, 0.1885, 0.4297, 0.0974, 0.2711,
      0.1483, 0.1173, 0.1034, 0.2915, 0.1993, 0.0677, 0.3217, 0.0966, 0.0526,
      0.5675, 0.3130, 0.1592, 0.1353, 0.0634, 0.1557, 0.1512, 0.0699, 0.0545,
      0.2692, 0.1143, 0.0572, 0.1990, 0.0558, 0.1500, 0.2214, 0.1878, 0.2501,
      0.1343, 0.0809, 0.1266, 0.0743, 0.0896, 0.0781, 0.0983, 0.0557, 0.0623,
      0.5808, 0.3090, 0.1050, 0.0524, 0.0513, 0.4501, 0.4167, 0.0623, 0.1749;

  std::vector<int> output_gt{1,  6,  7,  8,  11, 12, 13, 14, 17,
                             18, 19, 21, 23, 24, 25, 26, 30, 32,
                             33, 34, 35, 37, 43, 44, 47, 50};

  auto cur_out = utils::nms_cpu(proposals, scores, 0.5);
  std::sort(cur_out.begin(), cur_out.end());
  EXPECT_EQ(output_gt, cur_out);
}

TEST(UtilsNMSTest, TestSoftNMS) {
  Eigen::ArrayXXf input(5, 5);
  input.row(0) << 5.18349426e+02, 1.77783920e+02, 9.06085266e+02,
      2.59163239e+02, 8.17906916e-01;
  input.row(1) << 2.11392624e+02, 1.76144958e+02, 6.14215149e+02,
      2.48934662e+02, 9.52467501e-01;
  input.row(2) << 4.65724518e+02, 1.83594269e+02, 9.39000000e+02,
      2.55136627e+02, 6.73921347e-01;
  input.row(3) << 6.07164246e+02, 2.60230377e+02, 8.32768127e+02,
      3.39919891e+02, 9.99834776e-01;
  input.row(4) << 3.23936859e+02, 3.43427063e+02, 6.20561157e+02,
      3.98286072e+02, 9.99737203e-01;

  const auto& proposals = input.block(0, 0, input.rows(), 4);
  const auto& scores = input.col(4);

  vector<int> method{1, 1, 2, 2};
  vector<float> overlap_thresh{0.1f, 0.3f, 0.1f, 0.3f};

  // Ground truth generated based on
  //   detectron/lib/utils/cython_nms.pyx
  std::vector<int> keep_gt{3, 4, 1, 0, 2};

  // Explicitly use colmajor order to match scores
  Eigen::ArrayXXf scores_gt(5, 4);
  // Linear, overlap_thresh=0.1
  scores_gt.col(0) << 7.13657320e-01, 9.52467501e-01, 1.44501388e-01,
      9.99834776e-01, 9.99737203e-01;
  // Linear, overlap_thresh=0.3
  scores_gt.col(1) << 8.17906916e-01, 9.52467501e-01, 1.76800430e-01,
      9.99834776e-01, 9.99737203e-01;
  // Gaussian, overlap_thresh=0.1
  scores_gt.col(2) << 7.91758895e-01, 9.52467501e-01, 2.12320581e-01,
      9.99834776e-01, 9.99737203e-01;
  // Gaussian, overlap_thresh=0.3
  scores_gt.col(3) << 7.91758895e-01, 9.52467501e-01, 2.12320581e-01,
      9.99834776e-01, 9.99737203e-01;

  Eigen::ArrayXf out_scores;
  for (int i = 0; i < method.size(); ++i) {
    LOG(INFO) << "Testing SoftNMS with method=" << method[i]
              << ", overlap_thresh=" << overlap_thresh[i];
    const auto& expected_scores = scores_gt.col(i);

    auto keep = utils::soft_nms_cpu(
        &out_scores,
        proposals,
        scores,
        0.5,
        overlap_thresh[i],
        0.0001,
        method[i]);
    EXPECT_EQ(keep, keep_gt);
    {
      auto diff = expected_scores - out_scores;
      EXPECT_TRUE((diff.abs() < 1e-6).all());
    }

    // Test with topN
    for (int topN = 1; topN <= 3; ++topN) {
      keep = utils::soft_nms_cpu(
          &out_scores,
          proposals,
          scores,
          0.5,
          overlap_thresh[i],
          0.0001,
          method[i],
          topN);
      std::vector<int> expected_keep(keep_gt.begin(), keep_gt.begin() + topN);
      EXPECT_EQ(expected_keep, keep);
    }

    // Test with filtered indices
    auto indices = utils::GetArrayIndices(scores >= 0.9);
    keep = utils::soft_nms_cpu(
        &out_scores,
        proposals,
        scores,
        indices,
        0.5,
        overlap_thresh[i],
        0.0001,
        method[i]);
    std::sort(keep.begin(), keep.end());
    EXPECT_EQ(indices, keep);
    {
      const auto& expected = utils::GetSubArray(expected_scores, indices);
      const auto& actual = utils::GetSubArray(out_scores, indices);
      EXPECT_TRUE(((expected - actual).abs() < 1e-6).all());
    }

    // Test with high score_thresh
    float score_thresh = 0.9;
    keep = utils::soft_nms_cpu(
        &out_scores,
        proposals,
        scores,
        0.5,
        overlap_thresh[i],
        score_thresh,
        method[i]);
    {
      auto expected_keep =
          utils::GetArrayIndices(expected_scores >= score_thresh);
      std::sort(keep.begin(), keep.end());
      EXPECT_EQ(expected_keep, keep);

      const auto& expected = utils::GetSubArray(expected_scores, expected_keep);
      const auto& actual = utils::GetSubArray(out_scores, expected_keep);
      EXPECT_TRUE(((expected - actual).abs() < 1e-6).all());
    }
  }
}

#if defined(CV_MAJOR_VERSION) && (CV_MAJOR_VERSION >= 3)
TEST(UtilsNMSTest, TestNMSRotatedAngle0) {
  // Same inputs as TestNMS, but in RRPN format with angle 0 for testing
  // nms_cpu_rotated
  Eigen::ArrayXXf input(5, 5);
  input << 10, 10, 50, 60, 0.5, 11, 12, 48, 60, 0.7, 8, 9, 40, 50, 0.6, 100,
      100, 150, 140, 0.9, 99, 110, 155, 139, 0.8;

  std::vector<float> input_thresh{0.1f, 0.3f, 0.5f, 0.8f, 0.9f};
  // ground truth generated based on detection.caffe2/lib/nms/py_cpu_nms.py
  std::vector<std::vector<int>> output_gt{
      {3, 1}, {3, 1}, {3, 1}, {3, 4, 1, 2}, {3, 4, 1, 2, 0}};

  // test utils::nms_cpu without indices input.
  // Add additional dim for angle and convert from
  // [x1, y1, x2, y1] to [ctr_x, ctr_y, w, h] format.
  Eigen::ArrayXXf proposals = Eigen::ArrayXXf::Zero(input.rows(), 5);
  proposals.col(0) = (input.col(0) + input.col(2)) / 2.0; // ctr_x = (x1 + x2)/2
  proposals.col(1) = (input.col(1) + input.col(3)) / 2.0; // ctr_y = (y1 + y2)/2
  proposals.col(2) = input.col(2) - input.col(0) + 1.0; // w = x2 - x1 + 1
  proposals.col(3) = input.col(3) - input.col(1) + 1.0; // h = y2 - y1 + 1

  auto scores = input.col(4);
  for (int i = 0; i < input_thresh.size(); i++) {
    auto cur_out = utils::nms_cpu(proposals, scores, input_thresh[i]);
    EXPECT_EQ(output_gt[i], cur_out);
  }

  // test utils::nms_cpu with indices
  std::vector<int> indices(proposals.rows());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(
      indices.data(),
      indices.data() + indices.size(),
      [&scores](int lhs, int rhs) { return scores(lhs) > scores(rhs); });
  for (int i = 0; i < input_thresh.size(); i++) {
    auto cur_out = utils::nms_cpu(proposals, scores, indices, input_thresh[i]);
    EXPECT_EQ(output_gt[i], cur_out);
  }

  // test utils::nms_cpu with topN
  std::vector<int> top_n = {1, 1, 2, 2, 3};
  auto gt_out = output_gt;
  for (int i = 0; i < input_thresh.size(); i++) {
    auto cur_out =
        utils::nms_cpu(proposals, scores, indices, input_thresh[i], top_n[i]);
    gt_out[i].resize(top_n[i]);
    EXPECT_EQ(gt_out[i], cur_out);
  }
}

TEST(UtilsNMSTest, TestSoftNMSRotatedAngle0) {
  // Same inputs as TestSoftNMS, but in RRPN format with angle 0 for testing
  // nms_cpu_rotated
  Eigen::ArrayXXf input(5, 5);
  input.row(0) << 5.18349426e+02, 1.77783920e+02, 9.06085266e+02,
      2.59163239e+02, 8.17906916e-01;
  input.row(1) << 2.11392624e+02, 1.76144958e+02, 6.14215149e+02,
      2.48934662e+02, 9.52467501e-01;
  input.row(2) << 4.65724518e+02, 1.83594269e+02, 9.39000000e+02,
      2.55136627e+02, 6.73921347e-01;
  input.row(3) << 6.07164246e+02, 2.60230377e+02, 8.32768127e+02,
      3.39919891e+02, 9.99834776e-01;
  input.row(4) << 3.23936859e+02, 3.43427063e+02, 6.20561157e+02,
      3.98286072e+02, 9.99737203e-01;

  // Add additional dim for angle and convert from
  // [x1, y1, x2, y1] to [ctr_x, ctr_y, w, h] format.
  Eigen::ArrayXXf proposals = Eigen::ArrayXXf::Zero(input.rows(), 5);
  proposals.col(0) = (input.col(0) + input.col(2)) / 2.0; // ctr_x = (x1 + x2)/2
  proposals.col(1) = (input.col(1) + input.col(3)) / 2.0; // ctr_y = (y1 + y2)/2
  proposals.col(2) = input.col(2) - input.col(0) + 1.0; // w = x2 - x1 + 1
  proposals.col(3) = input.col(3) - input.col(1) + 1.0; // h = y2 - y1 + 1

  const auto& scores = input.col(4);

  vector<int> method{1, 1, 2, 2};
  vector<float> overlap_thresh{0.1f, 0.3f, 0.1f, 0.3f};

  // Ground truth generated based on
  //   detectron/lib/utils/cython_nms.pyx
  std::vector<int> keep_gt{3, 4, 1, 0, 2};

  // Explicitly use colmajor order to match scores
  Eigen::ArrayXXf scores_gt(5, 4);
  // Linear, overlap_thresh=0.1
  scores_gt.col(0) << 7.13657320e-01, 9.52467501e-01, 1.44501388e-01,
      9.99834776e-01, 9.99737203e-01;
  // Linear, overlap_thresh=0.3
  scores_gt.col(1) << 8.17906916e-01, 9.52467501e-01, 1.76800430e-01,
      9.99834776e-01, 9.99737203e-01;
  // Gaussian, overlap_thresh=0.1
  scores_gt.col(2) << 7.91758895e-01, 9.52467501e-01, 2.12320581e-01,
      9.99834776e-01, 9.99737203e-01;
  // Gaussian, overlap_thresh=0.3
  scores_gt.col(3) << 7.91758895e-01, 9.52467501e-01, 2.12320581e-01,
      9.99834776e-01, 9.99737203e-01;

  Eigen::ArrayXf out_scores;
  for (int i = 0; i < method.size(); ++i) {
    LOG(INFO) << "Testing SoftNMS with method=" << method[i]
              << ", overlap_thresh=" << overlap_thresh[i];
    const auto& expected_scores = scores_gt.col(i);

    auto keep = utils::soft_nms_cpu(
        &out_scores,
        proposals,
        scores,
        0.5,
        overlap_thresh[i],
        0.0001,
        method[i]);
    EXPECT_EQ(keep, keep_gt);
    {
      auto diff = expected_scores - out_scores;
      EXPECT_TRUE((diff.abs() < 1e-6).all());
    }

    // Test with topN
    for (int topN = 1; topN <= 3; ++topN) {
      keep = utils::soft_nms_cpu(
          &out_scores,
          proposals,
          scores,
          0.5,
          overlap_thresh[i],
          0.0001,
          method[i],
          topN);
      std::vector<int> expected_keep(keep_gt.begin(), keep_gt.begin() + topN);
      EXPECT_EQ(expected_keep, keep);
    }

    // Test with filtered indices
    auto indices = utils::GetArrayIndices(scores >= 0.9);
    keep = utils::soft_nms_cpu(
        &out_scores,
        proposals,
        scores,
        indices,
        0.5,
        overlap_thresh[i],
        0.0001,
        method[i]);
    std::sort(keep.begin(), keep.end());
    EXPECT_EQ(indices, keep);
    {
      const auto& expected = utils::GetSubArray(expected_scores, indices);
      const auto& actual = utils::GetSubArray(out_scores, indices);
      EXPECT_TRUE(((expected - actual).abs() < 1e-6).all());
    }

    // Test with high score_thresh
    float score_thresh = 0.9;
    keep = utils::soft_nms_cpu(
        &out_scores,
        proposals,
        scores,
        0.5,
        overlap_thresh[i],
        score_thresh,
        method[i]);
    {
      auto expected_keep =
          utils::GetArrayIndices(expected_scores >= score_thresh);
      std::sort(keep.begin(), keep.end());
      EXPECT_EQ(expected_keep, keep);

      const auto& expected = utils::GetSubArray(expected_scores, expected_keep);
      const auto& actual = utils::GetSubArray(out_scores, expected_keep);
      EXPECT_TRUE(((expected - actual).abs() < 1e-6).all());
    }
  }
}

TEST(UtilsNMSTest, RotatedBBoxOverlaps) {
  {
    // Simple case with angle 0 (upright boxes)
    Eigen::ArrayXXf boxes(2, 5);
    boxes << 10.5, 15.5, 21, 31, 0, 14.0, 17, 4, 10, 0;

    Eigen::ArrayXXf query_boxes(3, 5);
    query_boxes << 30.5, 10.5, 41, 1, 0, 13.5, 21.5, 5, 21, 0, 10.5, 15.5, 21,
        31, 0;

    Eigen::ArrayXXf expected(2, 3);
    expected << 0.0161527172, 0.152439028, 1., 0., 0.38095239, 0.0614439324;

    auto actual = utils::bbox_overlaps_rotated(boxes, query_boxes);
    EXPECT_TRUE(((expected - actual).abs() < 1e-6).all());
  }

  {
    // Angle 45
    Eigen::ArrayXXf boxes(1, 5);
    boxes << 0, 0, 2.0 * std::sqrt(2), 2.0 * std::sqrt(2), 45;

    Eigen::ArrayXXf query_boxes(1, 5);
    query_boxes << 1, 1, 2, 2, 0;

    Eigen::ArrayXXf expected(1, 1);
    expected << 0.2;

    auto actual = utils::bbox_overlaps_rotated(boxes, query_boxes);
    EXPECT_TRUE(((expected - actual).abs() < 1e-6).all());
  }

  {
    Eigen::ArrayXXf boxes(2, 5);
    boxes << 60.0, 60.0, 100.0, 100.0, 0.0, 50.0, 50.0, 100.0, 100.0, 135.0;

    Eigen::ArrayXXf query_boxes(6, 5);
    query_boxes << 60.0, 60.0, 100.0, 100.0, 180.0, 50.0, 50.0, 100.0, 100.0,
        45.0, 80.0, 50.0, 100.0, 100.0, 0.0, 50.0, 50.0, 200.0, 50.0, 45.0,
        200.0, 200.0, 100.0, 100.0, 0, 60.0, 60.0, 100.0, 100.0, 1.0;

    Eigen::ArrayXXf expected(2, 6);
    expected << 1., 0.6507467031, 0.5625, 0.3718426526, 0., 0.9829941392,
        0.6507467628, 1., 0.4893216789, 0.3333334029, 0., 0.6508141756;

    auto actual = utils::bbox_overlaps_rotated(boxes, query_boxes);
    EXPECT_TRUE(((expected - actual).abs() < 1e-6).all());
  }
}
#endif // CV_MAJOR_VERSION >= 3

} // namespace caffe2
