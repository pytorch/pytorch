// Copyright 2004-present Facebook. All Rights Reserved.

#include <ATen/native/vulkan/ops/heatmap_max_keypoint.h>

namespace at {
namespace native {

/**
Mask R-CNN uses bicubic upscaling before taking the maximum of the heat map
for keypoints. We would like to avoid bicubic upscaling, because it is
computationally expensive. This approach uses the Taylor expansion up to the
quadratic terms on approximation of the heatmap function.
**/
at::Tensor HeatmapMaxKeypointCPUKernel(
    const at::Tensor& heatmaps_in_,
    const at::Tensor& bboxes_in_,
    bool should_output_softmax_,
    c10::optional<std::vector<at::Tensor>>
) {
  // only support NCHW format
  const auto heatmaps_in = heatmaps_in_.contiguous();
  const auto bboxes_in = bboxes_in_.contiguous();

  TORCH_CHECK(heatmaps_in.dim() == 4);
  const int N = heatmaps_in.size(0);
  TORCH_CHECK(heatmaps_in.size(0) == N);
  const int keypoint_count = heatmaps_in.size(1);
  const int heatmap_size = heatmaps_in.size(2);
  TORCH_CHECK(heatmap_size >= 2); // at least 2x2 for approx
  TORCH_CHECK(heatmaps_in.size(2) == heatmaps_in.size(3));

  TORCH_CHECK(bboxes_in.dim() == 2);
  TORCH_CHECK(bboxes_in.size(0) == N);
  TORCH_CHECK(bboxes_in.size(1) >= 4);

  // Wrap inputs in Eigen
  Eigen::Map<const caffe2::ERArrXXf> heatmaps(
      heatmaps_in.data_ptr<float>(),
      heatmaps_in.size(0) * heatmaps_in.size(1),
      heatmaps_in.size(2) * heatmaps_in.size(3));
  Eigen::Map<const caffe2::ERArrXXf> bboxes(
      bboxes_in.data_ptr<float>(), bboxes_in.size(0), bboxes_in.size(1));

  // Calculate the softmax
  caffe2::ERArrXXf probs(
      heatmaps_in.size(0) * heatmaps_in.size(1),
      heatmaps_in.size(2) * heatmaps_in.size(3));
  if (should_output_softmax_) {
    // softmax output is expensive to compute, if should_output_softmax is not
    // specified, don't populate it
    caffe2::ERArrXXf heatmap_exp = heatmaps.exp();
    for (int r = 0; r < N * keypoint_count; r++) {
      probs.row(r) = heatmap_exp.row(r) / heatmap_exp.row(r).sum();
    }
  } /* otherwise not initialized */

  // Resize and wrap outputs in Eigen
  at::Tensor keypoints_out = torch::zeros({N, 4, keypoint_count}, torch::dtype(torch::kFloat32));
  Eigen::Map<caffe2::ERArrXXf> keypoints(
      keypoints_out.data_ptr<float>(), N, 4 * keypoint_count);

  caffe2::EArrXi maxIndices(N * keypoint_count);
  // finding max value first (only maxCoeff() is vectorized, not
  // maxCoeff(&index)), then find the index (equalness check is also fast)
  caffe2::EArrXf maxScores = heatmaps.rowwise().maxCoeff();
  for (int r = 0; r < N * keypoint_count; r++) {
    float maxScore = maxScores[r];
    for (int c = 0; c < heatmap_size * heatmap_size; c++) {
      if (heatmaps(r, c) == maxScore) {
        maxIndices[r] = c;
        break;
      }
    }
  }

  // Populate outputs
  for (int k = 0; k < N; k++) { // For each box, even skipped

    float x0 = bboxes(k, 0);
    float y0 = bboxes(k, 1);
    float xLen = std::max(bboxes(k, 2) - bboxes(k, 0), 1.0f);
    float yLen = std::max(bboxes(k, 3) - bboxes(k, 1), 1.0f);

    // Extract max keypoints and probabilities from heatmaps
    for (int j = 0; j < keypoint_count; j++) {
      const int heatmap_index = k * keypoint_count + j;
      const int maxIndex = maxIndices[heatmap_index];
      const float maxScore = maxScores[heatmap_index];
      const int maxY = maxIndex / heatmap_size;
      const int maxX = maxIndex - heatmap_size * maxY;

      assert(heatmaps(heatmap_index, maxIndex) == maxScore);
      caffe2::ERArrXXf fmax = caffe2::ERArrXXf::Zero(3, 3);

      // initialize fmax values of local 3x3 grid
      // when 3x3 grid going out-of-bound, mirrowing around center
      for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
          int xx = x - 2 * (x + maxX >= heatmap_size) + 2 * (x + maxX < 0);
          int yy = y - 2 * (y + maxY >= heatmap_size) + 2 * (y + maxY < 0);
          assert((xx + maxX < heatmap_size) && (xx + maxX >= 0));
          assert((yy + maxY < heatmap_size) && (yy + maxY >= 0));
          const int coord_index = (yy + maxY) * heatmap_size + xx + maxX;
          fmax(y + 1, x + 1) = heatmaps(heatmap_index, coord_index);
        }
      }

      // b = -f'(0), A = f''(0) Hessian matrix
      caffe2::EVecXf b(2);
      b << -(fmax(1, 2) - fmax(1, 0)) / 2, -(fmax(2, 1) - fmax(0, 1)) / 2;
      caffe2::EMatXf A(2, 2);
      A << fmax(1, 0) - 2 * fmax(1, 1) + fmax(1, 2),
          (fmax(2, 2) - fmax(2, 0) - fmax(0, 2) + fmax(0, 0)) / 4,
          (fmax(2, 2) - fmax(2, 0) - fmax(0, 2) + fmax(0, 0)) / 4,
          fmax(0, 1) - 2 * fmax(1, 1) + fmax(2, 1);

      // Solve Ax=b
      const float div = A.determinant();
      caffe2::EVecXf delta(2);
      float deltaScore;
      const float MAX_DELTA = 1.5;
      if (std::abs(div) < 1e-4f) {
        delta << 0.0f, 0.0f;
        deltaScore = maxScore;
      } else {
        delta = A.ldlt().solve(b);
        // clip delta if going out-of-range of 3x3 grid
        if (std::abs(delta(0)) > MAX_DELTA || std::abs(delta(1)) > MAX_DELTA) {
          float larger_delta = std::max(std::abs(delta(0)), std::abs(delta(1)));
          delta(0) = delta(0) / larger_delta * MAX_DELTA;
          delta(1) = delta(1) / larger_delta * MAX_DELTA;
        }
        deltaScore = fmax(1, 1) - b.transpose() * delta +
            1.0 / 2.0 * delta.transpose() * A * delta;
      }
      assert(std::abs(delta(0)) <= MAX_DELTA);
      assert(std::abs(delta(1)) <= MAX_DELTA);
      // find maximum of delta scores
      keypoints(k, 0 * keypoint_count + j) =
          x0 + (0.5 + maxX + delta(0)) * xLen / heatmap_size;
      keypoints(k, 1 * keypoint_count + j) =
          y0 + (0.5 + maxY + delta(1)) * yLen / heatmap_size;
      keypoints(k, 2 * keypoint_count + j) = deltaScore;
      if (should_output_softmax_) {
        keypoints(k, 3 * keypoint_count + j) = probs(heatmap_index, maxIndex);
      } else {
        keypoints(k, 3 * keypoint_count + j) = .0f;
      }
    }
  }

  return keypoints_out;
}

} // namespace fb
} // namespace caffe2
