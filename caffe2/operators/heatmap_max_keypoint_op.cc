#include "heatmap_max_keypoint_op.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(
    HeatmapMaxKeypoint,
    HeatmapMaxKeypointOp<float, CPUContext>);

// Input: heatmaps [size x size], boxes [x0, y0, x1, y1]
// Output: keypoints (#rois, 4, #keypoints)
OPERATOR_SCHEMA(HeatmapMaxKeypoint).NumInputs(2).NumOutputs(1);

SHOULD_NOT_DO_GRADIENT(HeatmapMaxKeypoint);
} // namespace

/**
Mask R-CNN uses bicubic upscaling before taking the maximum of the heat map
for keypoints. We would like to avoid bicubic upscaling, because it is
computationally expensive. This approach uses the Taylor expansion up to the
quadratic terms on approximation of the heatmap function.
**/
template <>
bool HeatmapMaxKeypointOp<float, CPUContext>::RunOnDevice() {
  const auto& heatmaps_in = Input(0);
  const auto& bboxes_in = Input(1);

  CAFFE_ENFORCE_EQ(heatmaps_in.dim(), 4);
  const int N = heatmaps_in.dim32(0);
  CAFFE_ENFORCE_EQ(heatmaps_in.dim32(0), N);
  const int keypoint_count = heatmaps_in.dim32(1);
  const int heatmap_size = heatmaps_in.dim32(2);
  CAFFE_ENFORCE_GE(heatmap_size, 2); // at least 2x2 for approx
  CAFFE_ENFORCE_EQ(heatmaps_in.dim32(2), heatmaps_in.dim32(3));

  CAFFE_ENFORCE_EQ(bboxes_in.dim(), 2);
  CAFFE_ENFORCE_EQ(bboxes_in.dim32(0), N);
  CAFFE_ENFORCE_GE(bboxes_in.dim32(1), 4);

  // Wrap inputs in Eigen
  Eigen::Map<const ERArrXXf> heatmaps(
      heatmaps_in.data<float>(),
      heatmaps_in.dim32(0) * heatmaps_in.dim32(1),
      heatmaps_in.dim32(2) * heatmaps_in.dim32(3));
  Eigen::Map<const ERArrXXf> bboxes(
      bboxes_in.data<float>(), bboxes_in.dim32(0), bboxes_in.dim32(1));

  // Calculate the softmax
  ERArrXXf probs(
      heatmaps_in.dim32(0) * heatmaps_in.dim32(1),
      heatmaps_in.dim32(2) * heatmaps_in.dim32(3));
  if (should_output_softmax_) {
    // softmax output is expensive to compute, if should_output_softmax is not
    // specified, don't populate it
    ERArrXXf heatmap_exp = heatmaps.exp();
    for (int r = 0; r < N * keypoint_count; r++) {
      probs.row(r) = heatmap_exp.row(r) / heatmap_exp.row(r).sum();
    }
  } /* otherwise not initialized */

  // Resize and wrap outputs in Eigen
  auto* keypoints_out = Output(0, {N, 4, keypoint_count}, at::dtype<float>());
  Eigen::Map<ERArrXXf> keypoints(
      keypoints_out->mutable_data<float>(), N, 4 * keypoint_count);

  EArrXi maxIndices(N * keypoint_count);
  // finding max value first (only maxCoeff() is vectorized, not
  // maxCoeff(&index)), then find the index (equalness check is also fast)
  EArrXf maxScores = heatmaps.rowwise().maxCoeff();
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
      ERArrXXf fmax = ERArrXXf::Zero(3, 3);

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
      EVecXf b(2);
      b << -(fmax(1, 2) - fmax(1, 0)) / 2, -(fmax(2, 1) - fmax(0, 1)) / 2;
      EMatXf A(2, 2);
      A << fmax(1, 0) - 2 * fmax(1, 1) + fmax(1, 2),
          (fmax(2, 2) - fmax(2, 0) - fmax(0, 2) + fmax(0, 0)) / 4,
          (fmax(2, 2) - fmax(2, 0) - fmax(0, 2) + fmax(0, 0)) / 4,
          fmax(0, 1) - 2 * fmax(1, 1) + fmax(2, 1);

      // Solve Ax=b
      const float div = A.determinant();
      EVecXf delta(2);
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

  return true;
}

} // namespace caffe2

using HeatmapMaxKeypointOpFloatCPU =
    caffe2::HeatmapMaxKeypointOp<float, caffe2::CPUContext>;

// clang-format off
C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    HeatmapMaxKeypoint,
    "_caffe2::HeatmapMaxKeypoint("
      "Tensor heatmaps, "
      "Tensor bboxes_in, "
      "bool should_output_softmax = True"
    ") -> Tensor keypoints",
    HeatmapMaxKeypointOpFloatCPU);

// clang-format on
