#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/flags.h"
#include "caffe2/operators/generate_proposals_op_util_nms.h"
#include "caffe2/operators/generate_proposals_op_util_nms_gpu.h"
#include "caffe2/operators/utility_ops.h"
#include "caffe2/utils/eigen_utils.h"

#include <gtest/gtest.h>

#include <chrono>
#include <random>

namespace caffe2 {

TEST(UtilsNMSTest, TestNMSGPU) {
  if (!HasCudaGPU())
    return;
  const int box_dim = 4;
  std::vector<float> boxes = {10, 10, 50,  60,  11,  12,  48, 60,  8,   9,
                              40, 50, 100, 100, 150, 140, 99, 110, 155, 139};

  std::vector<float> scores = {0.5f, 0.7f, 0.6f, 0.9f, 0.8f};

  std::vector<int> indices(scores.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&scores](int lhs, int rhs) {
    return scores[lhs] > scores[rhs];
  });
  std::vector<float> sorted_boxes(boxes.size());
  for (int i = 0; i < scores.size(); ++i) {
    for (int d = 0; d < box_dim; ++d)
      sorted_boxes[i * box_dim + d] = boxes[indices[i] * box_dim + d];
  }

  Workspace ws;
  DeviceOption option;
  option.set_device_type(PROTO_CUDA);
  CUDAContext cuda_context(option);

  Tensor dev_sorted_boxes{CUDA};
  Tensor dev_scores{CUDA};
  Tensor dev_boxes_valid_flags{CUDA};
  Tensor dev_list{CUDA};
  Tensor dev_delete_mask{CUDA};
  Tensor host_delete_mask{CPU};
  Tensor dev_list_nitems{CUDA};
  Tensor host_list{CPU};

  int nboxes = boxes.size() / box_dim;
  dev_sorted_boxes.Resize(box_dim * nboxes);
  dev_list.Resize(nboxes);
  host_list.Resize(nboxes);

  float* d_sorted_boxes = dev_sorted_boxes.template mutable_data<float>();
  int* d_list = dev_list.template mutable_data<int>();
  int* h_list = host_list.template mutable_data<int>();

  CUDA_CHECK(cudaMemcpyAsync(
      d_sorted_boxes,
      &sorted_boxes[0],
      sizeof(*d_sorted_boxes) * box_dim * nboxes,
      cudaMemcpyHostToDevice,
      cuda_context.cuda_stream()));

  std::vector<float> input_thresh{0.1f, 0.3f, 0.5f, 0.8f, 0.9f};
  std::vector<std::set<int>> output_gt{
      {0, 2}, {0, 2}, {0, 2}, {0, 1, 2, 3}, {0, 1, 2, 3, 4}};

  std::vector<int> keep(nboxes);
  std::set<int> keep_as_set;
  for (int itest = 0; itest < input_thresh.size(); ++itest) {
    const float thresh = input_thresh[itest];
    int list_nitems;
    utils::nms_gpu(
        d_sorted_boxes,
        nboxes,
        thresh,
        true, /* legacy_plus_one */
        d_list,
        &list_nitems,
        dev_delete_mask,
        host_delete_mask,
        &cuda_context,
        box_dim);

    cuda_context.FinishDeviceComputation();
    host_list.CopyFrom(dev_list);

    keep_as_set.clear();
    for (int i = 0; i < list_nitems; ++i) {
      keep_as_set.insert(h_list[i]);
    }

    // Sets are sorted
    // sets are equal <=> sets contains the same elements
    EXPECT_TRUE(output_gt[itest] == keep_as_set);
  }

  cuda_context.FinishDeviceComputation();
}

void generateRandomBoxes(float* h_boxes, float* h_scores, const int nboxes) {
  const float x_y_max = 100;
  const float w_h_max = 10;
  const float score_max = 1;

  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  std::uniform_real_distribution<float> coordinate_distribution(
      0.0, x_y_max - w_h_max);
  std::uniform_real_distribution<float> length_distribution(0.0, w_h_max);
  std::uniform_real_distribution<float> score_distribution(0.0, score_max);

  for (int ibox = 0; ibox < nboxes; ++ibox) {
    float x1, y1, x2, y2;
    x1 = coordinate_distribution(generator);
    y1 = coordinate_distribution(generator);
    x2 = x1 + length_distribution(generator);
    y2 = y1 + length_distribution(generator);
    h_boxes[ibox * 4 + 0] = x1;
    h_boxes[ibox * 4 + 1] = y1;
    h_boxes[ibox * 4 + 2] = x2;
    h_boxes[ibox * 4 + 3] = y2;
    h_scores[ibox] = score_distribution(generator);
  }
}

void generateRandomRotatedBoxes(
    float* h_boxes,
    float* h_scores,
    const int nboxes) {
  const float x_y_max = 100;
  const float w_h_max = 50;
  const float score_max = 1;
  const float angle_min = -90;
  const float angle_max = 90;

  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  std::uniform_real_distribution<float> coordinate_distribution(
      w_h_max / 2.0 + 1, x_y_max - w_h_max / 2.0 - 1);
  std::uniform_real_distribution<float> length_distribution(0.0, w_h_max);
  std::uniform_real_distribution<float> score_distribution(0.0, score_max);
  std::uniform_real_distribution<float> angle_distribution(
      angle_min, angle_max);

  for (int ibox = 0; ibox < nboxes; ++ibox) {
    float x_ctr, y_ctr, w, h, angle;
    x_ctr = coordinate_distribution(generator);
    y_ctr = coordinate_distribution(generator);
    w = length_distribution(generator);
    h = length_distribution(generator);
    angle = angle_distribution(generator);
    h_boxes[ibox * 5 + 0] = x_ctr;
    h_boxes[ibox * 5 + 1] = y_ctr;
    h_boxes[ibox * 5 + 2] = w;
    h_boxes[ibox * 5 + 3] = h;
    h_boxes[ibox * 5 + 4] = angle;
    h_scores[ibox] = score_distribution(generator);
  }
}

TEST(UtilsNMSTest, TestPerfNMS) {
  if (!HasCudaGPU())
    return;
  const int box_dim = 4;
  const int nboxes = 6000;

  Workspace ws;
  DeviceOption option;
  option.set_device_type(PROTO_CUDA);
  CUDAContext cuda_context(option);

  Tensor host_boxes{CPU};
  Tensor host_scores{CPU};
  host_boxes.Resize(box_dim * nboxes);
  host_scores.Resize(nboxes);

  float* h_boxes = host_boxes.template mutable_data<float>();
  float* h_scores = host_scores.template mutable_data<float>();

  // Generating random input
  generateRandomBoxes(h_boxes, h_scores, nboxes);

  Eigen::ArrayXXf proposals(nboxes, box_dim);
  Eigen::ArrayXXf scores(nboxes, 1);
  for (int i = 0; i < nboxes; ++i) {
    for (int d = 0; d < box_dim; ++d)
      proposals(i, d) = h_boxes[box_dim * i + d];
    scores(i, 0) = h_scores[i];
  }

  const int ntests = 50;
  const float thresh = 0.7;
  // Not timing the sort for the CPU
  // in the real-world use case scores already have been sorted earlier in the
  // generate proposals workflow
  std::vector<int> indices(proposals.rows());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(
      indices.data(),
      indices.data() + indices.size(),
      [&scores](int lhs, int rhs) { return scores(lhs) > scores(rhs); });

  // Running ntests runs of CPU NMS
  auto cpu_start = std::chrono::steady_clock::now();
  for (int itest = 0; itest < ntests; ++itest) {
    utils::nms_cpu(
        proposals,
        scores,
        indices,
        thresh,
        -1, /* topN */
        true /* legacy_plus_one */);
  }
  auto cpu_stop = std::chrono::steady_clock::now();

  std::vector<float> sorted_boxes(nboxes * box_dim);
  for (int i = 0; i < scores.size(); ++i) {
    for (int d = 0; d < box_dim; ++d)
      sorted_boxes[i * box_dim + d] = h_boxes[indices[i] * box_dim + d];
  }

  Tensor dev_boxes{CUDA};
  Tensor dev_delete_mask{CUDA};
  Tensor host_delete_mask{CPU};
  Tensor dev_list{CUDA};

  dev_boxes.Resize(box_dim * nboxes);
  float* d_sorted_boxes = dev_boxes.template mutable_data<float>();
  dev_list.Resize(nboxes);
  int* d_list = dev_list.template mutable_data<int>();
  int list_nitems;

  // No timing the memcpies because data is already on the GPU in the real-world
  // use case (generated by the GPU generate_proposals)
  CUDA_CHECK(cudaMemcpyAsync(
      d_sorted_boxes,
      &sorted_boxes[0],
      sizeof(*d_sorted_boxes) * box_dim * nboxes,
      cudaMemcpyHostToDevice,
      cuda_context.cuda_stream()));

  // Running ntests runs of GPU NMS
  auto gpu_start = std::chrono::steady_clock::now();
  for (int itest = 0; itest < ntests; ++itest) {
    utils::nms_gpu(
        d_sorted_boxes,
        nboxes,
        thresh,
        true, /* legacy_plus_one */
        d_list,
        &list_nitems,
        dev_delete_mask,
        host_delete_mask,
        &cuda_context,
        box_dim);
  }
  // Waiting for everything to be done
  CUDA_CHECK(cudaStreamSynchronize(cuda_context.cuda_stream()));
  auto gpu_stop = std::chrono::steady_clock::now();

  double total_cpu_time =
      std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();
  double total_gpu_time =
      std::chrono::duration<double, std::milli>(gpu_stop - gpu_start).count();
  double ratio = total_cpu_time / total_gpu_time;

  double avg_cpu_time = total_cpu_time / ntests;
  double avg_gpu_time = total_gpu_time / ntests;

  printf(
      "NMS, nproposals=%i, ntests=%i, Avg GPU time = %fms, Avg CPU time = %fms, GPU speed up = %fX \n",
      nboxes,
      ntests,
      avg_gpu_time,
      avg_cpu_time,
      ratio);
}

TEST(UtilsNMSTest, GPUEqualsCPUCorrectnessTest) {
  if (!HasCudaGPU())
    return;
  Workspace ws;
  DeviceOption option;
  option.set_device_type(PROTO_CUDA);
  CUDAContext cuda_context(option);

  const int box_dim = 4;
  const std::vector<int> nboxes_vec = {10, 100, 1000, 2000, 6000, 12000};
  for (int nboxes : nboxes_vec) {
    Tensor host_boxes{CPU};
    Tensor host_scores{CPU};
    host_boxes.Resize(box_dim * nboxes);
    host_scores.Resize(nboxes);

    float* h_boxes = host_boxes.template mutable_data<float>();
    float* h_scores = host_scores.template mutable_data<float>();

    // Generating random input
    generateRandomBoxes(h_boxes, h_scores, nboxes);

    const int ntests = 1;
    const float thresh = 0.7;
    // Not timing the sort for the CPU
    // in the real-world use case scores already have been sorted earlier in the
    // generate proposals workflow
    std::vector<int> indices(nboxes);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [h_scores](int lhs, int rhs) {
      return h_scores[lhs] > h_scores[rhs];
    });

    std::vector<float> sorted_boxes(nboxes * box_dim);
    std::vector<float> sorted_scores(nboxes);
    Eigen::ArrayXXf eig_proposals(nboxes, box_dim);
    Eigen::ArrayXXf eig_scores(nboxes, 1);
    for (int i = 0; i < nboxes; ++i) {
      for (int d = 0; d < box_dim; ++d) {
        sorted_boxes[i * box_dim + d] = h_boxes[indices[i] * box_dim + d];
        eig_proposals(i, d) = h_boxes[indices[i] * box_dim + d];
      }
      sorted_scores[i] = h_scores[indices[i]];
      eig_scores(i) = h_scores[indices[i]];
    }
    std::vector<int> sorted_indices(nboxes);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);

    Tensor dev_boxes{CUDA};
    Tensor dev_delete_mask{CUDA};
    Tensor host_delete_mask{CPU};
    Tensor dev_list{CUDA};

    dev_boxes.Resize(box_dim * nboxes);
    float* d_sorted_boxes = dev_boxes.template mutable_data<float>();
    dev_list.Resize(nboxes);
    int* d_list = dev_list.template mutable_data<int>();

    // No timing the memcpies because data is already on the GPU in the
    // real-world use case (generated by the GPU generate_proposals)
    CUDA_CHECK(cudaMemcpyAsync(
        d_sorted_boxes,
        &sorted_boxes[0],
        sizeof(*d_sorted_boxes) * box_dim * nboxes,
        cudaMemcpyHostToDevice,
        cuda_context.cuda_stream()));

    // Running ntests runs of CPU NMS
    for (int itest = 0; itest < ntests; ++itest) {
      std::vector<int> keep = utils::nms_cpu(
          eig_proposals,
          eig_scores,
          sorted_indices,
          thresh,
          -1, /* topN */
          true /* legacy_plus_one */);
      int list_nitems;
      utils::nms_gpu(
          d_sorted_boxes,
          nboxes,
          thresh,
          true, /* legacy_plus_one */
          d_list,
          &list_nitems,
          dev_delete_mask,
          host_delete_mask,
          &cuda_context,
          box_dim);
      std::vector<int> gpu_keep(list_nitems);
      CUDA_CHECK(cudaMemcpyAsync(
          &gpu_keep[0],
          d_list,
          list_nitems * sizeof(int),
          cudaMemcpyDeviceToHost,
          cuda_context.cuda_stream()));
      CUDA_CHECK(cudaStreamSynchronize(cuda_context.cuda_stream()));

      ASSERT_EQ(keep.size(), gpu_keep.size());
      std::sort(keep.begin(), keep.end());
      std::sort(gpu_keep.begin(), gpu_keep.end());

      for (int i = 0; i < list_nitems; ++i)
        EXPECT_EQ(keep[i], gpu_keep[i]);
    }
  }
}

TEST(UtilsNMSTest, TestNMSGPURotatedAngle0) {
  if (!HasCudaGPU())
    return;
  const int box_dim = 5;
  // Same boxes in TestNMS with (x_ctr, y_ctr, w, h, angle) format
  std::vector<float> boxes = {30, 35, 41,   51,    0,  29.5, 36,  38,  49,
                              0,  24, 29.5, 33,    42, 0,    125, 120, 51,
                              41, 0,  127,  124.5, 57, 30,   0};

  std::vector<float> scores = {0.5f, 0.7f, 0.6f, 0.9f, 0.8f};

  std::vector<int> indices(scores.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&scores](int lhs, int rhs) {
    return scores[lhs] > scores[rhs];
  });
  std::vector<float> sorted_boxes(boxes.size());
  for (int i = 0; i < scores.size(); ++i) {
    for (int d = 0; d < box_dim; ++d)
      sorted_boxes[i * box_dim + d] = boxes[indices[i] * box_dim + d];
  }

  Workspace ws;
  DeviceOption option;
  option.set_device_type(PROTO_CUDA);
  CUDAContext cuda_context(option);

  Tensor dev_sorted_boxes{CUDA};
  Tensor dev_scores{CUDA};
  Tensor dev_boxes_valid_flags{CUDA};
  Tensor dev_list{CUDA};
  Tensor dev_delete_mask{CUDA};
  Tensor host_delete_mask{CPU};
  Tensor dev_list_nitems{CUDA};
  Tensor host_list{CPU};

  int nboxes = boxes.size() / box_dim;
  dev_sorted_boxes.Resize(box_dim * nboxes);
  dev_list.Resize(nboxes);
  host_list.Resize(nboxes);

  float* d_sorted_boxes = dev_sorted_boxes.template mutable_data<float>();
  int* d_list = dev_list.template mutable_data<int>();
  int* h_list = host_list.template mutable_data<int>();

  CUDA_CHECK(cudaMemcpyAsync(
      d_sorted_boxes,
      &sorted_boxes[0],
      sizeof(*d_sorted_boxes) * box_dim * nboxes,
      cudaMemcpyHostToDevice,
      cuda_context.cuda_stream()));

  std::vector<float> input_thresh{0.1f, 0.3f, 0.5f, 0.8f, 0.9f};
  std::vector<std::set<int>> output_gt{
      {0, 2}, {0, 2}, {0, 2}, {0, 1, 2, 3}, {0, 1, 2, 3, 4}};

  std::vector<int> keep(nboxes);
  std::set<int> keep_as_set;
  for (int itest = 0; itest < input_thresh.size(); ++itest) {
    const float thresh = input_thresh[itest];
    int list_nitems;
    utils::nms_gpu(
        d_sorted_boxes,
        nboxes,
        thresh,
        true, /* legacy_plus_one */
        d_list,
        &list_nitems,
        dev_delete_mask,
        host_delete_mask,
        &cuda_context,
        box_dim);

    cuda_context.FinishDeviceComputation();
    host_list.CopyFrom(dev_list);

    keep_as_set.clear();
    for (int i = 0; i < list_nitems; ++i) {
      keep_as_set.insert(h_list[i]);
    }

    // Sets are sorted
    // sets are equal <=> sets contains the same elements
    EXPECT_TRUE(output_gt[itest] == keep_as_set);
  }

  cuda_context.FinishDeviceComputation();
}

TEST(UtilsNMSTest, TestPerfRotatedNMS) {
  if (!HasCudaGPU())
    return;
  const int box_dim = 5;
  const int nboxes = 2000;

  Workspace ws;
  DeviceOption option;
  option.set_device_type(PROTO_CUDA);
  CUDAContext cuda_context(option);

  Tensor host_boxes{CPU};
  Tensor host_scores{CPU};
  host_boxes.Resize(box_dim * nboxes);
  host_scores.Resize(nboxes);

  float* h_boxes = host_boxes.template mutable_data<float>();
  float* h_scores = host_scores.template mutable_data<float>();

  // Generating random input
  generateRandomRotatedBoxes(h_boxes, h_scores, nboxes);

  Eigen::ArrayXXf proposals(nboxes, box_dim);
  Eigen::ArrayXXf scores(nboxes, 1);
  for (int i = 0; i < nboxes; ++i) {
    for (int d = 0; d < box_dim; ++d)
      proposals(i, d) = h_boxes[box_dim * i + d];
    scores(i, 0) = h_scores[i];
  }

  const int ntests = 10;
  const float thresh = 0.7;
  // Not timing the sort for the CPU
  // in the real-world use case scores already have been sorted earlier in the
  // generate proposals workflow
  std::vector<int> indices(proposals.rows());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(
      indices.data(),
      indices.data() + indices.size(),
      [&scores](int lhs, int rhs) { return scores(lhs) > scores(rhs); });

  // Running ntests runs of CPU NMS
  auto cpu_start = std::chrono::steady_clock::now();
  for (int itest = 0; itest < ntests; ++itest) {
    utils::nms_cpu(
        proposals,
        scores,
        indices,
        thresh,
        -1, /* topN */
        true /* legacy_plus_one */);
  }
  auto cpu_stop = std::chrono::steady_clock::now();

  std::vector<float> sorted_boxes(nboxes * box_dim);
  for (int i = 0; i < scores.size(); ++i) {
    for (int d = 0; d < box_dim; ++d)
      sorted_boxes[i * box_dim + d] = h_boxes[indices[i] * box_dim + d];
  }

  Tensor dev_boxes{CUDA};
  Tensor dev_delete_mask{CUDA};
  Tensor host_delete_mask{CPU};
  Tensor dev_list{CUDA};

  dev_boxes.Resize(box_dim * nboxes);
  float* d_sorted_boxes = dev_boxes.template mutable_data<float>();
  dev_list.Resize(nboxes);
  int* d_list = dev_list.template mutable_data<int>();
  int list_nitems;

  // No timing the memcpies because data is already on the GPU in the real-world
  // use case (generated by the GPU generate_proposals)
  CUDA_CHECK(cudaMemcpyAsync(
      d_sorted_boxes,
      &sorted_boxes[0],
      sizeof(*d_sorted_boxes) * box_dim * nboxes,
      cudaMemcpyHostToDevice,
      cuda_context.cuda_stream()));

  // Running ntests runs of GPU NMS
  auto gpu_start = std::chrono::steady_clock::now();
  for (int itest = 0; itest < ntests; ++itest) {
    utils::nms_gpu(
        d_sorted_boxes,
        nboxes,
        thresh,
        true, /* legacy_plus_one */
        d_list,
        &list_nitems,
        dev_delete_mask,
        host_delete_mask,
        &cuda_context,
        box_dim);
  }
  // Waiting for everything to be done
  CUDA_CHECK(cudaStreamSynchronize(cuda_context.cuda_stream()));
  auto gpu_stop = std::chrono::steady_clock::now();

  double total_cpu_time =
      std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();
  double total_gpu_time =
      std::chrono::duration<double, std::milli>(gpu_stop - gpu_start).count();
  double ratio = total_cpu_time / total_gpu_time;

  double avg_cpu_time = total_cpu_time / ntests;
  double avg_gpu_time = total_gpu_time / ntests;

  printf(
      "RotatedNMS, nproposals=%i, ntests=%i, Avg GPU time = %fms, Avg CPU time = %fms, GPU speed up = %fX \n",
      nboxes,
      ntests,
      avg_gpu_time,
      avg_cpu_time,
      ratio);
}

// Skipped. See https://github.com/pytorch/pytorch/issues/26811
// TEST(UtilsNMSTest, GPUEqualsCPURotatedCorrectnessTest) {
//   if (!HasCudaGPU())
//     return;
//   Workspace ws;
//   DeviceOption option;
//   option.set_device_type(PROTO_CUDA);
//   CUDAContext cuda_context(option);

//   const int box_dim = 5;
//   const std::vector<int> nboxes_vec = {10, 100, 1000, 2000};
//   for (int nboxes : nboxes_vec) {
//     Tensor host_boxes{CPU};
//     Tensor host_scores{CPU};
//     host_boxes.Resize(box_dim * nboxes);
//     host_scores.Resize(nboxes);

//     float* h_boxes = host_boxes.template mutable_data<float>();
//     float* h_scores = host_scores.template mutable_data<float>();

//     // Generating random input
//     generateRandomRotatedBoxes(h_boxes, h_scores, nboxes);

//     const int ntests = 1;
//     const float thresh = 0.7;
//     // Not timing the sort for the CPU
//     // in the real-world use case scores already have been sorted earlier in the
//     // generate proposals workflow
//     std::vector<int> indices(nboxes);
//     std::iota(indices.begin(), indices.end(), 0);
//     std::sort(indices.begin(), indices.end(), [h_scores](int lhs, int rhs) {
//       return h_scores[lhs] > h_scores[rhs];
//     });

//     std::vector<float> sorted_boxes(nboxes * box_dim);
//     std::vector<float> sorted_scores(nboxes);
//     Eigen::ArrayXXf eig_proposals(nboxes, box_dim);
//     Eigen::ArrayXXf eig_scores(nboxes, 1);
//     for (int i = 0; i < nboxes; ++i) {
//       for (int d = 0; d < box_dim; ++d) {
//         sorted_boxes[i * box_dim + d] = h_boxes[indices[i] * box_dim + d];
//         eig_proposals(i, d) = h_boxes[indices[i] * box_dim + d];
//       }
//       sorted_scores[i] = h_scores[indices[i]];
//       eig_scores(i) = h_scores[indices[i]];
//     }
//     std::vector<int> sorted_indices(nboxes);
//     std::iota(sorted_indices.begin(), sorted_indices.end(), 0);

//     Tensor dev_boxes{CUDA};
//     Tensor dev_delete_mask{CUDA};
//     Tensor host_delete_mask{CPU};
//     Tensor dev_list{CUDA};

//     dev_boxes.Resize(box_dim * nboxes);
//     float* d_sorted_boxes = dev_boxes.template mutable_data<float>();
//     dev_list.Resize(nboxes);
//     int* d_list = dev_list.template mutable_data<int>();

//     // No timing the memcpies because data is already on the GPU in the
//     // real-world use case (generated by the GPU generate_proposals)
//     CUDA_CHECK(cudaMemcpyAsync(
//         d_sorted_boxes,
//         &sorted_boxes[0],
//         sizeof(*d_sorted_boxes) * box_dim * nboxes,
//         cudaMemcpyHostToDevice,
//         cuda_context.cuda_stream()));

//     // Running ntests runs of CPU NMS
//     for (int itest = 0; itest < ntests; ++itest) {
//       std::vector<int> keep = utils::nms_cpu(
//           eig_proposals,
//           eig_scores,
//           sorted_indices,
//           thresh,
//           -1, /* topN */
//           true /* legacy_plus_one */);
//       int list_nitems;
//       utils::nms_gpu(
//           d_sorted_boxes,
//           nboxes,
//           thresh,
//           true, /* legacy_plus_one */
//           d_list,
//           &list_nitems,
//           dev_delete_mask,
//           host_delete_mask,
//           &cuda_context,
//           box_dim);
//       std::vector<int> gpu_keep(list_nitems);
//       CUDA_CHECK(cudaMemcpyAsync(
//           &gpu_keep[0],
//           d_list,
//           list_nitems * sizeof(int),
//           cudaMemcpyDeviceToHost,
//           cuda_context.cuda_stream()));
//       CUDA_CHECK(cudaStreamSynchronize(cuda_context.cuda_stream()));

//       ASSERT_EQ(keep.size(), gpu_keep.size());
//       std::sort(keep.begin(), keep.end());
//       std::sort(gpu_keep.begin(), gpu_keep.end());

//       for (int i = 0; i < list_nitems; ++i)
//         EXPECT_EQ(keep[i], gpu_keep[i]);
//     }
//   }
// }

} // namespace caffe2
