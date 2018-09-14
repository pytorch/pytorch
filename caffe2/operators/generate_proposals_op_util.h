#ifndef CAFFE2_OPERATORS_GENERATE_PROPOSALS_OP_UTIL_H_
#define CAFFE2_OPERATORS_GENERATE_PROPOSALS_OP_UTIL_H_

#include "caffe2/core/tensor.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

namespace utils {
// Generate a list of bounding box shapes for each pixel based on predefined
//     bounding box shapes 'anchors'.
// anchors: predefined anchors, size(A, 4)
// Return: all_anchors_vec: (H * W, A * 4)
// Need to reshape to (H * W * A, 4) to match the format in python
CAFFE2_API ERMatXf ComputeAllAnchors(
    const TensorCPU& anchors,
    int height,
    int width,
    float feat_stride);


} // namespace utils

} // namespace caffe2

#endif // CAFFE2_OPERATORS_GENERATE_PROPOSALS_OP_UTIL_H_
