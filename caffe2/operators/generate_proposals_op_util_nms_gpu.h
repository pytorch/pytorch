#ifndef CAFFE2_OPERATORS_UTILS_NMS_GPU_H_
#define CAFFE2_OPERATORS_UTILS_NMS_GPU_H_

#include <vector>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {
namespace utils {

// Computes Non-Maximum Suppression on the GPU
// Reject a bounding box if its region has an intersection-overunion (IoU)
//    overlap with a higher scoring selected bounding box larger than a
//    threshold.
//
// d_desc_sorted_boxes : pixel coordinates of proposed bounding boxes
//    size: (N,4), format: [x1; y1; x2; y2]
//    the boxes are sorted by scores in descending order
// N : number of boxes
// d_keep_sorted_list : row indices of the selected proposals, sorted by score
// h_nkeep  : number of selected proposals
// dev_delete_mask, host_delete_mask : Tensors that will be used as temp storage
// by NMS
//    Those tensors will be resized to the necessary size
// context : current CUDA context
CAFFE2_API void nms_gpu_upright(
    const float* d_desc_sorted_boxes,
    const int N,
    const float thresh,
    int* d_keep_sorted_list,
    int* h_nkeep,
    TensorCUDA& dev_delete_mask,
    TensorCPU& host_delete_mask,
    CUDAContext* context);

} // namespace utils
} // namespace caffe2

#endif // CAFFE2_OPERATORS_UTILS_NMS_GPU_H_
