#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/generate_proposals_op_util_nms_gpu.h"

namespace caffe2 {
namespace utils {
namespace {
// Helper data structure used locally
struct
#ifndef __HIP_PLATFORM_HCC__
    __align__(16)
#endif
        Box {
  float x1, y1, x2, y2;
};

#define BOXES_PER_THREAD (8 * sizeof(int))
#define CHUNK_SIZE 2000

const dim3 CAFFE_CUDA_NUM_THREADS_2D = {
    static_cast<unsigned int>(CAFFE_CUDA_NUM_THREADS_2D_DIMX),
    static_cast<unsigned int>(CAFFE_CUDA_NUM_THREADS_2D_DIMY),
    1u};

__launch_bounds__(
    CAFFE_CUDA_NUM_THREADS_2D_DIMX* CAFFE_CUDA_NUM_THREADS_2D_DIMY,
    4) __global__
    void NMSKernel(
        const Box* d_desc_sorted_boxes,
        const int nboxes,
        const float thresh,
        const int mask_ld,
        int* d_delete_mask) {
  // Storing boxes used by this CUDA block in the shared memory
  __shared__ Box shared_i_boxes[CAFFE_CUDA_NUM_THREADS_2D_DIMX];
  // Same thing with areas
  __shared__ float shared_i_areas[CAFFE_CUDA_NUM_THREADS_2D_DIMX];
  // The condition of the for loop is common to all threads in the block
  // This is necessary to be able to call __syncthreads() inside of the loop
  for (int i_block_offset = blockIdx.x * blockDim.x; i_block_offset < nboxes;
       i_block_offset += blockDim.x * gridDim.x) {
    const int i_to_load = i_block_offset + threadIdx.x;
    if (i_to_load < nboxes) {
      // One 1D line load the boxes for x-dimension
      if (threadIdx.y == 0) {
        const Box box = d_desc_sorted_boxes[i_to_load];
        shared_i_areas[threadIdx.x] =
            (box.x2 - box.x1 + 1.0f) * (box.y2 - box.y1 + 1.0f);
        shared_i_boxes[threadIdx.x] = box;
      }
    }
    __syncthreads();
    const int i = i_block_offset + threadIdx.x;
    for (int j_thread_offset =
             BOXES_PER_THREAD * (blockIdx.y * blockDim.y + threadIdx.y);
         j_thread_offset < nboxes;
         j_thread_offset += BOXES_PER_THREAD * blockDim.y * gridDim.y) {
      // Note : We can do everything using multiplication,
      // and use fp16 - we are comparing against a low precision
      // threshold
      int above_thresh = 0;
      bool valid = false;
      for (int ib = 0; ib < BOXES_PER_THREAD; ++ib) {
        // This thread will compare Box i and Box j
        const int j = j_thread_offset + ib;
        if (i < j && i < nboxes && j < nboxes) {
          valid = true;
          const Box j_box = d_desc_sorted_boxes[j];
          const Box i_box = shared_i_boxes[threadIdx.x];
          const float j_area =
              (j_box.x2 - j_box.x1 + 1.0f) * (j_box.y2 - j_box.y1 + 1.0f);
          const float i_area = shared_i_areas[threadIdx.x];
          // The following code will not be valid with empty boxes
          if (i_area == 0.0f || j_area == 0.0f)
            continue;
          const float xx1 = fmaxf(i_box.x1, j_box.x1);
          const float yy1 = fmaxf(i_box.y1, j_box.y1);
          const float xx2 = fminf(i_box.x2, j_box.x2);
          const float yy2 = fminf(i_box.y2, j_box.y2);

          // fdimf computes the positive difference between xx2+1 and xx1
          const float w = fdimf(xx2 + 1.0f, xx1);
          const float h = fdimf(yy2 + 1.0f, yy1);
          const float intersection = w * h;

          // Testing for a/b > t
          // eq with a > b*t (b is !=0)
          // avoiding divisions
          const float a = intersection;
          const float b = i_area + j_area - intersection;
          const float bt = b * thresh;
          // eq. to if ovr > thresh
          if (a > bt) {
            // we have score[j] <= score[i]
            above_thresh |= (1U << ib);
          }
        }
      }
      if (valid)
        d_delete_mask[i * mask_ld + j_thread_offset / BOXES_PER_THREAD] =
            above_thresh;
    }
    __syncthreads(); // making sure everyone is done reading smem
  }
}
} // namespace

void nms_gpu_upright(
    const float* d_desc_sorted_boxes_float_ptr,
    const int N,
    const float thresh,
    int* d_keep_sorted_list,
    int* h_nkeep,
    TensorCUDA& dev_delete_mask,
    TensorCPU& host_delete_mask,
    CUDAContext* context) {
  // Making sure we respect the __align(16)__ we promised to the compiler
  auto iptr = reinterpret_cast<std::uintptr_t>(d_desc_sorted_boxes_float_ptr);
  CAFFE_ENFORCE_EQ(iptr % 16, 0);

  // The next kernel expects squares
  CAFFE_ENFORCE_EQ(
      CAFFE_CUDA_NUM_THREADS_2D_DIMX, CAFFE_CUDA_NUM_THREADS_2D_DIMY);

  const int mask_ld = (N + BOXES_PER_THREAD - 1) / BOXES_PER_THREAD;
  const Box* d_desc_sorted_boxes =
      reinterpret_cast<const Box*>(d_desc_sorted_boxes_float_ptr);
  dev_delete_mask.Resize(N * mask_ld);
  int* d_delete_mask = dev_delete_mask.template mutable_data<int>();
  NMSKernel<<<
      CAFFE_GET_BLOCKS_2D(N, mask_ld),
      CAFFE_CUDA_NUM_THREADS_2D,
      0,
      context->cuda_stream()>>>(
      d_desc_sorted_boxes, N, thresh, mask_ld, d_delete_mask);

  host_delete_mask.Resize(N * mask_ld);
  int* h_delete_mask = host_delete_mask.template mutable_data<int>();

  // Overlapping CPU computes and D2H memcpy
  // both take about the same time
  cudaEvent_t copy_done;
  cudaEventCreate(&copy_done);
  int nto_copy = std::min(CHUNK_SIZE, N);
  CUDA_CHECK(cudaMemcpyAsync(
      &h_delete_mask[0],
      &d_delete_mask[0],
      nto_copy * mask_ld * sizeof(int),
      cudaMemcpyDeviceToHost,
      context->cuda_stream()));
  CUDA_CHECK(cudaEventRecord(copy_done, context->cuda_stream()));
  int offset = 0;
  std::vector<int> h_keep_sorted_list;
  std::vector<int> rmv(mask_ld, 0);
  while (offset < N) {
    const int ncopied = nto_copy;
    int next_offset = offset + ncopied;
    nto_copy = std::min(CHUNK_SIZE, N - next_offset);
    if (nto_copy > 0) {
      CUDA_CHECK(cudaMemcpyAsync(
          &h_delete_mask[next_offset * mask_ld],
          &d_delete_mask[next_offset * mask_ld],
          nto_copy * mask_ld * sizeof(int),
          cudaMemcpyDeviceToHost,
          context->cuda_stream()));
    }
    // Waiting for previous copy
    CUDA_CHECK(cudaEventSynchronize(copy_done));
    if (nto_copy > 0)
      cudaEventRecord(copy_done, context->cuda_stream());
    for (int i = offset; i < next_offset; ++i) {
      int iblock = i / BOXES_PER_THREAD;
      int inblock = i % BOXES_PER_THREAD;
      if (!(rmv[iblock] & (1 << inblock))) {
        h_keep_sorted_list.push_back(i);
        int* p = &h_delete_mask[i * mask_ld];
        for (int ib = 0; ib < mask_ld; ++ib) {
          rmv[ib] |= p[ib];
        }
      }
    }
    offset = next_offset;
  }
  cudaEventDestroy(copy_done);

  const int nkeep = h_keep_sorted_list.size();
  cudaMemcpyAsync(
      d_keep_sorted_list,
      &h_keep_sorted_list[0],
      nkeep * sizeof(int),
      cudaMemcpyHostToDevice,
      context->cuda_stream());

  *h_nkeep = nkeep;
}
} // namespace utils
} // namespace caffe2
