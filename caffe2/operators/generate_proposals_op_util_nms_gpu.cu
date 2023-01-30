#include "caffe2/operators/generate_proposals_op_util_nms_gpu.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {
namespace utils {
namespace {
// Helper data structure used locally
struct
#if !defined(USE_ROCM)
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
        const bool legacy_plus_one,
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
            (box.x2 - box.x1 + float(int(legacy_plus_one))) *
            (box.y2 - box.y1 + float(int(legacy_plus_one)));
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
              (j_box.x2 - j_box.x1 + float(int(legacy_plus_one))) *
              (j_box.y2 - j_box.y1 + float(int(legacy_plus_one)));
          const float i_area = shared_i_areas[threadIdx.x];
          // The following code will not be valid with empty boxes
          if (i_area == 0.0f || j_area == 0.0f)
            continue;
          const float xx1 = fmaxf(i_box.x1, j_box.x1);
          const float yy1 = fmaxf(i_box.y1, j_box.y1);
          const float xx2 = fminf(i_box.x2, j_box.x2);
          const float yy2 = fminf(i_box.y2, j_box.y2);

          // fdimf computes the positive difference between xx2+1 and xx1
          const float w = fdimf(xx2 + float(int(legacy_plus_one)), xx1);
          const float h = fdimf(yy2 + float(int(legacy_plus_one)), yy1);
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
    const bool legacy_plus_one,
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
      d_desc_sorted_boxes, N, thresh, legacy_plus_one, mask_ld, d_delete_mask);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  host_delete_mask.Resize(N * mask_ld);
  int* h_delete_mask = host_delete_mask.template mutable_data<int>();

  // Overlapping CPU computes and D2H memcpy
  // both take about the same time
  cudaEvent_t copy_done;
  C10_CUDA_CHECK(cudaEventCreate(&copy_done));
  int nto_copy = std::min(CHUNK_SIZE, N);
  C10_CUDA_CHECK(cudaMemcpyAsync(
      &h_delete_mask[0],
      &d_delete_mask[0],
      nto_copy * mask_ld * sizeof(int),
      cudaMemcpyDeviceToHost,
      context->cuda_stream()));
  C10_CUDA_CHECK(cudaEventRecord(copy_done, context->cuda_stream()));
  int offset = 0;
  std::vector<int> h_keep_sorted_list;
  std::vector<int> rmv(mask_ld, 0);
  while (offset < N) {
    const int ncopied = nto_copy;
    int next_offset = offset + ncopied;
    nto_copy = std::min(CHUNK_SIZE, N - next_offset);
    if (nto_copy > 0) {
      C10_CUDA_CHECK(cudaMemcpyAsync(
          &h_delete_mask[next_offset * mask_ld],
          &d_delete_mask[next_offset * mask_ld],
          nto_copy * mask_ld * sizeof(int),
          cudaMemcpyDeviceToHost,
          context->cuda_stream()));
    }
    // Waiting for previous copy
    C10_CUDA_CHECK(cudaEventSynchronize(copy_done));
    if (nto_copy > 0){
      C10_CUDA_CHECK(cudaEventRecord(copy_done, context->cuda_stream()));
    }
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
  C10_CUDA_CHECK(cudaEventDestroy(copy_done));

  const int nkeep = h_keep_sorted_list.size();
  C10_CUDA_CHECK(cudaMemcpyAsync(
      d_keep_sorted_list,
      &h_keep_sorted_list[0],
      nkeep * sizeof(int),
      cudaMemcpyHostToDevice,
      context->cuda_stream()));

  *h_nkeep = nkeep;
}

namespace {
struct Point {
  float x, y;
};

// Including duplicates based on get_intersection_points()
const int MAX_INTERSECTION_PTS = 12;

__device__ __forceinline__ void get_rotated_vertices(
    const RotatedBox* box,
    Point* pts) {
  constexpr float PI = 3.14159265358979323846;
  float theta = box->a * PI / 180.0;
  float cosTheta = cos(theta);
  float sinTheta = sin(theta);

  float w = box->w;
  float h = box->h;
  float x[4] = {-w / 2, -w / 2, w / 2, w / 2};
  float y[4] = {-h / 2, h / 2, h / 2, -h / 2};

  // y: top --> down; x: left --> right
  for (int i = 0; i < 4; i++) {
    pts[i].x = sinTheta * y[i] + cosTheta * x[i] + box->x_ctr;
    pts[i].y = cosTheta * y[i] - sinTheta * x[i] + box->y_ctr;
  }
}

__device__ __forceinline__ bool is_point_within_rect(
    const Point* pt,
    const Point* rect_pts,
    const Point* rect_lines) {
  // We do a sign test to see on which side the point lies.
  // If the point lies on the same side for all 4 sides of the rect,
  // then it lies within the rectangle.
  int total_sign = 0;
  for (int i = 0; i < 4; ++i) {
    // Line equation: Ax + By + C = 0.
    // See which side of the line this point is at.
    // float causes underflow!
    double A = -rect_lines[i].y;
    double B = rect_lines[i].x;
    double C = -(A * rect_pts[i].x + B * rect_pts[i].y);
    double s = A * pt->x + B * pt->y + C;
    total_sign += (s >= 0.f) ? 1 : -1;
  }
  return (total_sign == 4 || total_sign == -4);
}

__device__ __forceinline__ bool same_rects(
    const Point* pts1,
    const Point* pts2) {
  bool same = true;
  for (int i = 0; i < 4; ++i) {
    same &= (fabs(pts1[i].x - pts2[i].x) <= 1e-5);
    same &= (fabs(pts1[i].y - pts2[i].y) <= 1e-5);
  }
  return same;
}

__device__ __forceinline__ int get_intersection_points(
    const Point* pts1,
    const Point* pts2,
    Point* intersection_pts) {
  // Special case for rect1 == rect2
  if (same_rects(pts1, pts2)) {
    for (int i = 0; i < 4; i++) {
      intersection_pts[i] = pts1[i];
    }
    return 4;
  }

  // Calculate line vectors.
  // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1].
  Point lines1[4], lines2[4];
  for (int i = 0; i < 4; i++) {
    lines1[i].x = pts1[(i + 1) % 4].x - pts1[i].x;
    lines1[i].y = pts1[(i + 1) % 4].y - pts1[i].y;

    lines2[i].x = pts2[(i + 1) % 4].x - pts2[i].x;
    lines2[i].y = pts2[(i + 1) % 4].y - pts2[i].y;
  }

  // Line test - test all line combos for intersection
  int count = 0;
  Point int_pt;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      // Solve for 2x2 Ax=b
      float x21 = pts2[j].x - pts1[i].x;
      float y21 = pts2[j].y - pts1[i].y;

      const Point& l1 = lines1[i];
      const Point& l2 = lines2[j];

      // This takes care of parallel lines
      float det = l2.x * l1.y - l1.x * l2.y;
      if (fabs(det) <= 1e-14) {
        continue;
      }

      float t1 = (l2.x * y21 - l2.y * x21) / det;
      float t2 = (l1.x * y21 - l1.y * x21) / det;

      if (t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f) {
        int_pt.x = pts1[i].x + lines1[i].x * t1;
        int_pt.y = pts1[i].y + lines1[i].y * t1;
        intersection_pts[count++] = int_pt;
      }
    }
  }

  // Check for vertices of rect1 inside rect2
  for (int i = 0; i < 4; i++) {
    if (is_point_within_rect(&pts1[i], pts2, lines2)) {
      intersection_pts[count++] = pts1[i];
    }
  }

  // Reverse the check - check for vertices of rect2 inside rect1
  for (int i = 0; i < 4; i++) {
    if (is_point_within_rect(&pts2[i], pts1, lines1)) {
      intersection_pts[count++] = pts2[i];
    }
  }

  return count;
}

__device__ __forceinline__ void reorder_points(Point* pts, int count) {
  if (count <= 3) {
    return;
  }

  // Compute center point
  Point center{0.f, 0.f};
  for (int i = 0; i < count; i++) {
    center.x += pts[i].x;
    center.y += pts[i].y;
  }
  center.x /= count;
  center.y /= count;

  // Calculate distance of each point from center and store the x component
  Point dist;
  float d;
  float xs[MAX_INTERSECTION_PTS];
  for (int i = 0; i < count; ++i) {
    dist.x = pts[i].x - center.x;
    dist.y = pts[i].y - center.y;
    d = sqrt(dist.x * dist.x + dist.y * dist.y);
    dist.x /= d;
    dist.y /= d;
    xs[i] = (dist.y >= 0) ? dist.x : (-2 - dist.x);
  }

  // Order points based on x component of the distance.
  // Could use thrust::sort_by_key(thrust::seq, xs, xs + count, pts),
  // but it results in a big perf hit.
  float temp_x;
  Point temp_pt;
  int j;
  for (int i = 1; i < count; ++i) {
    if (xs[i - 1] > xs[i]) {
      temp_x = xs[i];
      temp_pt = pts[i];
      j = i;
      while (j > 0 && xs[j - 1] > temp_x) {
        xs[j] = xs[j - 1];
        pts[j] = pts[j - 1];
        j--;
      }
      xs[j] = temp_x;
      pts[j] = temp_pt;
    }
  }
}

__device__ __forceinline__ float
triangle_area(const Point* a, const Point* b, const Point* c) {
  return ((a->x - c->x) * (b->y - c->y) - (a->y - c->y) * (b->x - c->x)) / 2.0;
}

__device__ __forceinline__ float polygon_area(const Point* pts, int count) {
  float area = 0.0;
  for (int i = 1; i < count - 1; ++i) {
    area += fabs(triangle_area(&pts[0], &pts[i], &pts[i + 1]));
  }
  return area;
}

__launch_bounds__(
    CAFFE_CUDA_NUM_THREADS_2D_DIMX* CAFFE_CUDA_NUM_THREADS_2D_DIMY,
    4) __global__
    void RotatedNMSKernel(
        const RotatedBox* d_desc_sorted_boxes,
        const int nboxes,
        const float thresh,
        const int mask_ld,
        int* d_delete_mask) {
  // Storing box areas used by this CUDA block in the shared memory
  __shared__ float shared_i_areas[CAFFE_CUDA_NUM_THREADS_2D_DIMX];
  // Same thing with vertices of boxes
  __shared__ Point shared_i_pts[CAFFE_CUDA_NUM_THREADS_2D_DIMX * 4];

  // The condition of the for loop is common to all threads in the block
  // This is necessary to be able to call __syncthreads() inside of the loop
  for (int i_block_offset = blockIdx.x * blockDim.x; i_block_offset < nboxes;
       i_block_offset += blockDim.x * gridDim.x) {
    const int i_to_load = i_block_offset + threadIdx.x;
    if (i_to_load < nboxes) {
      // One 1D line load the boxes for x-dimension
      if (threadIdx.y == 0) {
        const RotatedBox box = d_desc_sorted_boxes[i_to_load];
        shared_i_areas[threadIdx.x] = box.w * box.h;
        get_rotated_vertices(&box, &shared_i_pts[threadIdx.x * 4]);
      }
    }

    __syncthreads();

    Point intersection_pts[MAX_INTERSECTION_PTS];
    Point j_pts[4];
    const int i = i_block_offset + threadIdx.x;
    for (int j_thread_offset =
             BOXES_PER_THREAD * (blockIdx.y * blockDim.y + threadIdx.y);
         j_thread_offset < nboxes;
         j_thread_offset += BOXES_PER_THREAD * blockDim.y * gridDim.y) {
      int above_thresh = 0;
      bool valid = false;
      for (int ib = 0; ib < BOXES_PER_THREAD; ++ib) {
        // This thread will compare Box i and Box j
        const int j = j_thread_offset + ib;
        if (i < j && i < nboxes && j < nboxes) {
          valid = true;
          const RotatedBox j_box = d_desc_sorted_boxes[j];
          const float j_area = j_box.w * j_box.h;
          const float i_area = shared_i_areas[threadIdx.x];
          // The following code will not be valid with empty boxes
          if (i_area == 0.0f || j_area == 0.0f) {
            continue;
          }

          const Point* i_pts = &shared_i_pts[threadIdx.x * 4];
          get_rotated_vertices(&j_box, j_pts);
          int count = get_intersection_points(i_pts, j_pts, intersection_pts);
          reorder_points(intersection_pts, count);
          const float intersection = polygon_area(intersection_pts, count);

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

void nms_gpu_rotated(
    const float* d_desc_sorted_boxes_float_ptr,
    const int N,
    const float thresh,
    int* d_keep_sorted_list,
    int* h_nkeep,
    TensorCUDA& dev_delete_mask,
    TensorCPU& host_delete_mask,
    CUDAContext* context) {
  // The next kernel expects squares
  CAFFE_ENFORCE_EQ(
      CAFFE_CUDA_NUM_THREADS_2D_DIMX, CAFFE_CUDA_NUM_THREADS_2D_DIMY);

  const int mask_ld = (N + BOXES_PER_THREAD - 1) / BOXES_PER_THREAD;
  const RotatedBox* d_desc_sorted_boxes =
      reinterpret_cast<const RotatedBox*>(d_desc_sorted_boxes_float_ptr);
  dev_delete_mask.Resize(N * mask_ld);
  int* d_delete_mask = dev_delete_mask.template mutable_data<int>();
  RotatedNMSKernel<<<
      CAFFE_GET_BLOCKS_2D(N, mask_ld),
      CAFFE_CUDA_NUM_THREADS_2D,
      0,
      context->cuda_stream()>>>(
      d_desc_sorted_boxes, N, thresh, mask_ld, d_delete_mask);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  host_delete_mask.Resize(N * mask_ld);
  int* h_delete_mask = host_delete_mask.template mutable_data<int>();

  // Overlapping CPU computes and D2H memcpy
  // both take about the same time
  cudaEvent_t copy_done;
  C10_CUDA_CHECK(cudaEventCreate(&copy_done));
  int nto_copy = std::min(CHUNK_SIZE, N);
  C10_CUDA_CHECK(cudaMemcpyAsync(
      &h_delete_mask[0],
      &d_delete_mask[0],
      nto_copy * mask_ld * sizeof(int),
      cudaMemcpyDeviceToHost,
      context->cuda_stream()));
  C10_CUDA_CHECK(cudaEventRecord(copy_done, context->cuda_stream()));
  int offset = 0;
  std::vector<int> h_keep_sorted_list;
  std::vector<int> rmv(mask_ld, 0);
  while (offset < N) {
    const int ncopied = nto_copy;
    int next_offset = offset + ncopied;
    nto_copy = std::min(CHUNK_SIZE, N - next_offset);
    if (nto_copy > 0) {
      C10_CUDA_CHECK(cudaMemcpyAsync(
          &h_delete_mask[next_offset * mask_ld],
          &d_delete_mask[next_offset * mask_ld],
          nto_copy * mask_ld * sizeof(int),
          cudaMemcpyDeviceToHost,
          context->cuda_stream()));
    }
    // Waiting for previous copy
    C10_CUDA_CHECK(cudaEventSynchronize(copy_done));
    if (nto_copy > 0){
      C10_CUDA_CHECK(cudaEventRecord(copy_done, context->cuda_stream()));
    }
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
  C10_CUDA_CHECK(cudaEventDestroy(copy_done));

  const int nkeep = h_keep_sorted_list.size();
  C10_CUDA_CHECK(cudaMemcpyAsync(
      d_keep_sorted_list,
      &h_keep_sorted_list[0],
      nkeep * sizeof(int),
      cudaMemcpyHostToDevice,
      context->cuda_stream()));

  *h_nkeep = nkeep;
}

void nms_gpu(
    const float* d_desc_sorted_boxes,
    const int N,
    const float thresh,
    const bool legacy_plus_one,
    int* d_keep_sorted_list,
    int* h_nkeep,
    TensorCUDA& dev_delete_mask,
    TensorCPU& host_delete_mask,
    CUDAContext* context,
    const int box_dim) {
  CAFFE_ENFORCE(box_dim == 4 || box_dim == 5);
  if (box_dim == 4) {
    nms_gpu_upright(
        d_desc_sorted_boxes,
        N,
        thresh,
        legacy_plus_one,
        d_keep_sorted_list,
        h_nkeep,
        dev_delete_mask,
        host_delete_mask,
        context);
  } else {
    nms_gpu_rotated(
        d_desc_sorted_boxes,
        N,
        thresh,
        d_keep_sorted_list,
        h_nkeep,
        dev_delete_mask,
        host_delete_mask,
        context);
  }
}
} // namespace utils
} // namespace caffe2
