#ifdef _MSC_VER
#define _USE_MATH_DEFINES // For M_PI
#endif // _MSC_VER
#include <cmath>

#include "caffe2/operators/roi_align_rotated_op.h"

#include <stdio.h>
#include <cfloat>
#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T>
__device__ T bilinear_interpolate(
    const T* bottom_data,
    const int height,
    const int width,
    T y,
    T x,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
__global__ void RoIAlignRotatedForward(
    const int nthreads,
    const T* bottom_data,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const T* bottom_rois,
    T* top_data,
    bool continuous_coordinate) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not round
    T roi_offset = continuous_coordinate ? T(0.5) : 0;
    T roi_center_w = offset_bottom_rois[1] * spatial_scale - roi_offset;
    T roi_center_h = offset_bottom_rois[2] * spatial_scale - roi_offset;
    T roi_width = offset_bottom_rois[3] * spatial_scale;
    T roi_height = offset_bottom_rois[4] * spatial_scale;
    T theta = offset_bottom_rois[5] * M_PI / 180.0;

    if (!continuous_coordinate) { // backward compatibility
      // Force malformed ROIs to be 1x1
      roi_width = c10::cuda::compat::max(roi_width, (T)1.);
      roi_height = c10::cuda::compat::max(roi_height, (T)1.);
    }
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    T roi_start_h = -roi_height / 2.0;
    T roi_start_w = -roi_width / 2.0;
    T cosTheta = cos(theta);
    T sinTheta = sin(theta);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T yy = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T xx = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        // Rotate by theta around the center and translate
        T x = xx * cosTheta + yy * sinTheta + roi_center_w;
        T y = yy * cosTheta - xx * sinTheta + roi_center_h;

        T val = bilinear_interpolate(
            offset_bottom_data, height, width, y, x, index);
        output_val += val;
      }
    }
    output_val /= count;

    top_data[index] = output_val;
  }
}

} // namespace

template <>
C10_EXPORT bool RoIAlignRotatedOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0); // Input data to pool
  auto& R = Input(1); // RoIs

  CAFFE_ENFORCE_EQ(order_, StorageOrder::NCHW, "RoIAlign CUDA impl needs NCHW");

  if (R.numel() == 0) {
    // Handle empty rois
    Output(
        0,
        {0, X.dim32(1), pooled_height_, pooled_width_},
        at::dtype<float>()); // RoI pooled data
    return true;
  }

  CAFFE_ENFORCE_EQ(R.dim(), 2);
  CAFFE_ENFORCE_EQ(R.dim32(1), 6);

  assert(sampling_ratio_ >= 0);

  auto* Y = Output(
      0,
      {R.dim32(0), X.dim32(1), pooled_height_, pooled_width_},
      at::dtype<float>()); // RoI pooled data

  int output_size = Y->numel();
  RoIAlignRotatedForward<float>
      <<<CAFFE_GET_BLOCKS(output_size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          output_size,
          X.data<float>(),
          spatial_scale_,
          X.dim32(1),
          X.dim32(2),
          X.dim32(3),
          pooled_height_,
          pooled_width_,
          sampling_ratio_,
          R.data<float>(),
          Y->mutable_data<float>(),
          aligned_);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(RoIAlignRotated, RoIAlignRotatedOp<float, CUDAContext>);
} // namespace caffe2

using RoIAlignRotatedOpFloatCUDA =
    caffe2::RoIAlignRotatedOp<float, caffe2::CUDAContext>;

C10_EXPORT_CAFFE2_OP_TO_C10_CUDA(RoIAlignRotated, RoIAlignRotatedOpFloatCUDA);
