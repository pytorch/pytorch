#include "roi_align_op.h"

#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace {

template <typename T>
struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  T w1;
  T w2;
  T w3;
  T w4;
};

template <typename T>
void pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int iy_upper,
    const int ix_upper,
    T roi_start_h,
    T roi_start_w,
    T bin_size_h,
    T bin_size_w,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    std::vector<PreCalc<T>>& pre_calc) {
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < iy_upper; iy++) {
        const T yy = roi_start_h + ph * bin_size_h +
            static_cast<T>(iy + .5f) * bin_size_h /
                static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
        for (int ix = 0; ix < ix_upper; ix++) {
          const T xx = roi_start_w + pw * bin_size_w +
              static_cast<T>(ix + .5f) * bin_size_w /
                  static_cast<T>(roi_bin_grid_w);

          T x = xx;
          T y = yy;
          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc<T> pc;
            pc.pos1 = 0;
            pc.pos2 = 0;
            pc.pos3 = 0;
            pc.pos4 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc[pre_calc_index] = pc;
            pre_calc_index += 1;
            continue;
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
          T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          // save weights and indeces
          PreCalc<T> pc;
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc[pre_calc_index] = pc;

          pre_calc_index += 1;
        }
      }
    }
  }
}

template <typename T>
void ROIAlignForward(
    const int nthreads,
    const T* bottom_data,
    const T& spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const T* bottom_rois,
    int roi_cols,
    T* top_data,
    StorageOrder order) {
  DCHECK(roi_cols == 4 || roi_cols == 5);

  int n_rois = nthreads / channels / pooled_width / pooled_height;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int n = 0; n < n_rois; n++) {
    int index_n = n * channels * pooled_width * pooled_height;

    // roi could have 4 or 5 columns
    const T* offset_bottom_rois = bottom_rois + n * roi_cols;
    int roi_batch_ind = 0;
    if (roi_cols == 5) {
      roi_batch_ind = offset_bottom_rois[0];
      offset_bottom_rois++;
    }

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[0] * spatial_scale;
    T roi_start_h = offset_bottom_rois[1] * spatial_scale;
    T roi_end_w = offset_bottom_rois[2] * spatial_scale;
    T roi_end_h = offset_bottom_rois[3] * spatial_scale;
    // T roi_start_w = round(offset_bottom_rois[0] * spatial_scale);
    // T roi_start_h = round(offset_bottom_rois[1] * spatial_scale);
    // T roi_end_w = round(offset_bottom_rois[2] * spatial_scale);
    // T roi_end_h = round(offset_bottom_rois[3] * spatial_scale);

    // Force malformed ROIs to be 1x1
    T roi_width = std::max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    // we want to precalculate indeces and weights shared by all chanels,
    // this is the key point of optimiation
    std::vector<PreCalc<T>> pre_calc(
        roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
    pre_calc_for_bilinear_interpolate(
        height,
        width,
        pooled_height,
        pooled_width,
        roi_bin_grid_h,
        roi_bin_grid_w,
        roi_start_h,
        roi_start_w,
        bin_size_h,
        bin_size_w,
        roi_bin_grid_h,
        roi_bin_grid_w,
        pre_calc);

    if (order == StorageOrder::NCHW) {
      for (int c = 0; c < channels; c++) {
        int index_n_c = index_n + c * pooled_width * pooled_height;
        const T* offset_bottom_data =
            bottom_data + (roi_batch_ind * channels + c) * height * width;
        int pre_calc_index = 0;

        for (int ph = 0; ph < pooled_height; ph++) {
          for (int pw = 0; pw < pooled_width; pw++) {
            int index = index_n_c + ph * pooled_width + pw;

            T output_val = 0.;
            for (int iy = 0; iy < roi_bin_grid_h; iy++) {
              for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                PreCalc<T> pc = pre_calc[pre_calc_index];
                output_val += pc.w1 * offset_bottom_data[pc.pos1] +
                    pc.w2 * offset_bottom_data[pc.pos2] +
                    pc.w3 * offset_bottom_data[pc.pos3] +
                    pc.w4 * offset_bottom_data[pc.pos4];

                pre_calc_index += 1;
              }
            }
            output_val /= count;

            top_data[index] = output_val;
          } // for pw
        } // for ph
      } // for c
    } // if nchw

    if (order == StorageOrder::NHWC) {
      const T* offset_bottom_data =
          bottom_data + roi_batch_ind * channels * height * width;
      int pre_calc_index = 0;

      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          EVecXf output_vals = EVecXf::Zero(channels);

          for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
              PreCalc<T> pc = pre_calc[pre_calc_index];

              ConstEigenVectorMap<T> data_1(
                  offset_bottom_data + channels * pc.pos1, channels);
              ConstEigenVectorMap<T> data_2(
                  offset_bottom_data + channels * pc.pos2, channels);
              ConstEigenVectorMap<T> data_3(
                  offset_bottom_data + channels * pc.pos3, channels);
              ConstEigenVectorMap<T> data_4(
                  offset_bottom_data + channels * pc.pos4, channels);

              output_vals += pc.w1 * data_1 + pc.w2 * data_2 + pc.w3 * data_3 +
                  pc.w4 * data_4;

              pre_calc_index += 1;
            }
          }
          output_vals /= count;

          int index_nhw = index_n + (ph * pooled_width + pw) * channels;
          std::memcpy(
              top_data + index_nhw, output_vals.data(), channels * sizeof(T));
        } // for pw
      } // for ph
    } // if nhwc

  } // for n
}

} // namespace

template <>
bool RoIAlignOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0); // Input data to pool, NCHW
  auto& R = Input(1); // RoIs
  auto* Y = Output(0); // RoI pooled data

  if (R.numel() == 0) {
    // Handle empty rois
    if (order_ == StorageOrder::NCHW) {
      Y->Resize(0, X.dim32(1), pooled_height_, pooled_width_);
    } else if (order_ == StorageOrder::NHWC) {
      Y->Resize(0, pooled_height_, pooled_width_, X.dim32(3));
    }
    // The following mutable_data calls are needed to allocate the tensors
    Y->template mutable_data<float>();
    return true;
  }

  CAFFE_ENFORCE_EQ(R.dim(), 2);
  // if R has 5 columns, the first column is the index, otherwise 0
  CAFFE_ENFORCE(R.dim32(1) == 4 || R.dim32(1) == 5);

  assert(sampling_ratio_ >= 0);

  if (order_ == StorageOrder::NCHW) {
    Y->Resize(R.dim32(0), X.dim32(1), pooled_height_, pooled_width_);
    int output_size = Y->numel();
    ROIAlignForward<float>(
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
        R.dim32(1),
        Y->template mutable_data<float>(),
        order_);
  } else if (order_ == StorageOrder::NHWC) {
    Y->Resize(R.dim32(0), pooled_height_, pooled_width_, X.dim32(3));
    int output_size = Y->numel();
    ROIAlignForward<float>(
        output_size,
        X.data<float>(),
        spatial_scale_,
        X.dim32(3),
        X.dim32(1),
        X.dim32(2),
        pooled_height_,
        pooled_width_,
        sampling_ratio_,
        R.data<float>(),
        R.dim32(1),
        Y->template mutable_data<float>(),
        order_);
  }

  return true;
}

REGISTER_CPU_OPERATOR(RoIAlign, RoIAlignOp<float, CPUContext>);

// Input: X, rois; Output: Y
OPERATOR_SCHEMA(RoIAlign)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Region of Interest (RoI) align operation as used in Mask R-CNN.
)DOC")
    .Arg(
        "spatial_scale",
        "(float) default 1.0; Spatial scale of the input feature map X "
        "relative to the input image. E.g., 0.0625 if X has a stride of 16 "
        "w.r.t. the input image.")
    .Arg("pooled_h", "(int) default 1; Pooled output Y's height.")
    .Arg("pooled_w", "(int) default 1; Pooled output Y's width.")
    .Arg(
        "sampling_ratio",
        "(int) default -1; number of sampling points in the interpolation grid "
        "used to compute the output value of each pooled output bin. If > 0, "
        "then exactly sampling_ratio x sampling_ratio grid points are used. If "
        "<= 0, then an adaptive number of grid points are used (computed as "
        "ceil(roi_width / pooled_w), and likewise for height).")
    .Input(0, "X", "4D feature map input of shape (N, C, H, W).")
    .Input(
        1,
        "RoIs",
        "2D input of shape (R, 4 or 5) specifying R RoIs "
        "representing: batch index in [0, N - 1], x1, y1, x2, y2. The RoI "
        "coordinates are in the coordinate system of the input image. For "
        "inputs corresponding to a single image, batch index can be excluded "
        "to have just 4 columns.")
    .Output(
        0,
        "Y",
        "4D output of shape (R, C, pooled_h, pooled_w). The r-th batch element "
        "is a pooled feature map cooresponding to the r-th RoI.");

} // namespace caffe2
