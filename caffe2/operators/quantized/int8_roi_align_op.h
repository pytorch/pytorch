#ifndef CAFFE2_OPERATORS_INT8_ROI_ALIGN_OP_H_
#define CAFFE2_OPERATORS_INT8_ROI_ALIGN_OP_H_

#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/operator_schema.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/quantized/int8_utils.h"
#include "caffe2/utils/math.h"
#include <c10/util/irange.h>

namespace caffe2 {

namespace int8 {

namespace {

struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  uint8_t w1;
  uint8_t w2;
  uint8_t w3;
  uint8_t w4;
};

void pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int iy_upper,
    const int ix_upper,
    float roi_start_h,
    float roi_start_w,
    float bin_size_h,
    float bin_size_w,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    std::vector<PreCalc>& pre_calc) {
  int pre_calc_index = 0;
  // boltnn use a smaller multiplier here. Sometimes w will shrink to 0.
  const float w_multiplier = 255.0;
  for (const auto ph : c10::irange(pooled_height)) {
    for (const auto pw : c10::irange(pooled_width)) {
      for (const auto iy : c10::irange(iy_upper)) {
        const float yy = roi_start_h + ph * bin_size_h +
            static_cast<float>(iy + .5f) * bin_size_h /
                static_cast<float>(roi_bin_grid_h); // e.g., 0.5, 1.5
        for (const auto ix : c10::irange(ix_upper)) {
          const float xx = roi_start_w + pw * bin_size_w +
              static_cast<float>(ix + .5f) * bin_size_w /
                  static_cast<float>(roi_bin_grid_w);

          float x = xx;
          float y = yy;
          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc pc;
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
            y = (float)y_low;
          } else {
            y_high = y_low + 1;
          }

          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (float)x_low;
          } else {
            x_high = x_low + 1;
          }

          float ly = y - y_low;
          float lx = x - x_low;
          float hy = 1. - ly, hx = 1. - lx;
          // w are not necessary 1
          uint8_t w1 = static_cast<uint8_t>(Round(hy * hx * w_multiplier));
          uint8_t w2 = static_cast<uint8_t>(Round(hy * lx * w_multiplier));
          uint8_t w3 = static_cast<uint8_t>(Round(ly * hx * w_multiplier));
          uint8_t w4 = static_cast<uint8_t>(Round(ly * lx * w_multiplier));

          // save weights and indices
          PreCalc pc;
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

void ROIAlignForward(
    const int nthreads,
    const uint8_t* bottom_data,
    const float& spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const float* bottom_rois,
    int roi_cols,
    uint8_t* top_data,
    const float x_scale,
    const float y_scale,
    const int32_t x_offset,
    const int32_t y_offset,
    StorageOrder order /* unused */,
    bool continuous_coordinate) {
  DCHECK(roi_cols == 4 || roi_cols == 5);

  int n_rois = nthreads / channels / pooled_width / pooled_height;

  for (const auto n : c10::irange(n_rois)) {
    int index_n = n * channels * pooled_width * pooled_height;

    // roi could have 4 or 5 columns
    const float* offset_bottom_rois = bottom_rois + n * roi_cols;
    int roi_batch_ind = 0;
    if (roi_cols == 5) {
      roi_batch_ind = offset_bottom_rois[0];
      offset_bottom_rois++;
    }

    // Do not using rounding; this implementation detail is critical
    float roi_offset = continuous_coordinate ? 0.5 : 0;
    float roi_start_w = offset_bottom_rois[0] * spatial_scale - roi_offset;
    float roi_start_h = offset_bottom_rois[1] * spatial_scale - roi_offset;
    float roi_end_w = offset_bottom_rois[2] * spatial_scale - roi_offset;
    float roi_end_h = offset_bottom_rois[3] * spatial_scale - roi_offset;

    float roi_width = roi_end_w - roi_start_w;
    float roi_height = roi_end_h - roi_start_h;
    if (continuous_coordinate) {
      CAFFE_ENFORCE(
          roi_width >= 0 && roi_height >= 0,
          "ROIs in ROIAlign do not have non-negative size!");
    } else { // backward compatibility
      // Force malformed ROIs to be 1x1
      roi_width = std::max(roi_width, (float)1.);
      roi_height = std::max(roi_height, (float)1.);
    }
    float bin_size_h =
        static_cast<float>(roi_height) / static_cast<float>(pooled_height);
    float bin_size_w =
        static_cast<float>(roi_width) / static_cast<float>(pooled_width);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const float count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    // calculate multiplier
    double real_multiplier = x_scale / (y_scale * 255.0 * count);
    int32_t Y_multiplier;
    int Y_shift;
    QuantizeMultiplierSmallerThanOne(real_multiplier, &Y_multiplier, &Y_shift);

    // we want to precalculate indices and weights shared by all channels,
    // this is the key point of optimization
    std::vector<PreCalc> pre_calc(
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

    const uint8_t* offset_bottom_data =
        bottom_data + roi_batch_ind * channels * height * width;
    int pre_calc_index = 0;
    for (const auto ph : c10::irange(pooled_height)) {
      for (const auto pw : c10::irange(pooled_width)) {
        vector<int32_t> acc_buffer(channels, 0);

        for (C10_UNUSED const auto iy : c10::irange(roi_bin_grid_h)) {
          for (C10_UNUSED const auto ix : c10::irange(roi_bin_grid_w)) {
            PreCalc pc = pre_calc[pre_calc_index];

            const uint8_t* data_1 = offset_bottom_data + channels * pc.pos1;
            const uint8_t* data_2 = offset_bottom_data + channels * pc.pos2;
            const uint8_t* data_3 = offset_bottom_data + channels * pc.pos3;
            const uint8_t* data_4 = offset_bottom_data + channels * pc.pos4;
            for (const auto c : c10::irange(channels)) {
              acc_buffer[c] += (uint32_t)(pc.w1) * (uint32_t)(data_1[c]);
              acc_buffer[c] += (uint32_t)(pc.w2) * (uint32_t)(data_2[c]);
              acc_buffer[c] += (uint32_t)(pc.w3) * (uint32_t)(data_3[c]);
              acc_buffer[c] += (uint32_t)(pc.w4) * (uint32_t)(data_4[c]);

              // w_1..4 are all multiplied by 255.0
              acc_buffer[c] -= x_offset * 255.0;
            }

            pre_calc_index += 1;
          }
        }
        int index_nhw = index_n + (ph * pooled_width + pw) * channels;
        uint8_t* out_ptr = top_data + index_nhw;
        for (const auto c : c10::irange(channels)) {
          int32_t a_mul = MultiplyByQuantizedMultiplierSmallerThanOne(
                              acc_buffer[c], Y_multiplier, Y_shift) +
              y_offset;
          int32_t clamped_a =
              std::min<int32_t>(255, std::max<int32_t>(0, a_mul));
          out_ptr[c] = static_cast<uint8_t>(clamped_a);
        }
      } // for pw
    } // for ph
  } // for n
}

} // namespace

class Int8RoIAlignOp final : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit Int8RoIAlignOp(Args&&... args)
      : Operator<CPUContext>(std::forward<Args>(args)...),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NHWC"))),
        spatial_scale_(
            this->template GetSingleArgument<float>("spatial_scale", 1.)),
        pooled_height_(this->template GetSingleArgument<int>("pooled_h", 1)),
        pooled_width_(this->template GetSingleArgument<int>("pooled_w", 1)),
        sampling_ratio_(
            this->template GetSingleArgument<int>("sampling_ratio", -1)),
        aligned_(this->template GetSingleArgument<bool>("aligned", false)) {
    TORCH_DCHECK_GT(spatial_scale_, 0);
    TORCH_DCHECK_GT(pooled_height_, 0);
    TORCH_DCHECK_GT(pooled_width_, 0);
    TORCH_DCHECK_GE(sampling_ratio_, 0);
    // only supports NHWC
    CAFFE_ENFORCE(order_ == StorageOrder::NHWC);
  }

  bool RunOnDevice() override {
    const auto& X = Inputs()[0]->template Get<Int8TensorCPU>(); // Input, NHWC
    auto& R = Input(1); // RoIs
    auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>(); // RoI pooled
    // calculate multiplier
    int32_t Y_offset = this->template GetSingleArgument<int>("Y_zero_point", 0);
    auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
    Y->scale = Y_scale;
    Y->zero_point = Y_offset;

    if (R.numel() == 0) {
      // Handle empty rois
      Y->t.Resize(0, pooled_height_, pooled_width_, X.t.dim32(3));
      // The following mutable_data calls are needed to allocate the tensors
      Y->t.mutable_data<uint8_t>();
      return true;
    }

    CAFFE_ENFORCE_EQ(R.dim(), 2);
    // if R has 5 columns, the first column is the index, otherwise 0
    CAFFE_ENFORCE(R.dim32(1) == 4 || R.dim32(1) == 5);

    assert(sampling_ratio_ >= 0);

    // only supports NHWC now
    ReinitializeTensor(
        &Y->t,
        {R.dim32(0), pooled_height_, pooled_width_, X.t.dim32(3)},
        at::dtype<uint8_t>().device(CPU));
    int output_size = Y->t.numel();

    ROIAlignForward(
        output_size,
        X.t.data<uint8_t>(),
        spatial_scale_,
        X.t.dim32(3),
        X.t.dim32(1),
        X.t.dim32(2),
        pooled_height_,
        pooled_width_,
        sampling_ratio_,
        R.data<float>(),
        R.dim32(1),
        Y->t.mutable_data<uint8_t>(),
        X.scale,
        Y_scale,
        X.zero_point,
        Y_offset,
        order_,
        aligned_);

    return true;
  }

 protected:
  StorageOrder order_;
  float spatial_scale_;
  int pooled_height_;
  int pooled_width_;
  int sampling_ratio_;
  bool aligned_;
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_ROI_ALIGN_OP_H_
