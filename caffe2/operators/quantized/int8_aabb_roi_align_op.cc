#include "caffe2/operators/quantized/int8_aabb_roi_align_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {
namespace {

void precalculate_interpolation_coefficients(
    const size_t height,
    const size_t width,
    const size_t output_height,
    const size_t output_width,
    const size_t sampling_height,
    const size_t sampling_width,
    float roi_y1,
    float roi_x1,
    float output_stride_y,
    float output_stride_x,
    float sample_stride_y,
    float sample_stride_x,
    std::vector<std::pair<uint32_t, float>>::iterator output_iterator) {
  const float max_x = float(int32_t(width - 1));
  const float max_y = float(int32_t(height - 1));
  for (size_t oy = 0; oy < output_height; oy++) {
    for (size_t ox = 0; ox < output_width; ox++) {
      for (size_t sy = 0; sy < sampling_height; sy++) {
        const float yy = roi_y1 + float(int32_t(oy)) * output_stride_y +
            float(int32_t(sy)) * sample_stride_y;
        for (size_t sx = 0; sx < sampling_width; sx++) {
          const float xx = roi_x1 + float(int32_t(ox)) * output_stride_x +
              float(int32_t(sx)) * sample_stride_x;

          float x = xx;
          float y = yy;

          const int32_t x_trunc = int32_t(x);
          const int32_t y_trunc = int32_t(y);

          /*
           * Check if inverse elements are inside of feature map boundary, i.e.
           *   -1.0f < x < width AND -1.0 < y < height
           * This condition is equivalent to
           *   -1 < trunc(x) < width AND -1 < trunc(y) < height
           * or
           *    0 <= trunc(x) < width AND 0 <= trunc(y) < height
           * or
           *    0 <= x_trunc < width AND 0 <= y_trunc < height
           * or
           *    unsigned(x_trunc) < unsigned(width) AND unsigned(y_trunc) <
           * unsigned(height)
           */
          if (uint32_t(x_trunc) < uint32_t(width) &&
              uint32_t(y_trunc) < uint32_t(height)) {
            x = std::min<float>(std::max<float>(x, 0.0f), max_x);
            y = std::min<float>(std::max<float>(y, 0.0f), max_y);

            const int32_t y_lo = y_trunc;
            const int32_t x_lo = x_trunc;
            const int32_t y_hi =
                std::min<int32_t>(y_lo + 1, int32_t(height) - 1);
            const int32_t x_hi =
                std::min<int32_t>(x_lo + 1, int32_t(width) - 1);

            const float ly = y - float(y_lo);
            const float lx = x - float(x_lo);
            const float hy = 1.0f - ly;
            const float hx = 1.0f - lx;

            const float w1 = hy * hx;
            const float w2 = hy * lx;
            const float w3 = ly * hx;
            const float w4 = ly * lx;

            *output_iterator++ = std::pair<uint32_t, float>(
                uint32_t(y_lo) * width + uint32_t(x_lo), w1);
            *output_iterator++ = std::pair<uint32_t, float>(
                uint32_t(y_lo) * width + uint32_t(x_hi), w2);
            *output_iterator++ = std::pair<uint32_t, float>(
                uint32_t(y_hi) * width + uint32_t(x_lo), w3);
            *output_iterator++ = std::pair<uint32_t, float>(
                uint32_t(y_hi) * width + uint32_t(x_hi), w4);
          } else {
            *output_iterator++ = std::pair<uint32_t, float>(0, 0.0f);
            *output_iterator++ = std::pair<uint32_t, float>(0, 0.0f);
            *output_iterator++ = std::pair<uint32_t, float>(0, 0.0f);
            *output_iterator++ = std::pair<uint32_t, float>(0, 0.0f);
          }
        }
      }
    }
  }
}

void ROIAlignForward(
    const size_t num_images,
    const int32_t* batch_splits,
    const size_t num_rois,
    const uint8_t* input_ptr,
    const int32_t input_zero_point,
    const float input_scale,
    const float spatial_scale,
    const size_t channels,
    const size_t input_height,
    const size_t input_width,
    const size_t output_height,
    const size_t output_width,
    const size_t sampling_height_arg,
    const size_t sampling_width_arg,
    const uint16_t* rois_ptr,
    uint8_t* output_ptr,
    const int32_t output_zero_point,
    const float output_scale)
{
  const size_t input_pixels = input_height * input_width;
  const size_t output_pixels = output_height * output_width;

  std::vector<size_t> batch_starts(num_images);
  size_t batch_start = 0;
  for (size_t i = 0; i < num_images; i++) {
    batch_starts[i] = batch_start;
    batch_start += batch_splits[i];
  }

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t roi_idx = 0; roi_idx < num_rois; roi_idx++) {
    const size_t input_image_idx = *std::lower_bound(
        batch_starts.crbegin(),
        batch_starts.crend(),
        roi_idx,
        std::greater<size_t>());

    uint8_t* roi_output_ptr =
        output_ptr + roi_idx * channels * output_width * output_height;

    const uint8_t* roi_input_ptr =
        input_ptr + input_image_idx * channels * input_pixels;

    /* DO NOT USE ROUNDING: this implementation detail is CRITICAL */
    const float roi_x1 = float(rois_ptr[roi_idx * 4 + 0]) * 0.125f * spatial_scale;
    const float roi_y1 = float(rois_ptr[roi_idx * 4 + 1]) * 0.125f * spatial_scale;
    const float roi_x2 = float(rois_ptr[roi_idx * 4 + 2]) * 0.125f * spatial_scale;
    const float roi_y2 = float(rois_ptr[roi_idx * 4 + 3]) * 0.125f * spatial_scale;

    /* Malformed RoIs are forced to be 1x1 */
    const float roi_width = std::max(roi_x2 - roi_x1, 1.0f);
    const float roi_height = std::max(roi_y2 - roi_y1, 1.0f);
    const float output_stride_x = roi_width / float(int32_t(output_width));
    const float output_stride_y = roi_height / float(int32_t(output_height));

    size_t sampling_height = sampling_height_arg;
    if (sampling_height == 0) {
      sampling_height =
          size_t(ceilf(roi_height / float(int32_t(output_height))));
    }
    size_t sampling_width = sampling_width_arg;
    if (sampling_width == 0) {
      sampling_width = size_t(ceilf(roi_width / float(int32_t(output_width))));
    }
    const size_t sampling_pixels = sampling_height * sampling_width;
    const float sample_stride_x =
        roi_width / float(int32_t(output_width * sampling_width));
    const float sample_stride_y =
        roi_height / float(int32_t(output_height * sampling_height));

    const float normalization = input_scale / (output_scale * float(int32_t(sampling_pixels)));

    // we want to precalculate indeces and weights shared by all channels,
    // this is the key point of optimiation
    std::vector<std::pair<uint32_t, float>> interpolation_data(
        sampling_pixels * output_pixels * 4);
    precalculate_interpolation_coefficients(
        input_height,
        input_width,
        output_height,
        output_width,
        sampling_height,
        sampling_width,
        roi_y1 + 0.5f * sample_stride_y,
        roi_x1 + 0.5f * sample_stride_x,
        output_stride_y,
        output_stride_x,
        sample_stride_y,
        sample_stride_x,
        interpolation_data.begin());

    auto interpolation_iterator = interpolation_data.begin();

    for (size_t output_idx = 0; output_idx < output_pixels; output_idx++) {
      std::vector<float> accumulator(channels);

      for (size_t sample_idx = 0; sample_idx < sampling_pixels; sample_idx++) {
        const std::pair<uint32_t, float> cll = *interpolation_iterator++;
        const std::pair<uint32_t, float> chl = *interpolation_iterator++;
        const std::pair<uint32_t, float> clh = *interpolation_iterator++;
        const std::pair<uint32_t, float> chh = *interpolation_iterator++;

        for (size_t c = 0; c < channels; c++) {
          accumulator[c] += cll.second * float(int32_t(roi_input_ptr[channels * cll.first + c]) - input_zero_point);
          accumulator[c] += chl.second * float(int32_t(roi_input_ptr[channels * chl.first + c]) - input_zero_point);
          accumulator[c] += clh.second * float(int32_t(roi_input_ptr[channels * clh.first + c]) - input_zero_point);
          accumulator[c] += chh.second * float(int32_t(roi_input_ptr[channels * chh.first + c]) - input_zero_point);
        }
      }
      for (size_t c = 0; c < channels; c++) {
        const long output_value = lrintf(accumulator[c] * normalization) + long(output_zero_point);
        roi_output_ptr[output_idx * channels + c] = uint8_t(std::min<long>(std::max<long>(output_value, 0), 255));
      }
    }
  }
}

} // namespace

template <>
bool Int8AABBRoIAlignOp<CPUContext>::RunOnDevice() {
  const Tensor& batch_splits_tensor = Input(0);
  const int8::Int8TensorCPU& input_images_tensor =
      Inputs()[1]->template Get<int8::Int8TensorCPU>();
  const Tensor& input_rois_tensor = Input(2);

  int8::Int8TensorCPU* output_images_tensor =
    Outputs()[0]->template GetMutable<int8::Int8TensorCPU>();
  const int32_t output_zero_point =
    this->template GetSingleArgument<int>("Y_zero_point", 0);
  const float output_scale = this->template GetSingleArgument<float>("Y_scale", 1.0f);
  output_images_tensor->scale = output_scale;
  output_images_tensor->zero_point = output_zero_point;
  output_images_tensor->t.Resize(
    input_rois_tensor.dim32(0), output_height_, output_width_, input_images_tensor.t.dim32(3));

  if (input_rois_tensor.numel() == 0) {
    /* Handle empty RoIs */

    /* Note: output Tensor is inititalized with proper sizes and data type */
    return true;
  }

  CAFFE_ENFORCE_EQ(batch_splits_tensor.dim(), 1);
  const auto num_images = batch_splits_tensor.size(0);

  const int32_t* batch_splits_ptr = batch_splits_tensor.data<int32_t>();

  CAFFE_ENFORCE_EQ(input_images_tensor.t.size(0), num_images);
  CAFFE_ENFORCE_EQ(input_rois_tensor.dim(), 2);
  CAFFE_ENFORCE_EQ(input_rois_tensor.size(1), 4);
  const float spatial_scale = 1.0f / roi_stride_;

  ROIAlignForward(
      num_images,
      batch_splits_ptr,
      output_images_tensor->t.size(0),
      input_images_tensor.t.data<uint8_t>(),
      input_images_tensor.zero_point,
      input_images_tensor.scale,
      spatial_scale,
      input_images_tensor.t.dim32(3),
      input_images_tensor.t.dim32(1),
      input_images_tensor.t.dim32(2),
      output_height_,
      output_width_,
      sampling_height_,
      sampling_width_,
      input_rois_tensor.data<uint16_t>(),
      output_images_tensor->t.template mutable_data<uint8_t>(),
      output_images_tensor->zero_point,
      output_images_tensor->scale);

  return true;
}

REGISTER_CPU_OPERATOR(Int8AABBRoIAlign, Int8AABBRoIAlignOp<CPUContext>);

OPERATOR_SCHEMA(Int8AABBRoIAlign)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Region of Interest (RoI) align operation as used in Mask input_rois_tensor-CNN.
)DOC")
    .Arg(
        "order",
        "(string): order of dimensions of scores and deltas tensor. "
        "Only \"NHWC\" order is supported.")
    .Arg("Y_scale", "Scale quantization parameter for output images tensor.")
    .Arg("Y_zero_point", "Zero point quantization parameter for output images tensor.")
    .Arg(
        "roi_stride",
        "(float) default 1.0. Stride, in pixels, for RoI coordinates. "
        "RoI stride specifies how many RoI pixels correspond to a single "
        "pixel of the feature maps of input images. "
        "Typically it is set to the ratio of image size on the input of the "
        "network to the size of the input feature maps for the operator.")
    .Arg("output_height", "(int) default 1. Height of pooled output images.")
    .Arg("output_width", "(int) default 1. Width of pooled output images.")
    .Arg(
        "sampling_height",
        "(int) default 0. Number of vertical sampling points in the "
        "interpolation grid used to compute the value of each output pixel. "
        "If sampling_height = 0, then an adaptive number of grid points would "
        "be used (computed as ceil(roi_height / output_height).")
    .Arg(
        "sampling_width",
        "(int) default 0. Number of horizontal sampling points in the "
        "interpolation grid used to compute the value of each output pixel. "
        "If sampling_width = 0, then an adaptive number of grid points would "
        "be used (computed as ceil(roi_width / output_width).")
    .Input(
        0,
        "batch_splits",
        "Tensor of shape (num_images) with each element denoting the number "
        "of RoIs belonging to the corresponding image in batch")
    .Input(
        1,
        "images",
        "4D feature map input. "
        "If order is \"NHWC\" the shape is (num_images, input_height, input_width, num_channels).")
    .Input(
        2,
        "rois",
        "2D input of shape (num_rois, 4) specifying axis-aligned bounding boxes for regions of interest (RoI). "
        "The format of RoI is [x1, y1, x2, y2], there x2 >= x1 and y2 >= y1. "
        "The RoI coordinates are in the coordinate system of the input image to the neural network.")
    .Output(
        0,
        "images",
        "4D tensor of pooled regions of input feature map. "
        "If order is \"NHWC\", the shape is (num_rois, output_height, output_width, num_channels)."
        "The k-th batch element is a pooled feature map cooresponding to the k-th RoI.");

} // namespace caffe2
