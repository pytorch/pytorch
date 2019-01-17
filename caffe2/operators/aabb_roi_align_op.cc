#include "caffe2/operators/aabb_roi_align_op.h"

#include "caffe2/utils/eigen_utils.h"
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
    const float* input_ptr,
    const float spatial_scale,
    const size_t channels,
    const size_t input_height,
    const size_t input_width,
    const size_t output_height,
    const size_t output_width,
    const size_t sampling_height_arg,
    const size_t sampling_width_arg,
    const float* rois_ptr,
    float* output_ptr,
    StorageOrder order) {
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

    float* roi_output_ptr =
        output_ptr + roi_idx * channels * output_width * output_height;

    const float* offset_bottom_rois = rois_ptr + roi_idx * 4;
    const float* roi_input_ptr =
        input_ptr + input_image_idx * channels * input_pixels;

    /* DO NOT USE ROUNDING: this implementation detail is CRITICAL */
    const float roi_x1 = offset_bottom_rois[0] * spatial_scale;
    const float roi_y1 = offset_bottom_rois[1] * spatial_scale;
    const float roi_x2 = offset_bottom_rois[2] * spatial_scale;
    const float roi_y2 = offset_bottom_rois[3] * spatial_scale;

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

    const float normalization = 1.0f / float(int32_t(sampling_pixels));

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

    if (order == StorageOrder::NCHW) {
      for (size_t c = 0; c < channels; c++) {
        for (size_t output_idx = 0; output_idx < output_pixels; output_idx++) {
          float output_value = 0.0f;
          for (size_t sample_idx = 0; sample_idx < sampling_pixels;
               sample_idx++) {
            const std::pair<uint32_t, float> cll = *interpolation_iterator++;
            const std::pair<uint32_t, float> chl = *interpolation_iterator++;
            const std::pair<uint32_t, float> clh = *interpolation_iterator++;
            const std::pair<uint32_t, float> chh = *interpolation_iterator++;
            output_value +=
                roi_input_ptr[c * input_pixels + cll.first] * cll.second +
                roi_input_ptr[c * input_pixels + chl.first] * chl.second +
                roi_input_ptr[c * input_pixels + clh.first] * clh.second +
                roi_input_ptr[c * input_pixels + chh.first] * chh.second;
          }
          roi_output_ptr[c * output_pixels + output_idx] =
              output_value * normalization;
        }
      }
    } else if (order == StorageOrder::NHWC) {
      for (size_t output_idx = 0; output_idx < output_pixels; output_idx++) {
        EVecXf output_values = EVecXf::Zero(channels);

        for (size_t sample_idx = 0; sample_idx < sampling_pixels;
             sample_idx++) {
          const std::pair<uint32_t, float> cll = *interpolation_iterator++;
          const std::pair<uint32_t, float> chl = *interpolation_iterator++;
          const std::pair<uint32_t, float> clh = *interpolation_iterator++;
          const std::pair<uint32_t, float> chh = *interpolation_iterator++;

          ConstEigenVectorMap<float> rowll(
              roi_input_ptr + channels * cll.first, channels);
          ConstEigenVectorMap<float> rowhl(
              roi_input_ptr + channels * chl.first, channels);
          ConstEigenVectorMap<float> rowlh(
              roi_input_ptr + channels * clh.first, channels);
          ConstEigenVectorMap<float> rowhh(
              roi_input_ptr + channels * chh.first, channels);

          output_values += cll.second * rowll + chl.second * rowhl +
              clh.second * rowlh + chh.second * rowhh;
        }
        output_values *= normalization;

        std::memcpy(
            roi_output_ptr + output_idx * channels,
            output_values.data(),
            channels * sizeof(float));
      }
    }
  }
}

} // namespace

template <>
bool AABBRoIAlignOp<CPUContext>::RunOnDevice() {
  const Tensor& batch_splits_tensor = Input(0);
  const Tensor& input_images_tensor = Input(1);
  const Tensor& input_rois_tensor = Input(2);

  if (input_rois_tensor.numel() == 0) {
    /* Handle empty RoIs */

    std::vector<int64_t> sizes;
    if (order_ == StorageOrder::NCHW) {
      sizes = {0, input_images_tensor.dim32(1), output_height_, output_width_};
    } else if (order_ == StorageOrder::NHWC) {
      sizes = {0, output_height_, output_width_, input_images_tensor.dim32(3)};
    }

    /* Note: output Tensor is inititalized with proper sizes and data type */
    Output(0, sizes, at::dtype<float>());
    return true;
  }

  CAFFE_ENFORCE_EQ(batch_splits_tensor.dim(), 1);
  const auto num_images = batch_splits_tensor.size(0);

  const int32_t* batch_splits_ptr = batch_splits_tensor.data<int32_t>();

  CAFFE_ENFORCE_EQ(input_images_tensor.size(0), num_images);
  CAFFE_ENFORCE_EQ(input_rois_tensor.dim(), 2);
  CAFFE_ENFORCE_EQ(input_rois_tensor.size(1), 4);
  const float spatial_scale = 1.0f / roi_stride_;

  if (order_ == StorageOrder::NCHW) {
    Tensor* output_images_tensor = Output(
        0,
        {input_rois_tensor.dim32(0),
         input_images_tensor.dim32(1),
         output_height_,
         output_width_},
        at::dtype<float>());
    ROIAlignForward(
        num_images,
        batch_splits_ptr,
        output_images_tensor->size(0),
        input_images_tensor.data<float>(),
        spatial_scale,
        input_images_tensor.dim32(1),
        input_images_tensor.dim32(2),
        input_images_tensor.dim32(3),
        output_height_,
        output_width_,
        sampling_height_,
        sampling_width_,
        input_rois_tensor.data<float>(),
        output_images_tensor->template mutable_data<float>(),
        order_);
  } else if (order_ == StorageOrder::NHWC) {
    Tensor* output_images_tensor = Output(
        0,
        {input_rois_tensor.dim32(0),
         output_height_,
         output_width_,
         input_images_tensor.dim32(3)},
        at::dtype<float>());
    ROIAlignForward(
        num_images,
        batch_splits_ptr,
        output_images_tensor->size(0),
        input_images_tensor.data<float>(),
        spatial_scale,
        input_images_tensor.dim32(3),
        input_images_tensor.dim32(1),
        input_images_tensor.dim32(2),
        output_height_,
        output_width_,
        sampling_height_,
        sampling_width_,
        input_rois_tensor.data<float>(),
        output_images_tensor->template mutable_data<float>(),
        order_);
  }

  return true;
}

REGISTER_CPU_OPERATOR(AABBRoIAlign, AABBRoIAlignOp<CPUContext>);

// Input: input_images_tensor, rois; Output: output_images_tensor
OPERATOR_SCHEMA(AABBRoIAlign)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Region of Interest (RoI) align operation as used in Mask input_rois_tensor-CNN.
)DOC")
    .Arg(
        "order",
        "(string): order of dimensions of scores and deltas tensor. "
        "Only \"NCHW\" (default) and \"NHWC\" order is supported.")
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
        "If order is \"NCHW\" the shape is (num_images, num_channels, input_height, input_width). "
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
        "If order is \"NCHW\", the shape is (num_rois, num_channels, output_height, output_width). "
        "If order is \"NHWC\", the shape is (num_rois, output_height, output_width, num_channels)."
        "The k-th batch element is a pooled feature map cooresponding to the k-th RoI.");

} // namespace caffe2
