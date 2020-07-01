#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/UpSample.h>
#include <ATen/Parallel.h>

namespace at {
namespace native {
namespace {

template <typename scalar_t>
static inline void compute_source_index_and_lambda(
    int64_t& input_index,
    scalar_t& lambda,
    scalar_t ratio,
    int64_t output_index,
    int64_t input_size,
    int64_t output_size,
    bool align_corners) {
  const scalar_t real_input_index = area_pixel_compute_source_index<scalar_t>(
    ratio, output_index, align_corners, /*cubic=*/true);
  input_index = floorf(real_input_index);
  lambda = real_input_index - input_index;
}


template <typename scalar_t, typename scale_type>
void cpu_upsample_bicubic(
    Tensor& output_,
    const Tensor& input_,
    bool align_corners,
    const scale_type& scales) {
  TORCH_CHECK(input_.dtype() == output_.dtype(), "expected dtype ", input_.dtype(),
              " for `output` but got dtype ", output_.dtype());
  auto input = input_.contiguous();
  auto output = output_.contiguous();
  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();
  auto input_sizes = input.sizes().vec();
  auto output_sizes = output.sizes().vec();
  auto ndim = input_sizes.size();

  // treat nbatch and channels as one dimension
  int64_t channels = input_sizes[0] * input_sizes[1];
  int64_t input_height = input_sizes[2];
  int64_t output_height = output_sizes[2];
  int64_t input_width = input_sizes[3];
  int64_t output_width = output_sizes[3];

  int64_t output_slice_size = output_height * output_width;

  // Special case: input/output same size, just copy
  if (input_height == output_height && input_width == output_width) {
    output_.copy_(input_);
    return;
  }

  auto loop2d = [&](int64_t begin, int64_t end) {
    const scalar_t height_scale = area_pixel_compute_scale<scalar_t>(
        input_height, output_height, align_corners, scales[0]);
    const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
        input_width, output_width, align_corners, scales[1]);

    for (int64_t c = begin; c < end; c++) {
      for (int64_t oh = 0; oh < output_height; oh++) {
        for (int64_t ow = 0; ow < output_width; ow++) {
          scalar_t* input_ptr = input_data + c * input_height * input_width;
          scalar_t* output_ptr = output_data + c * output_slice_size;

          const scalar_t real_iw = area_pixel_compute_source_index(width_scale, ow, align_corners, /*cubic=*/true);
          int64_t iw = floorf(real_iw);
          const scalar_t t_x = real_iw - iw;

          const scalar_t real_ih = area_pixel_compute_source_index(height_scale, oh, align_corners, /*cubic=*/true);
          int64_t ih = floorf(real_ih);
          const scalar_t t_y = real_ih - ih;

          // Interpolate 4 times in the x direction
          scalar_t coefficients[4];
          for (int64_t i = 0; i < 4; i++) {
            coefficients[i] = cubic_interp1d<scalar_t>(
                upsample_get_value_bounded<scalar_t>(
                    input_ptr, input_width, input_height, iw - 1, ih - 1 + i),
                upsample_get_value_bounded<scalar_t>(
                    input_ptr, input_width, input_height, iw + 0, ih - 1 + i),
                upsample_get_value_bounded<scalar_t>(
                    input_ptr, input_width, input_height, iw + 1, ih - 1 + i),
                upsample_get_value_bounded<scalar_t>(
                    input_ptr, input_width, input_height, iw + 2, ih - 1 + i),
                t_x);
          }
          
          // Interpolate in the y direction using x interpolations
          output_ptr[oh * output_width + ow] = cubic_interp1d<scalar_t>(
              coefficients[0],
              coefficients[1],
              coefficients[2],
              coefficients[3],
              t_y);
        }
      }
    }
  };

  at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 4, loop2d);

  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
}

using scale_t = std::vector<c10::optional<double>>;
void upsample_bicubic2d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto memory_format = input.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    std::cout << " TODO: cl path... " << std::endl;
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Byte, input.scalar_type(), "upsample_bicubic2d", [&] {
      cpu_upsample_bicubic<scalar_t, scale_t>(output, input, align_corners, {scales_h, scales_w});
    });
  }
}

} // anonymous namespace

REGISTER_DISPATCH(upsample_bicubic2d_kernel, &upsample_bicubic2d_kernel_impl);

} // namespace native
} // namespace at
