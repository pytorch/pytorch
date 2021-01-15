#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/native/cpu/PixelShuffleKernel.h>

namespace at { namespace native {

namespace {

template <typename scalar_t>
void cpu_pixel_shuffle(
    Tensor& output,
    const Tensor& input,
    int64_t upscale_factor) {
  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  // [(B1...Bn), C, H, W] => [N, C, H, W]
  int64_t channels = input.size(-3);
  int64_t height = input.size(-2);
  int64_t width = input.size(-1);
  int64_t sub_channels = channels / (upscale_factor * upscale_factor);
  int64_t numel = input.numel();
  int64_t nbatch = numel / (channels * height * width);
  int64_t S = upscale_factor;

  // input strides
  int64_t stride_n = channels * height * width;
  int64_t stride_c = S * S * height * width;
  int64_t stride_s1 = S * height * width;
  int64_t stride_s2 = height * width;
  int64_t stride_h = width;
  int64_t stride_w = 1;

  // input tensor shape of [n, c, s1, s2, h, w]
  // output tensor shape of [n, c, h, s1, w, s2]
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t n{0}, c{0}, h{0}, s1{0}, w{0}, s2{0};
    data_index_init(begin, n, nbatch, c, sub_channels, h, height, s1, S, w, width, s2, S);

    for (int64_t i = begin; i < end; i++) {
      int64_t input_offset = n * stride_n + c * stride_c + s1 * stride_s1 +
          s2 * stride_s2 + h * stride_h + w * stride_w;
      output_data[i] = input_data[input_offset];

      data_index_step(n, nbatch, c, sub_channels, h, height, s1, S, w, width, s2, S);
    }
  });
}

template <typename scalar_t>
void cpu_pixel_shuffle_channels_last(
    Tensor& output,
    const Tensor& input,
    int64_t upscale_factor) {
  TORCH_CHECK(input.ndimension() == 4,
              "pixel shuffle with channels last format supports tensors with 4 dims");
  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t height = input.size(2);
  int64_t width = input.size(3);
  int64_t sub_channels = channels / (upscale_factor * upscale_factor);
  int64_t numel = input.numel();
  int64_t S = upscale_factor;

  // input strides
  int64_t stride_n = height * width * channels;
  int64_t stride_h = width * channels;
  int64_t stride_w = channels;
  int64_t stride_c = S * S;
  int64_t stride_s1 = S;
  int64_t stride_s2 = 1;

  // input tensor shape of [n, h, w, c, s1, s2]
  // output tensor shape of [n, h, s1, w, s2, c]
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t n{0}, h{0}, s1{0}, w{0}, s2{0}, c{0};
    data_index_init(begin, n, nbatch, h, height, s1, S, w, width, s2, S, c, sub_channels);

    for (int64_t i = begin; i < end; i++) {
      int64_t input_offset = n * stride_n + h * stride_h + w * stride_w +
          c * stride_c + s1 * stride_s1 + s2 * stride_s2;
      output_data[i] = input_data[input_offset];

      data_index_step(n, nbatch, h, height, s1, S, w, width, s2, S, c, sub_channels);
    }
  });
}

template <typename scalar_t>
void cpu_pixel_shuffle_backward(
    Tensor& grad_input,
    const Tensor& grad_output,
    int64_t upscale_factor) {
  auto grad_input_data = grad_input.data_ptr<scalar_t>();
  auto grad_output_data = grad_output.data_ptr<scalar_t>();

  // [(B1...Bn), C, H, W] => [N, C, H, W]
  int64_t channels = grad_input.size(-3);
  int64_t height = grad_input.size(-2);
  int64_t width = grad_input.size(-1);
  int64_t sub_channels = channels / (upscale_factor * upscale_factor);
  int64_t numel = grad_input.numel();
  int64_t nbatch = numel / (channels * height * width);
  int64_t S = upscale_factor;

  // grad_output strides
  int64_t stride_n = channels * height * width;
  int64_t stride_c = height * S * width * S;
  int64_t stride_h = S * width * S;
  int64_t stride_s1 = width * S;
  int64_t stride_w = S;
  int64_t stride_s2 = 1;

  // grad_output tensor shape of [n, c, h, s1, w, s2]
  // grad_input tensor shape of [n, c, s1, s2, h, w]
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t n{0}, c{0}, s1{0}, s2{0}, h{0}, w{0};
    data_index_init(begin, n, nbatch, c, sub_channels, s1, S, s2, S, h, height, w, width);

    for (int64_t i = begin; i < end; i++) {
      int64_t output_offset = n * stride_n + c * stride_c + h * stride_h +
          s1 * stride_s1 + w * stride_w + s2 * stride_s2;
      grad_input_data[i] = grad_output_data[output_offset];

      data_index_step(n, nbatch, c, sub_channels, s1, S, s2, S, h, height, w, width);
    }
  });
}

template <typename scalar_t>
void cpu_pixel_shuffle_backward_channels_last(
    Tensor& grad_input,
    const Tensor& grad_output,
    int64_t upscale_factor) {
  TORCH_CHECK(grad_output.ndimension() == 4,
              "pixel shuffle with channels last format supports tensors with 4 dims");
  auto grad_input_data = grad_input.data_ptr<scalar_t>();
  auto grad_output_data = grad_output.data_ptr<scalar_t>();

  int64_t nbatch = grad_input.size(0);
  int64_t channels = grad_input.size(1);
  int64_t height = grad_input.size(2);
  int64_t width = grad_input.size(3);
  int64_t sub_channels = channels / (upscale_factor * upscale_factor);
  int64_t numel = grad_input.numel();
  int64_t S = upscale_factor;

  // grad_output strides
  int64_t stride_n = height * width * channels;
  int64_t stride_h = S * width * S * sub_channels;
  int64_t stride_s1 = width * S * sub_channels;
  int64_t stride_w = S * sub_channels;
  int64_t stride_s2 = sub_channels;
  int64_t stride_c = 1;

  // grad_output tensor shape of [n, h, s1, w, s2, c]
  // grad_input tensor shape of [n, h, w, c, s1, s2]
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t n{0}, h{0}, w{0}, c{0}, s1{0}, s2{0};
    data_index_init(begin, n, nbatch, h, height, w, width, c, sub_channels, s1, S, s2, S);

    for (int64_t i = begin; i < end; i++) {
      int64_t output_offset = n * stride_n + h * stride_h + s1 * stride_s1 +
          w * stride_w + s2 * stride_s2 + c * stride_c;
      grad_input_data[i] = grad_output_data[output_offset];

      data_index_step(n, nbatch, h, height, w, width, c, sub_channels, s1, S, s2, S);
    }
  });
}

void pixel_shuffle_kernel_impl(
    Tensor& output,
    const Tensor& input,
    int64_t upscale_factor) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "pixel_shuffle", [&] {
        cpu_pixel_shuffle<scalar_t>(output, input, upscale_factor);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "pixel_shuffle_channels_last", [&] {
        cpu_pixel_shuffle_channels_last<scalar_t>(output, input, upscale_factor);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void pixel_shuffle_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output,
    int64_t upscale_factor) {
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "pixel_shuffle_backward", [&] {
        cpu_pixel_shuffle_backward<scalar_t>(grad_input, grad_output, upscale_factor);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "pixel_shuffle_backward_channels_last", [&] {
        cpu_pixel_shuffle_backward_channels_last<scalar_t>(grad_input, grad_output, upscale_factor);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void pixel_unshuffle_kernel_impl(
    Tensor& output,
    const Tensor& input,
    int64_t downscale_factor) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      // input tensor shape of [N, C, Hr, Wr]
      // output tensor shape of [N, Crr, H, W]
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "pixel_unshuffle", [&] {
        cpu_pixel_shuffle_backward<scalar_t>(output, input, downscale_factor);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      // input tensor shape of [N, Hr, Wr, C]
      // output tensor shape of [N, H, W, Crr]
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "pixel_unshuffle_channels_last", [&] {
        cpu_pixel_shuffle_backward_channels_last<scalar_t>(output, input, downscale_factor);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void pixel_unshuffle_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output,
    int64_t downscale_factor) {
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      // grad_output tensor shape of [N, Crr, H, W]
      // grad_input tensor shape of [N, C, Hr, Wr]
      AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "pixel_unshuffle_backward", [&] {
        cpu_pixel_shuffle<scalar_t>(grad_input, grad_output, downscale_factor);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      // grad_output tensor shape of [N, H, W, Crr]
      // grad_input tensor shape of [N, Hr, Wr, C]
      AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "pixel_unshuffle_backward_channels_last", [&] {
        cpu_pixel_shuffle_channels_last<scalar_t>(grad_input, grad_output, downscale_factor);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

} // anonymous namespace

REGISTER_DISPATCH(pixel_shuffle_kernel, &pixel_shuffle_kernel_impl);
REGISTER_DISPATCH(pixel_shuffle_backward_kernel, &pixel_shuffle_backward_kernel_impl);
REGISTER_DISPATCH(pixel_unshuffle_kernel, &pixel_unshuffle_kernel_impl);
REGISTER_DISPATCH(pixel_unshuffle_backward_kernel, &pixel_unshuffle_backward_kernel_impl);

}} // at::native
