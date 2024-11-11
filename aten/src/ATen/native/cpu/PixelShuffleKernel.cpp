#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cpu/PixelShuffleKernel.h>

#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>

namespace at::native {

namespace {

template <typename scalar_t>
void cpu_pixel_shuffle(
    TensorBase& output,
    const TensorBase& input,
    int64_t upscale_factor) {
  auto input_data = input.const_data_ptr<scalar_t>();
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

  // input tensor shape of [n, c, s1, s2, h, w]
  // output tensor shape of [n, c, h, s1, w, s2]
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t n{0}, c{0}, h{0}, s1{0}, w{0}, s2{0};
    data_index_init(begin, n, nbatch, c, sub_channels, h, height, s1, S, w, width, s2, S);

    for (const auto i : c10::irange(begin, end)) {
      int64_t input_offset = n * stride_n + c * stride_c + s1 * stride_s1 +
          s2 * stride_s2 + h * stride_h + w;
      output_data[i] = input_data[input_offset];

      data_index_step(n, nbatch, c, sub_channels, h, height, s1, S, w, width, s2, S);
    }
  });
}

template <typename scalar_t>
void cpu_pixel_shuffle_channels_last(
    TensorBase& output,
    const TensorBase& input,
    int64_t upscale_factor) {
  TORCH_CHECK(input.ndimension() == 4,
              "pixel shuffle with channels last format supports tensors with 4 dims");
  auto input_data = input.const_data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t height = input.size(2);
  int64_t width = input.size(3);
  int64_t sub_channels = channels / (upscale_factor * upscale_factor);
  int64_t S = upscale_factor;

  // input tensor shape of [n, h, w, c, s1, s2]
  // output tensor shape of [n, h, s1, w, s2, c]
  using Vec = vec::Vectorized<scalar_t>;
  at::parallel_for(0, nbatch * height, 0, [&](int64_t begin, int64_t end) {
    // temp buffer holding each channel lane
    auto buffer = std::make_unique<scalar_t []>(channels);
    scalar_t* buffer_ptr = buffer.get();

    int64_t n{0}, h{0};
    data_index_init(begin, n, nbatch, h, height);
    for (const auto i : c10::irange(begin, end)) {
      for (const auto w : c10::irange(width)) {
        const scalar_t* input_ptr = input_data + n * height * width * channels + h * width * channels + w * channels;

        // step 1: transpose each channel lane
        //   from: [c, s1*s2]
        //   to:   [s1*s2, c]
        utils::transpose(sub_channels, S * S, input_ptr, S * S, buffer_ptr, sub_channels);

        // step 2: copy from temp buffer to output
        for (const auto s1 : c10::irange(S)) {
          scalar_t* x_ptr = buffer_ptr + s1 * S * sub_channels;
          scalar_t* y_ptr = output_data + i * width * channels + s1 * width * S * sub_channels + w * S * sub_channels;

          int64_t size = S * sub_channels;
          int64_t d = 0;
          for (; d < size - (size % Vec::size()); d += Vec::size()) {
            Vec data_vec = Vec::loadu(x_ptr + d);
            data_vec.store(y_ptr + d);
          }
          for (; d < size; d++) {
            y_ptr[d] = x_ptr[d];
          }
        }
      }

      data_index_step(n, nbatch, h, height);
    }
  });
}

template <typename scalar_t>
void cpu_pixel_unshuffle(
    TensorBase& output,
    const TensorBase& input,
    int64_t downscale_factor) {
  auto input_data = input.const_data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  // [(B1...Bn), C, H, W] => [N, C, H, W]
  int64_t sub_channels = input.size(-3);
  int64_t height = input.size(-2) / downscale_factor;
  int64_t width = input.size(-1) / downscale_factor;
  int64_t channels = sub_channels * downscale_factor * downscale_factor;
  int64_t numel = input.numel();
  int64_t nbatch = numel / (channels * height * width);
  int64_t S = downscale_factor;

  // input strides
  int64_t stride_n = channels * height * width;
  int64_t stride_c = height * S * width * S;
  int64_t stride_h = S * width * S;
  int64_t stride_s1 = width * S;
  int64_t stride_w = S;
  int64_t stride_s2 = 1;

  // input tensor shape of [n, c, h, s1, w, s2]
  // output tensor shape of [n, c, s1, s2, h, w]
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t n{0}, c{0}, s1{0}, s2{0}, h{0}, w{0};
    data_index_init(begin, n, nbatch, c, sub_channels, s1, S, s2, S, h, height, w, width);

    for (const auto i : c10::irange(begin, end)) {
      int64_t input_offset = n * stride_n + c * stride_c + h * stride_h +
          s1 * stride_s1 + w * stride_w + s2 * stride_s2;
      output_data[i] = input_data[input_offset];

      data_index_step(n, nbatch, c, sub_channels, s1, S, s2, S, h, height, w, width);
    }
  });
}

template <typename scalar_t>
void cpu_pixel_unshuffle_channels_last(
    TensorBase& output,
    const TensorBase& input,
    int64_t downscale_factor) {
  TORCH_CHECK(input.ndimension() == 4,
              "pixel unshuffle with channels last format supports tensors with 4 dims");
  auto input_data = input.const_data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t nbatch = input.size(0);
  int64_t sub_channels = input.size(1);
  int64_t height = input.size(2) / downscale_factor;
  int64_t width = input.size(3) / downscale_factor;
  int64_t channels = sub_channels * downscale_factor * downscale_factor;
  int64_t numel = input.numel();
  int64_t S = downscale_factor;

  // input strides
  int64_t stride_n = height * width * channels;
  int64_t stride_h = S * width * S * sub_channels;
  int64_t stride_s1 = width * S * sub_channels;
  int64_t stride_w = S * sub_channels;
  int64_t stride_s2 = sub_channels;
  int64_t stride_c = 1;

  // input tensor shape of [n, h, s1, w, s2, c]
  // output tensor shape of [n, h, w, c, s1, s2]
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t n{0}, h{0}, w{0}, c{0}, s1{0}, s2{0};
    data_index_init(begin, n, nbatch, h, height, w, width, c, sub_channels, s1, S, s2, S);

    for (const auto i : c10::irange(begin, end)) {
      int64_t input_offset = n * stride_n + h * stride_h + s1 * stride_s1 +
          w * stride_w + s2 * stride_s2 + c * stride_c;
      output_data[i] = input_data[input_offset];

      data_index_step(n, nbatch, h, height, w, width, c, sub_channels, s1, S, s2, S);
    }
  });
}

void pixel_shuffle_kernel_impl(
    TensorBase& output,
    const TensorBase& input,
    int64_t upscale_factor) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
          input.scalar_type(), "pixel_shuffle", [&] {
        cpu_pixel_shuffle<scalar_t>(output, input, upscale_factor);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
          input.scalar_type(), "pixel_shuffle_channels_last", [&] {
        cpu_pixel_shuffle_channels_last<scalar_t>(output, input, upscale_factor);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void pixel_unshuffle_kernel_impl(
    TensorBase& output,
    const TensorBase& input,
    int64_t downscale_factor) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      // input tensor shape of [N, C, Hr, Wr]
      // output tensor shape of [N, Crr, H, W]
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
          input.scalar_type(), "pixel_unshuffle", [&] {
        cpu_pixel_unshuffle<scalar_t>(output, input, downscale_factor);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      // input tensor shape of [N, Hr, Wr, C]
      // output tensor shape of [N, H, W, Crr]
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
          input.scalar_type(), "pixel_unshuffle_channels_last", [&] {
        cpu_pixel_unshuffle_channels_last<scalar_t>(output, input, downscale_factor);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

} // anonymous namespace

REGISTER_DISPATCH(pixel_shuffle_kernel, &pixel_shuffle_kernel_impl)
REGISTER_DISPATCH(pixel_unshuffle_kernel, &pixel_unshuffle_kernel_impl)

} // at::native
